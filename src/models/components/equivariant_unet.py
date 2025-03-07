import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from torch_scatter import scatter_mean
from e3nn import o3
from e3nn.nn import BatchNorm


class EquivariantConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim=None):
        super().__init__(aggr="mean")
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.sh = o3.Irreps.spherical_harmonics(2)
        sh_dim = self.sh.dim

        self.node_mlp = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

        edge_input_dim = sh_dim + (edge_dim if edge_dim is not None else 0)
        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

        self.output_mlp = nn.Sequential(
            nn.Linear(output_dim + input_dim, output_dim),
            nn.SiLU(),
            nn.Linear(output_dim, output_dim)
        )

        # Create an appropriate Irreps object for BatchNorm
        # This creates a scalar representation (l=0) with multiplicity=output_dim
        irreps_str = f"{output_dim}x0e"  # 'e' for even parity (scalar)
        self.irreps = o3.Irreps(irreps_str)
        self.norm = BatchNorm(self.irreps)

    def forward(self, x, edge_index, edge_attr=None, pos=None):
        row, col = edge_index
        rel_pos = pos[row] - pos[col]

        edge_length = torch.norm(rel_pos, dim=1, keepdim=True)

        directions = rel_pos / (edge_length + 1e-8)
        edge_sh = self.sh(directions)

        if edge_attr is not None:
            edge_features = torch.cat([edge_sh, edge_attr], dim=1)
        else:
            edge_features = edge_sh

        x_transformed = self.node_mlp(x)

        out = self.propagate(
            edge_index,
            x=x_transformed,
            edge_attr=self.edge_mlp(edge_features)
        )

        out = torch.cat([out, x], dim=1)
        out = self.output_mlp(out)

        # Modify this to work with the e3nn BatchNorm
        # We need to reshape for BatchNorm and reshape back
        batch_size = out.shape[0]
        out = out.reshape(batch_size, -1)  # flatten features
        out = self.norm(out)  # apply norm

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr


# Custom implementation of scatter_mean to replace torch_scatter
def custom_scatter_mean(src, index, dim_size=None):
    """
    Replacement for torch_scatter.scatter_mean

    Args:
        src: Source tensor
        index: Index tensor
        dim_size: Size of the output tensor

    Returns:
        Mean of elements in src grouped by index
    """
    if dim_size is None:
        dim_size = int(index.max()) + 1

    output = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=src.dtype)

    index_expanded = index.view(-1, *([1] * (src.dim() - 1)))
    index_expanded = index_expanded.expand_as(src)

    # Sum values
    output.scatter_add_(0, index_expanded, src)

    # Count values
    ones = torch.ones_like(index, device=src.device)
    count.scatter_add_(0, index, ones)

    # Avoid division by zero
    count = torch.clamp(count, min=1)
    count = count.view(dim_size, *([1] * (src.dim() - 1)))

    # Compute mean
    output = output / count

    return output


class EquivariantUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=4, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers

        self.atom_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Linear(3, hidden_dim)
        self.lattice_embedding = nn.Linear(6, hidden_dim)

        self.down_blocks = nn.ModuleList()
        for i in range(num_layers):
            block_dim = hidden_dim * (2 ** i)
            self.down_blocks.append(
                EquivariantBlock(block_dim, block_dim * 2, num_heads)
            )

        mid_dim = hidden_dim * (2 ** num_layers)
        self.mid_block = EquivariantBlock(mid_dim, mid_dim, num_heads)

        self.up_blocks = nn.ModuleList()
        for i in range(num_layers):
            block_dim = hidden_dim * (2 ** (num_layers - i))
            self.up_blocks.append(
                EquivariantBlock(block_dim * 2, block_dim // 2, num_heads)
            )

        self.atom_type_head = nn.Linear(hidden_dim, output_dim)
        self.position_head = nn.Linear(hidden_dim, 3)
        self.lattice_head = nn.Linear(hidden_dim, 6)

    def build_graph(self, x, cutoff=5.0):
        pos = x['positions']
        lattice = x['lattice']

        cell = self._params_to_matrix(lattice)

        edge_index, edge_attr = self._get_periodic_edges(pos, cell, cutoff)

        return edge_index, edge_attr

    def _params_to_matrix(self, lattice_params):
        a, b, c, alpha, beta, gamma = torch.split(lattice_params, 1, dim=-1)

        alpha = alpha * math.pi / 180
        beta = beta * math.pi / 180
        gamma = gamma * math.pi / 180

        v1 = torch.cat([a, torch.zeros_like(a), torch.zeros_like(a)], dim=-1)
        v2 = torch.cat([b * torch.cos(gamma), b * torch.sin(gamma), torch.zeros_like(b)], dim=-1)

        cx = torch.cos(beta)
        cy = (torch.cos(alpha) - torch.cos(beta) * torch.cos(gamma)) / torch.sin(gamma)
        cz = torch.sqrt(1 - cx**2 - cy**2)
        v3 = torch.cat([c * cx, c * cy, c * cz], dim=-1)

        cell = torch.stack([v1, v2, v3], dim=-2)
        return cell

    def _get_periodic_edges(self, pos, cell, cutoff):
        batch_size, num_atoms, _ = pos.shape
        device = pos.device

        rows = []
        cols = []
        edges = []

        use_gpu_accel = device.type == 'cuda' and num_atoms > 50

        for b in range(batch_size):
            pos_b = pos[b]
            cell_b = cell[b]
            batch_offset = b * num_atoms

            if use_gpu_accel:
                i_indices = torch.arange(num_atoms, device=device)
                j_indices = torch.arange(num_atoms, device=device)

                i_expanded = i_indices.repeat(num_atoms)
                j_expanded = j_indices.repeat_interleave(num_atoms)

                mask = i_expanded != j_expanded
                i_filtered = i_expanded[mask]
                j_filtered = j_expanded[mask]

                disp_vecs = pos_b[j_filtered] - pos_b[i_filtered]

                cell_inv = torch.inverse(cell_b)
                frac_dists = torch.matmul(disp_vecs, cell_inv)

                frac_dists = frac_dists - torch.round(frac_dists)

                pbc_dists = torch.matmul(frac_dists, cell_b)
                dists = torch.norm(pbc_dists, dim=1)

                cutoff_mask = dists <= cutoff
                keep_i = i_filtered[cutoff_mask]
                keep_j = j_filtered[cutoff_mask]
                keep_edges = pbc_dists[cutoff_mask]

                rows.extend((batch_offset + keep_i).tolist())
                cols.extend((batch_offset + keep_j).tolist())
                edges.extend(keep_edges)
            else:
                for i in range(num_atoms):
                    for j in range(num_atoms):
                        if i == j:
                            continue

                        dist_vec = pos_b[j] - pos_b[i]

                        frac_dist = torch.matmul(dist_vec, torch.inverse(cell_b))

                        frac_dist = frac_dist - torch.round(frac_dist)

                        pbc_dist = torch.matmul(frac_dist, cell_b)
                        dist = torch.norm(pbc_dist)

                        if dist <= cutoff:
                            rows.append(batch_offset + i)
                            cols.append(batch_offset + j)
                            edges.append(pbc_dist)

        if len(rows) > 0:
            edge_index = torch.tensor([rows, cols], device=device, dtype=torch.long)
            edge_attr = torch.stack(edges, dim=0)
        else:
            edge_index = torch.zeros((2, 0), device=device, dtype=torch.long)
            edge_attr = torch.zeros((0, 3), device=device)

        return edge_index, edge_attr

    def forward(self, x, t_emb, cond_emb=None):
        atom_types = x['atom_types']
        positions = x['positions']
        lattice = x['lattice']

        atom_features = self.atom_embedding(atom_types)
        pos_features = self.pos_embedding(positions)
        lattice_features = self.lattice_embedding(lattice)

        h = atom_features + pos_features

        batch_size = atom_types.shape[0]
        t_emb = t_emb.view(batch_size, 1, -1)

        if cond_emb is not None:
            cond_emb = cond_emb.view(batch_size, 1, -1)
            h = h + t_emb + cond_emb
        else:
            h = h + t_emb

        edge_index, edge_attr = self.build_graph(x)

        skip_connections = []

        for down_block in self.down_blocks:
            skip_connections.append(h)
            h = down_block(h, edge_index, edge_attr, positions)

        h = self.mid_block(h, edge_index, edge_attr, positions)

        for i, up_block in enumerate(self.up_blocks):
            skip = skip_connections.pop()
            h = torch.cat([h, skip], dim=-1)
            h = up_block(h, edge_index, edge_attr, positions)

        atom_output = self.atom_type_head(h)
        position_output = self.position_head(h)

        # Use our custom scatter_mean instead of torch_scatter.scatter_mean
        batch_indices = torch.arange(batch_size, device=h.device).repeat_interleave(h.size(0) // batch_size)
        pooled = custom_scatter_mean(h, batch_indices, batch_size)

        pooled = pooled + lattice_features.mean(dim=1)
        lattice_output = self.lattice_head(pooled)

        return {
            'atom_types': atom_output,
            'positions': position_output,
            'lattice': lattice_output
        }


class EquivariantBlock(nn.Module):
    def __init__(self, input_dim, output_dim, num_heads):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = EquivariantConv(input_dim, output_dim)
        self.conv2 = EquivariantConv(output_dim, output_dim)

        self.attention = EquivariantAttention(output_dim, num_heads)

        irreps_str = f"{output_dim}x0e"
        self.irreps = o3.Irreps(irreps_str)

        self.norm1 = BatchNorm(self.irreps)
        self.norm2 = BatchNorm(self.irreps)

    def forward(self, x, edge_index, edge_attr, pos):
        h = self.conv1(x, edge_index, edge_attr, pos)
        h = self.norm1(h)
        h = F.silu(h)

        h = self.conv2(h, edge_index, edge_attr, pos)
        h = self.norm2(h)
        h = F.silu(h)

        h = self.attention(h)

        return h


class EquivariantAttention(nn.Module):
    def __init__(self, dim, num_heads):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        assert dim % num_heads == 0, "Dimension must be divisible by number of heads"

        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = self.k_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = self.v_proj(x).reshape(batch_size, seq_len, self.num_heads, self.head_dim).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = F.softmax(scores, dim=-1)

        out = torch.matmul(attn, v)

        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.dim)
        out = self.out_proj(out)

        return out
