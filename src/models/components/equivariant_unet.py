import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from e3nn import o3
from e3nn.nn import BatchNorm


class EquivariantConv(MessagePassing):
    def __init__(self, input_dim, output_dim, edge_dim=None):
        super().__init__(aggr="mean")
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.irreps_sh = o3.Irreps.spherical_harmonics(2)
        self.sh_calc = o3.SphericalHarmonics(self.irreps_sh, normalize=True)
        sh_dim = self.irreps_sh.dim

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
        # Handle empty edge lists
        if edge_index.size(1) == 0:
            # Return input as is in case of no edges
            return x

        row, col = edge_index

        # Make sure indices are valid
        if torch.max(row) >= pos.size(0) or torch.max(col) >= pos.size(0):
            print(f"Warning: Invalid indices in edge_index. Max row: {torch.max(row).item()}, "
                  f"Max col: {torch.max(col).item()}, Pos size: {pos.size(0)}")
            valid_mask = (row < pos.size(0)) & (col < pos.size(0))
            if not valid_mask.all():
                row = row[valid_mask]
                col = col[valid_mask]
                if edge_attr is not None:
                    edge_attr = edge_attr[valid_mask]
                if row.size(0) == 0:  # If no valid edges remain
                    return x

        # Compute relative positions with explicit indexing to avoid out-of-bounds
        rel_pos = pos.index_select(0, row) - pos.index_select(0, col)

        # Compute edge lengths
        edge_length_squared = torch.sum(rel_pos ** 2, dim=1, keepdim=True)
        edge_length = torch.sqrt(edge_length_squared + 1e-12)  # Add epsilon for stability

        # Create unit direction vectors, handle zero lengths
        zero_length_mask = edge_length < 1e-10
        safe_edge_length = torch.where(zero_length_mask, torch.ones_like(edge_length), edge_length)
        directions = rel_pos / safe_edge_length

        # For zero-length edges, use a fixed direction
        if zero_length_mask.any():
            print(f"Warning: {zero_length_mask.sum().item()} zero-length edges detected")
            fixed_direction = torch.tensor([[1.0, 0.0, 0.0]], device=directions.device)
            directions = torch.where(
                zero_length_mask.expand_as(directions),
                fixed_direction.expand_as(directions),
                directions
            )

        # Custom spherical harmonics calculation with error handling
        try:
            # Apply manual normalization to ensure unit vectors
            norm = torch.norm(directions, dim=1, keepdim=True)
            directions = directions / (norm + 1e-10)

            # Ensure no NaNs
            directions = torch.nan_to_num(directions, nan=0.0, posinf=0.0, neginf=0.0)

            edge_sh = self.sh_calc(directions)
        except RuntimeError as e:
            print(f"Error in spherical harmonics calculation: {str(e)}")
            # Fall back to a simpler representation
            edge_sh = torch.zeros((directions.shape[0], self.irreps_sh.dim),
                                 device=directions.device)

        # Ensure no NaNs in edge_sh
        if torch.isnan(edge_sh).any():
            print("Warning: NaNs detected in edge_sh")
            edge_sh = torch.nan_to_num(edge_sh, nan=0.0, posinf=0.0, neginf=0.0)

        # Combine edge features
        if edge_attr is not None:
            # Check dimensions and fix if necessary
            if edge_sh.dim() != edge_attr.dim():
                print(f"Dimension mismatch: edge_sh.shape={edge_sh.shape}, edge_attr.shape={edge_attr.shape}")
                if edge_sh.dim() > edge_attr.dim():
                    # Unsqueeze edge_attr to match
                    while edge_attr.dim() < edge_sh.dim():
                        edge_attr = edge_attr.unsqueeze(-1)
                elif edge_sh.dim() < edge_attr.dim():
                    # Flatten additional dimensions of edge_attr
                    edge_attr = edge_attr.view(edge_attr.shape[0], -1)

            try:
                edge_features = torch.cat([edge_sh, edge_attr], dim=1)
            except RuntimeError as e:
                print(f"Error in cat operation: {str(e)}, using only edge_sh")
                edge_features = edge_sh
        else:
            edge_features = edge_sh

        # Transform node features
        x_transformed = self.node_mlp(x)

        # Message passing
        try:
            out = self.propagate(
                edge_index,
                x=x_transformed,
                edge_attr=self.edge_mlp(edge_features)
            )
        except RuntimeError as e:
            print(f"Error in propagate: {str(e)}")
            # Return original features if message passing fails
            return x

        # Combine with skip connection
        out = torch.cat([out, x], dim=1)
        out = self.output_mlp(out)

        # Apply batch norm
        try:
            batch_size = out.shape[0]
            out = out.reshape(batch_size, -1)  # flatten features
            out = self.norm(out)
        except RuntimeError as e:
            print(f"Error in batch norm: {str(e)}")
            # Skip batch norm if it fails
            pass

        return out

    def message(self, x_j, edge_attr):
        return x_j * edge_attr


# Custom implementation of scatter_mean to replace torch_scatter
def custom_scatter_mean(src, index, dim_size=None):
    """Safer implementation of scatter_mean"""
    # Handle empty input
    if src.numel() == 0 or index.numel() == 0:
        return torch.zeros(1, *src.shape[1:], device=src.device, dtype=src.dtype)

    # Determine output size
    if dim_size is None:
        if index.numel() == 0:
            dim_size = 1
        else:
            dim_size = int(index.max().item() + 1)

    # Handle the case where index has out-of-bounds values
    if torch.any(index < 0) or torch.any(index >= dim_size):
        print(f"Warning: Invalid scatter indices. Max: {index.max().item()}, Min: {index.min().item()}, Dim size: {dim_size}")
        valid_mask = (index >= 0) & (index < dim_size)
        if not valid_mask.all():
            # Only keep valid indices
            src = src[valid_mask]
            index = index[valid_mask]

        # If all indices are invalid, return zeros
        if src.numel() == 0:
            return torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)

    output = torch.zeros(dim_size, *src.shape[1:], device=src.device, dtype=src.dtype)
    count = torch.zeros(dim_size, device=src.device, dtype=torch.float32)

    # Expand index for scattering
    index_expanded = index.view(-1, *([1] * (src.dim() - 1)))
    index_expanded = index_expanded.expand_as(src)

    # Scatter add values
    output.scatter_add_(0, index_expanded, src)

    # Count values
    ones = torch.ones_like(index, device=src.device, dtype=torch.float32)
    count.scatter_add_(0, index, ones)

    # Avoid division by zero
    count = torch.clamp(count, min=1)
    count = count.view(dim_size, *([1] * (src.dim() - 1)))

    # Compute mean and handle NaNs
    output = output / count
    output = torch.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0)

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
        """
        Build crystal graph with error handling
        """
        try:
            pos = x['positions']
            lattice = x['lattice']

            # Check for NaN values
            if torch.isnan(pos).any() or torch.isnan(lattice).any():
                print("Warning: NaN values detected in positions or lattice")
                pos = torch.nan_to_num(pos, nan=0.0, posinf=0.0, neginf=0.0)
                lattice = torch.nan_to_num(lattice, nan=0.0, posinf=0.0, neginf=0.0)

            # Check for invalid lattice parameters
            if (lattice[:, :3] <= 0).any():
                print("Warning: Non-positive lattice parameters detected")
                # Ensure positive lattice parameters
                lattice = lattice.clone()
                lattice[:, :3] = torch.clamp(lattice[:, :3], min=1.0)

            cell = self._params_to_matrix(lattice)
            edge_index, edge_attr = self._get_periodic_edges(pos, cell, cutoff)

            return edge_index, edge_attr

        except Exception as e:
            print(f"Error in build_graph: {str(e)}")
            # Return empty edge list as fallback
            batch_size = x['positions'].shape[0]
            device = x['positions'].device

            # Create minimal valid graph (one self-loop per batch)
            rows = []
            cols = []
            attrs = []

            for b in range(batch_size):
                offset = b * x['positions'].shape[1]
                rows.append(offset)
                cols.append(offset)
                attrs.append(torch.zeros(3, device=device))

            edge_index = torch.tensor([rows, cols], device=device, dtype=torch.long)
            edge_attr = torch.stack(attrs, dim=0)

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

        # Use a smaller cutoff if the number of atoms is large to avoid memory issues
        if num_atoms > 50:
            adj_cutoff = min(cutoff, 4.0)
            print(f"Adjusting cutoff to {adj_cutoff} due to large structure size")
        else:
            adj_cutoff = cutoff

        try:
            for b in range(batch_size):
                pos_b = pos[b]
                cell_b = cell[b]
                batch_offset = b * num_atoms

                # Get atom mask (non-zero atoms)
                atom_mask = torch.any(torch.abs(pos_b) > 1e-6, dim=1)
                valid_atoms = torch.nonzero(atom_mask).squeeze(-1)

                # Skip batch if no valid atoms
                if valid_atoms.numel() == 0:
                    continue

                # Only process valid atoms
                valid_pos = pos_b[valid_atoms]

                # Build edges only for valid atoms
                for i_idx, i in enumerate(valid_atoms.tolist()):
                    for j_idx, j in enumerate(valid_atoms.tolist()):
                        if i == j:
                            continue

                        dist_vec = valid_pos[j_idx] - valid_pos[i_idx]

                        # Handle potential numerical issues in matrix inversion
                        try:
                            cell_inv = torch.inverse(cell_b)
                        except RuntimeError:
                            print(f"Warning: Singular matrix in batch {b}. Using pseudo-inverse.")
                            cell_inv = torch.pinverse(cell_b)

                        # Get minimum image convention
                        frac_dist = torch.matmul(dist_vec, cell_inv)
                        frac_dist = frac_dist - torch.round(frac_dist)
                        pbc_dist = torch.matmul(frac_dist, cell_b)

                        dist = torch.norm(pbc_dist)

                        if dist <= adj_cutoff and dist > 1e-10:  # Avoid self-loops with zero distance
                            rows.append(batch_offset + i)
                            cols.append(batch_offset + j)
                            edges.append(pbc_dist)  # Store 3D vector for edge attribute

            # Return edges or create a minimal valid graph if empty
            if len(rows) > 0:
                edge_index = torch.tensor([rows, cols], device=device, dtype=torch.long)
                edge_attr = torch.stack(edges, dim=0)  # Shape: [num_edges, 3]
            else:
                print("Warning: No edges found in the entire batch. Creating minimal graph.")
                # Create one dummy edge per batch for stability
                dummy_edges = []
                for b in range(batch_size):
                    batch_offset = b * num_atoms
                    # Add a self-loop on first atom
                    rows.append(batch_offset)
                    cols.append(batch_offset)
                    dummy_edges.append(torch.zeros(3, device=device))  # 3D vector

                edge_index = torch.tensor([rows, cols], device=device, dtype=torch.long)
                edge_attr = torch.stack(dummy_edges, dim=0)  # Shape: [num_edges, 3]

        except Exception as e:
            print(f"Error in _get_periodic_edges: {str(e)}")
            # Create a minimal valid graph
            rows = [0]
            cols = [0]
            edge_index = torch.tensor([rows, cols], device=device, dtype=torch.long)
            edge_attr = torch.zeros((1, 3), device=device)  # Shape: [1, 3]

        return edge_index, edge_attr

    def forward(self, x, t_emb, cond_emb=None):
        try:
            # Extract inputs
            atom_types = x['atom_types']
            positions = x['positions']
            lattice = x['lattice']

            # Apply embeddings
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

            # Build graph with error handling
            try:
                edge_index, edge_attr = self.build_graph(x)
            except Exception as e:
                print(f"Error building graph: {str(e)}")
                # Create a minimal valid graph
                edge_index = torch.zeros((2, 0), device=positions.device, dtype=torch.long)
                edge_attr = torch.zeros((0, 3), device=positions.device)

            # U-Net processing with error recovery
            skip_connections = []

            # Down blocks
            for i, down_block in enumerate(self.down_blocks):
                skip_connections.append(h)
                try:
                    h = down_block(h, edge_index, edge_attr, positions)
                except Exception as e:
                    print(f"Error in down_block {i}: {str(e)}")
                    # Fall back to previous features
                    h = skip_connections[-1]

            # Mid block
            try:
                h = self.mid_block(h, edge_index, edge_attr, positions)
            except Exception as e:
                print(f"Error in mid_block: {str(e)}")
                # Keep h as is

            # Up blocks
            for i, up_block in enumerate(self.up_blocks):
                try:
                    skip = skip_connections.pop()
                    h = torch.cat([h, skip], dim=-1)
                    h = up_block(h, edge_index, edge_attr, positions)
                except Exception as e:
                    print(f"Error in up_block {i}: {str(e)}")
                    # Keep h as is if error

            # Output heads
            atom_output = self.atom_type_head(h)
            position_output = self.position_head(h)

            # Global pooling with error handling
            try:
                batch_indices = torch.arange(batch_size, device=h.device).repeat_interleave(h.size(0) // batch_size)
                pooled = custom_scatter_mean(h, batch_indices, batch_size)
            except Exception as e:
                print(f"Error in global pooling: {str(e)}")
                # Fallback to simple mean
                pooled = h.reshape(batch_size, -1, h.size(-1)).mean(dim=1)

            # Add lattice features to pooled representation
            pooled = pooled + lattice_features.mean(dim=1)
            lattice_output = self.lattice_head(pooled)

            return {
                'atom_types': atom_output,
                'positions': position_output,
                'lattice': lattice_output
            }
        except Exception as e:
            print(f"Critical error in forward pass: {str(e)}")
            # Return empty outputs as fallback
            batch_size = x['atom_types'].shape[0]
            num_atoms = x['atom_types'].shape[1]

            # Create zero outputs with correct shapes
            atom_output = torch.zeros(batch_size, num_atoms, self.output_dim, device=x['atom_types'].device)
            position_output = torch.zeros(batch_size, num_atoms, 3, device=x['positions'].device)
            lattice_output = torch.zeros(batch_size, 6, device=x['lattice'].device)

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

        # Make sure the irreps string matches the output dimension
        irreps_str = f"{output_dim}x0e"
        self.irreps = o3.Irreps(irreps_str)

        self.norm1 = BatchNorm(self.irreps)
        self.norm2 = BatchNorm(self.irreps)

    def forward(self, x, edge_index, edge_attr, pos):
        try:
            h = self.conv1(x, edge_index, edge_attr, pos)
            h = self.norm1(h)
            h = F.silu(h)

            h = self.conv2(h, edge_index, edge_attr, pos)
            h = self.norm2(h)
            h = F.silu(h)

            h = self.attention(h)
        except Exception as e:
            print(f"Error in EquivariantBlock: {str(e)}")
            # Fall back to identity function to avoid breaking the entire model
            h = x

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
