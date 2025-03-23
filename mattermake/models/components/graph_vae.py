import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATv2Conv


class NodeEncoder(nn.Module):
    def __init__(self, num_elements=100, coord_dim=3, embedding_dim=64, hidden_dim=128):
        super().__init__()

        self.element_embedding = nn.Embedding(num_elements, embedding_dim)

        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

        self.node_combiner = nn.Sequential(
            nn.Linear(embedding_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

    def forward(self, atomic_numbers, frac_coords):
        element_idx = torch.clamp(atomic_numbers - 1, min=0, max=99)

        element_features = self.element_embedding(element_idx)

        coord_features = self.coord_encoder(frac_coords)

        combined = torch.cat([element_features, coord_features], dim=-1)
        node_features = self.node_combiner(combined)

        return node_features


class EdgeEncoder(nn.Module):
    def __init__(self, edge_feat_dim=4, hidden_dim=128, embedding_dim=64):
        super().__init__()

        self.distance_encoder = nn.Sequential(
            nn.Linear(1, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, embedding_dim // 2),
        )

        self.period_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, embedding_dim // 2),
        )

        self.edge_combiner = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU()
        )

    def forward(self, edge_features):
        distances = edge_features[:, 0:1]
        periodicity = edge_features[:, 1:4]

        dist_embedding = self.distance_encoder(distances)
        period_embedding = self.period_encoder(periodicity)

        combined = torch.cat([dist_embedding, period_embedding], dim=-1)
        edge_embedding = self.edge_combiner(combined)

        return edge_embedding


class QuotientGraphEncoder(nn.Module):
    def __init__(
        self, node_feat_dim=103, edge_feat_dim=4, hidden_dim=256, latent_dim=128
    ):
        super().__init__()

        self.node_encoder = NodeEncoder(
            num_elements=100, coord_dim=3, embedding_dim=64, hidden_dim=hidden_dim
        )

        self.edge_encoder = EdgeEncoder(
            edge_feat_dim=edge_feat_dim, hidden_dim=hidden_dim, embedding_dim=64
        )

        self.conv1 = GATv2Conv(
            hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim
        )
        self.conv2 = GATv2Conv(
            hidden_dim, hidden_dim // 4, heads=4, edge_dim=hidden_dim
        )
        self.conv3 = GATv2Conv(
            hidden_dim, hidden_dim // 2, heads=2, edge_dim=hidden_dim
        )

        self.final_project = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
        )

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, node_features, edge_index, edge_features):
        atom_types = torch.argmax(node_features[:, :100], dim=1) + 1
        frac_coords = node_features[:, -3:]

        x = self.node_encoder(atom_types, frac_coords)
        edge_attr = self.edge_encoder(edge_features)

        x = F.relu(self.conv1(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv2(x, edge_index, edge_attr=edge_attr))
        x = F.dropout(x, p=0.2, training=self.training)

        x = F.relu(self.conv3(x, edge_index, edge_attr=edge_attr))

        global_x = torch.mean(x, dim=0, keepdim=True)
        global_edge = torch.mean(edge_attr, dim=0, keepdim=True)

        global_repr = torch.cat([global_x, global_edge], dim=1)
        global_repr = self.final_project(global_repr)

        mu = self.mu(global_repr)
        log_var = self.log_var(global_repr)

        return mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class QuotientGraphDecoder(nn.Module):
    def __init__(
        self,
        latent_dim=128,
        hidden_dim=256,
        node_feat_dim=103,
        edge_feat_dim=4,
        max_nodes=100,
    ):
        super().__init__()
        self.max_nodes = max_nodes

        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.SiLU(),
        )

        self.atom_type_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 100 * max_nodes),
        )

        self.coord_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3 * max_nodes),
        )

        self.edge_existence = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.SiLU(),
            nn.Linear(hidden_dim * 2, max_nodes * max_nodes),
        )

        self.distance_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
            nn.Softplus(),
        )

        self.periodicity_decoder = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 3),
            nn.Tanh(),
        )

        self.num_nodes = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, max_nodes),
        )

        self.cell_params = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, z):
        h = self.latent_to_hidden(z)
        batch_size = z.shape[0]

        atom_logits = self.atom_type_decoder(h)
        atom_logits = atom_logits.view(batch_size, self.max_nodes, 100)

        coords = self.coord_decoder(h)
        coords = coords.view(batch_size, self.max_nodes, 3)

        node_features = torch.cat([atom_logits, coords], dim=-1)

        edge_logits = self.edge_existence(h)
        edge_logits = edge_logits.view(batch_size, self.max_nodes, self.max_nodes)

        num_nodes_logits = self.num_nodes(h)

        cell_params_raw = self.cell_params(h)

        a, b, c = (
            F.softplus(cell_params_raw[:, 0:1]),
            F.softplus(cell_params_raw[:, 1:2]),
            F.softplus(cell_params_raw[:, 2:3]),
        )

        alpha = 30 + 120 * torch.sigmoid(cell_params_raw[:, 3:4])
        beta = 30 + 120 * torch.sigmoid(cell_params_raw[:, 4:5])
        gamma = 30 + 120 * torch.sigmoid(cell_params_raw[:, 5:6])

        cell_params = torch.cat([a, b, c, alpha, beta, gamma], dim=1)

        return {
            "node_features": node_features,
            "edge_logits": edge_logits,
            "num_nodes_logits": num_nodes_logits,
            "cell_params": cell_params,
            "edge_h": h,
        }

    def generate_edge_features(self, edge_indices, hidden_state):
        """Generate features for specific edges based on their indices."""
        num_edges = edge_indices.size(1)
        batch_size = hidden_state.size(0)

        edge_hidden_expanded = hidden_state.unsqueeze(1).expand(-1, num_edges, -1)

        distances = self.distance_decoder(edge_hidden_expanded)
        periodicity = self.periodicity_decoder(edge_hidden_expanded)

        edge_features = torch.cat([distances, periodicity], dim=-1)

        if batch_size == 1:
            return edge_features.squeeze(0)

        return edge_features


class QuotientGraphVAE(nn.Module):
    def __init__(
        self,
        node_feat_dim=103,
        edge_feat_dim=4,
        hidden_dim=256,
        latent_dim=128,
        max_nodes=100,
    ):
        super().__init__()

        self.encoder = QuotientGraphEncoder(
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            hidden_dim=hidden_dim,
            latent_dim=latent_dim,
        )

        self.decoder = QuotientGraphDecoder(
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            node_feat_dim=node_feat_dim,
            edge_feat_dim=edge_feat_dim,
            max_nodes=max_nodes,
        )

    def encode(self, node_features, edge_index, edge_features):
        mu, log_var = self.encoder(node_features, edge_index, edge_features)
        z = self.encoder.sample(mu, log_var)
        return z, mu, log_var

    def decode(self, z):
        return self.decoder(z)

    def forward(self, node_features, edge_index, edge_features):
        z, mu, log_var = self.encode(node_features, edge_index, edge_features)
        decoded = self.decode(z)
        return decoded, mu, log_var
