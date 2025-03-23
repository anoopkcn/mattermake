import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv


class QuotientGraphEncoder(nn.Module):
    def __init__(self, node_feat_dim, edge_feat_dim, hidden_dim, latent_dim):
        super().__init__()

        self.node_conv1 = GATConv(node_feat_dim, hidden_dim // 4, heads=4)
        self.node_conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4)
        # self.node_conv3 = GATConv(hidden_dim, hidden_dim // 2, heads=2)

        self.edge_embedding = nn.Sequential(
            nn.Linear(edge_feat_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.combined_linear1 = nn.Linear(hidden_dim * 2, hidden_dim * 2)
        self.combined_linear2 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.log_var = nn.Linear(hidden_dim, latent_dim)

    def forward(self, node_features, edge_index, edge_features):
        x = self.node_conv1(node_features, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.2, training=self.training)

        x = self.node_conv2(x, edge_index)
        x = F.relu(x)
        # x = F.dropout(x, p=0.2, training=self.training)

        # x = self.node_conv3(x, edge_index)
        # x = F.relu(x)

        edge_emb = self.edge_embedding(edge_features)

        graph_emb = torch.cat(
            [
                torch.mean(x, dim=0, keepdim=True),
                torch.mean(edge_emb, dim=0, keepdim=True),
            ],
            dim=1,
        )

        graph_emb = self.combined_linear1(graph_emb)
        graph_emb = F.relu(graph_emb)
        graph_emb = self.combined_linear2(graph_emb)
        graph_emb = F.relu(graph_emb)

        mu = self.mu(graph_emb)
        log_var = self.log_var(graph_emb)

        return mu, log_var

    def sample(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std


class QuotientGraphDecoder(nn.Module):
    def __init__(
        self, latent_dim, hidden_dim, node_feat_dim, edge_feat_dim, max_nodes=100
    ):
        super().__init__()
        self.max_nodes = max_nodes

        self.latent_to_hidden = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )

        self.node_hidden1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.node_hidden2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.node_features = nn.Linear(hidden_dim, node_feat_dim * max_nodes)

        self.edge_hidden1 = nn.Linear(hidden_dim, hidden_dim * 2)
        self.edge_hidden2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.edge_existence = nn.Linear(hidden_dim, max_nodes * max_nodes)
        self.edge_features = nn.Linear(hidden_dim, edge_feat_dim)

        self.num_nodes_hidden = nn.Linear(hidden_dim, hidden_dim)
        self.num_nodes = nn.Linear(hidden_dim, max_nodes)

        self.cell_hidden = nn.Linear(hidden_dim, hidden_dim // 2)
        self.cell_params = nn.Linear(hidden_dim // 2, 6)  # a, b, c, alpha, beta, gamma

    def forward(self, z):
        h = self.latent_to_hidden(z)
        h = F.relu(h)

        node_h = self.node_hidden1(h)
        node_h = F.relu(node_h)
        node_h = self.node_hidden2(node_h)
        node_h = F.relu(node_h)
        node_features_flat = self.node_features(node_h)
        node_features = node_features_flat.view(
            -1, self.max_nodes, node_features_flat.size(-1) // self.max_nodes
        )

        edge_h = self.edge_hidden1(h)
        edge_h = F.relu(edge_h)
        edge_h = self.edge_hidden2(edge_h)
        edge_h = F.relu(edge_h)
        edge_logits = self.edge_existence(edge_h)
        edge_logits = edge_logits.view(-1, self.max_nodes, self.max_nodes)

        num_nodes_h = self.num_nodes_hidden(h)
        num_nodes_h = F.relu(num_nodes_h)
        num_nodes_logits = self.num_nodes(num_nodes_h)

        cell_h = self.cell_hidden(h)
        cell_h = F.relu(cell_h)
        cell_params_raw = self.cell_params(cell_h)

        return {
            "node_features": node_features,
            "edge_logits": edge_logits,
            "num_nodes_logits": num_nodes_logits,
            "cell_params": cell_params_raw,
            "edge_h": h,
            "edge_feature_generator": self.edge_features,
        }

    def generate_edge_features(self, edge_indices, hidden_state):
        """Generate features for specific edges based on their indices."""
        num_edges = edge_indices.size(1)
        batch_size = hidden_state.size(0)
        edge_h = self.edge_hidden1(hidden_state)
        edge_h = F.relu(edge_h)
        edge_h = self.edge_hidden2(edge_h)
        edge_h = F.relu(edge_h)

        edge_hidden_expanded = edge_h.unsqueeze(1).expand(
            -1, num_edges, -1
        )  # [batch_size, num_edges, hidden_dim]

        edge_features = self.edge_features(
            edge_hidden_expanded
        )  # [batch_size, num_edges, edge_feat_dim]

        # If batch_size is 1, which is common when processing individual graphs
        if batch_size == 1:
            return edge_features.squeeze(0)  # [num_edges, edge_feat_dim]

        return edge_features


class QuotientGraphVAE(nn.Module):
    def __init__(
        self, node_feat_dim, edge_feat_dim, hidden_dim=128, latent_dim=64, max_nodes=50
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
