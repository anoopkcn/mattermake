import torch
import torch.nn as nn

from src.models.components.equivariant_unet import EquivariantBlock
from src.models.components.equivariant_unet import EquivariantUNet


class CrystalEncoder(nn.Module):
    def __init__(self, input_dim=95, hidden_dim=256, latent_dim=128, num_layers=3, num_heads=8):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.atom_embedding = nn.Embedding(input_dim, hidden_dim)
        self.pos_embedding = nn.Linear(3, hidden_dim)
        self.lattice_embedding = nn.Linear(6, hidden_dim)
        self.graph_builder = EquivariantUNet(input_dim, hidden_dim, input_dim)

        # Stack of equivariant graph layers
        self.graph_layers = nn.ModuleList([
            EquivariantBlock(
                hidden_dim if i == 0 else hidden_dim * (2**min(i, 2)),
                hidden_dim * (2**min(i+1, 3)),
                num_heads
            ) for i in range(num_layers)
        ])

        # Global pooling and projection to latent space
        final_dim = hidden_dim * (2**min(num_layers, 3))
        self.latent_projection = nn.Sequential(
            nn.Linear(final_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, x):
        # Extract components
        atom_types = x['atom_types']
        positions = x['positions']
        lattice = x['lattice']
        atom_mask = x.get('atom_mask', None)

        # Embed inputs
        if atom_types.dtype == torch.long:
            atom_features = self.atom_embedding(atom_types)
        else:
            # Handle one-hot inputs
            atom_features = torch.matmul(atom_types, self.atom_embedding.weight)

        pos_features = self.pos_embedding(positions)
        lattice_features = self.lattice_embedding(lattice).unsqueeze(1)  # [B, 1, H]

        # Initial node features
        h = atom_features + pos_features

        # Add global lattice features to each atom
        h = h + lattice_features

        # Build crystal graph
        edge_index, edge_attr = self._build_graph(x)

        # Process through equivariant layers
        for layer in self.graph_layers:
            h = layer(h, edge_index, edge_attr, positions)

        # Apply mask if provided
        if atom_mask is not None:
            mask = atom_mask.unsqueeze(-1).float()  # [B, N, 1]
            h = h * mask

        # Global pooling (mean of node features)
        if atom_mask is not None:
            # Masked mean
            mask_sum = torch.sum(mask, dim=1)
            global_features = torch.sum(h * mask.unsqueeze(-1), dim=1) / (mask_sum + 1e-10)
        else:
            global_features = h.mean(dim=1)

        # Project to latent space
        latent = self.latent_projection(global_features)

        return latent

    def _build_graph(self, x):
        return self.graph_builder.build_graph(x)
