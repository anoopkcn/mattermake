import torch.nn as nn

from src.models.components.equivariant_unet import EquivariantUNet
from src.models.components.crystal_diffusion_model import timestep_embedding


class CrystalDecoder(nn.Module):
    def __init__(self, latent_dim=128, hidden_dim=256, num_atom_types=95, max_atoms=80):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_atom_types = num_atom_types
        self.max_atoms = max_atoms

        # Process latent code
        self.latent_processor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

        # Use EquivariantUNet as the backbone
        self.backbone = EquivariantUNet(
            input_dim=num_atom_types,
            hidden_dim=hidden_dim,
            output_dim=num_atom_types,
            num_layers=4,
            num_heads=8,
        )

        # Time embedding for diffusion process
        self.time_embed = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x_noisy, t, latent_code):
        """
        Forward pass of the decoder.

        Args:
            x_noisy: Noisy crystal data with atom_types, positions, lattice
            t: Diffusion timesteps
            latent_code: Latent embedding from the encoder
        """
        # Process latent code
        cond_emb = self.latent_processor(latent_code)

        # Process timestep
        t_emb = timestep_embedding(t, self.hidden_dim)
        t_emb = self.time_embed(t_emb)

        # Get noise prediction from backbone
        noise_pred = self.backbone(x_noisy, t_emb, cond_emb)

        return noise_pred
