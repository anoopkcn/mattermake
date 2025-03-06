import math
import torch
import torch.nn as nn

from src.models.components.equivariant_unet import EquivariantUNet


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    Args:
        timesteps: 1-D Tensor of N indices, one per batch element.
        dim: The dimension of the output.
        max_period: Maximum period of the sinusoidal embedding.

    Returns:
        An [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class CrystalDiffusionModel(nn.Module):
    """
    Main diffusion model for crystal structure generation.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.backbone = EquivariantUNet(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            output_dim=config.output_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
        )

        if config.use_conditioning:
            self.condition_processor = nn.Sequential(
                nn.Linear(config.condition_dim, config.hidden_dim),
                nn.SiLU(),
                nn.Linear(config.hidden_dim, config.hidden_dim)
            )
        else:
            self.condition_processor = None

        self.time_embed = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim * 4),
            nn.SiLU(),
            nn.Linear(config.hidden_dim * 4, config.hidden_dim)
        )

    def forward(self, x, timesteps, condition=None):
        """
        Forward pass of the diffusion model.

        Args:
            x: Dictionary containing noisy crystal data with keys:
               - 'atom_types': [B, N] tensor of atom type indices
               - 'positions': [B, N, 3] tensor of atom coordinates
               - 'lattice': [B, 6] tensor of lattice parameters
            timesteps: [B] tensor of diffusion timesteps
            condition: Optional [B, C] tensor of property embeddings (DoS, pXRD)

        Returns:
            Dictionary with predicted noise for each component
        """
        t_emb = timestep_embedding(timesteps, self.config.hidden_dim)
        t_emb = self.time_embed(t_emb)

        cond_emb = None
        if condition is not None and self.condition_processor is not None:
            cond_emb = self.condition_processor(condition)

        return self.backbone(x, t_emb, cond_emb)
