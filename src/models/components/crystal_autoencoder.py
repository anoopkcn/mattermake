import torch
import torch.nn as nn

from src.models.components.crystal_encoder import CrystalEncoder
from src.models.components.crystal_decoder import CrystalDecoder
from src.models.diffusion.diffusion_process import CrystalDiffusionProcess


class CrystalAutoencoderDiffusion(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.encoder = CrystalEncoder(
            input_dim=config.input_dim,
            hidden_dim=config.hidden_dim,
            latent_dim=config.latent_dim,
            num_layers=config.encoder_layers,
            num_heads=config.num_heads
        )

        self.decoder = CrystalDecoder(
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_atom_types=config.input_dim,
            max_atoms=config.max_atoms
        )

        # Diffusion process handler
        self.diffusion = CrystalDiffusionProcess(config)

    def encode(self, x):
        """Encode crystal structure to latent space"""
        return self.encoder(x)

    def decode(self, latent_code, num_atoms=None):
        """Generate crystal from latent code using diffusion process"""
        if num_atoms is None:
            num_atoms = self.config.max_atoms

        # Initial noise
        device = latent_code.device
        batch_size = latent_code.shape[0]

        # Start with random noise
        x = {
            'atom_types': torch.randn((batch_size, num_atoms, self.config.input_dim), device=device),
            'positions': torch.randn((batch_size, num_atoms, 3), device=device),
            'lattice': torch.randn((batch_size, 6), device=device)
        }

        # Reverse diffusion process
        for i in reversed(range(0, self.config.timesteps)):
            t = torch.full((batch_size,), i, device=device, dtype=torch.long)

            # Update x using diffusion process
            x = self.diffusion.p_sample(self.decoder, x, t, condition=latent_code)

        return self.diffusion._finalize_crystal(x)

    def forward(self, x, t=None):
        """
        Full forward pass for training.
        If t is provided, train the decoder with diffusion.
        If t is None, train the autoencoder reconstruction.
        """
        # Validate input shapes
        if 'atom_types' not in x or 'positions' not in x or 'lattice' not in x:
            raise ValueError("Input must contain atom_types, positions, and lattice")

        batch_size = x['atom_types'].shape[0]

        # Ensure all tensors have consistent batch size
        for key, value in x.items():
            if value.shape[0] != batch_size:
                raise ValueError(f"Inconsistent batch size in {key}: {value.shape[0]} vs {batch_size}")

        latent = self.encoder(x)

        if t is None:
            # Standard autoencoder training - encode and decode
            # For this we need to sample a low-noise timestep
            device = latent.device
            t = torch.ones((batch_size,), device=device, dtype=torch.long) * 10  # Low-noise timestep

        # Generate noisy sample
        x_noisy, noise_dict = self.diffusion.q_sample(x, t)

        # Predict noise
        noise_pred = self.decoder(x_noisy, t, latent)

        return {
            "noise_pred": noise_pred,
            "noise_target": noise_dict,
            "latent": latent
        }
