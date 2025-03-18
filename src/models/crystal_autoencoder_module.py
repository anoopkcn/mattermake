import torch
import torch.nn.functional as F
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm

from src.models.components.crystal_autoencoder import CrystalAutoencoderDiffusion


class CrystalAutoencoderModule(pl.LightningModule):
    def __init__(
        self,
        input_dim=95,           # Number of atom types
        hidden_dim=256,         # Hidden dimension size
        latent_dim=128,         # Latent space dimension
        encoder_layers=3,       # Number of layers in encoder
        num_heads=8,            # Number of attention heads
        timesteps=1000,         # Number of diffusion timesteps
        beta_start=1e-4,        # Starting value for noise schedule
        beta_end=0.02,          # Ending value for noise schedule
        beta_schedule="linear", # Type of noise schedule
        num_atom_types=95,      # Number of atom types
        discretize_t=50,        # Timestep to start discretizing atom types
        max_atoms=80,           # Maximum number of atoms in a unit cell
        learning_rate=1e-4,     # Learning rate
        weight_decay=1e-6,      # Weight decay for optimizer
        positions_loss_weight=1.0,  # Weight for positions loss
        lattice_loss_weight=1.0,    # Weight for lattice parameters loss
        atom_types_loss_weight=1.0, # Weight for atom types loss
        scheduler_step_size=100, # Learning rate scheduler step size
        scheduler_gamma=0.5,    # Learning rate decay factor
        device="cuda",          # Device to use
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create model
        self.model = CrystalAutoencoderDiffusion(self.hparams)

    def forward(self, x, t=None):
        return self.model(x, t)

    def training_step(self, batch, batch_idx):
        crystal = batch["crystal"]
        batch_size = crystal["positions"].shape[0]

        # Sample timesteps
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)

        # Forward pass
        outputs = self(crystal, t)

        # Calculate losses
        loss_dict = {}
        total_loss = 0

        for key in outputs["noise_pred"]:
            pred = outputs["noise_pred"][key]
            target = outputs["noise_target"][key]
            loss = F.mse_loss(pred, target)

            # Apply weights from config
            weight = getattr(self.hparams, f"{key}_loss_weight", 1.0)
            weighted_loss = weight * loss

            loss_dict[f"{key}_loss"] = loss
            total_loss += weighted_loss

        loss_dict["total_loss"] = total_loss

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"train/{name}", value, on_step=True, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch, batch_idx):
        crystal = batch["crystal"]
        batch_size = crystal["positions"].shape[0]

        # Sample timesteps
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)

        # Forward pass
        outputs = self(crystal, t)

        # Calculate losses
        loss_dict = {}
        total_loss = 0

        for key in outputs["noise_pred"]:
            pred = outputs["noise_pred"][key]
            target = outputs["noise_target"][key]
            loss = F.mse_loss(pred, target)

            # Apply weights from config
            weight = getattr(self.hparams, f"{key}_loss_weight", 1.0)
            weighted_loss = weight * loss

            loss_dict[f"{key}_loss"] = loss
            total_loss += weighted_loss

        loss_dict["total_loss"] = total_loss

        # Log losses
        for name, value in loss_dict.items():
            self.log(f"val/{name}", value, on_epoch=True, prog_bar=True)

        return total_loss

    def on_before_optimizer_step(self, optimizer):
        # Log gradient norms
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )

        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=self.hparams.scheduler_step_size,
            gamma=self.hparams.scheduler_gamma
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1
            }
        }

    def encode(self, x):
        """Encode crystal structure"""
        return self.model.encode(x)

    def decode(self, latent, num_atoms=None):
        """Generate crystal from latent vector"""
        return self.model.decode(latent, num_atoms)

    def interpolate(self, x1, x2, num_steps=10):
        """Interpolate between two crystal structures in latent space"""
        z1 = self.encode(x1)
        z2 = self.encode(x2)

        alphas = torch.linspace(0, 1, num_steps, device=self.device)
        crystals = []

        for alpha in alphas:
            z_interp = (1 - alpha) * z1 + alpha * z2
            crystal = self.decode(z_interp)
            crystals.append(crystal)

        return crystals

    @torch.no_grad()
    def generate(self, batch_size=1, num_atoms=None, condition=None):
        """
        Generate new crystal structures.

        Args:
            batch_size: Number of structures to generate
            num_atoms: Number of atoms per structure (defaults to max_atoms)
            condition: Optional latent vectors for conditioned generation

        Returns:
            Dictionary with generated crystal structures
        """
        if condition is None:
            # Sample from standard normal distribution
            condition = torch.randn(batch_size, self.hparams.latent_dim, device=self.device)

        return self.decode(condition, num_atoms)

    @torch.no_grad()
    def reconstruct(self, x):
        """Encode and decode a crystal structure"""
        latent = self.encode(x)
        return self.decode(latent)
