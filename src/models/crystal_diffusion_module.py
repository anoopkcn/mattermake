import torch
import lightning.pytorch as pl
from lightning.pytorch.utilities import grad_norm
from src.models.components.crystal_diffusion_model import CrystalDiffusionModel
from src.models.diffusion.diffusion_process import CrystalDiffusionProcess


class CrystalDiffusionLightningModule(pl.LightningModule):
    """
    Lightning module for training the crystal diffusion model.
    """
    def __init__(
        self,
        input_dim=95,           # Number of atom types
        hidden_dim=256,         # Hidden dimension size
        output_dim=95,          # Output dimension (same as input for atom types)
        num_layers=6,           # Number of layers in the model
        num_heads=8,            # Number of attention heads
        timesteps=1000,         # Number of diffusion timesteps
        beta_start=1e-4,        # Starting value for noise schedule
        beta_end=0.02,          # Ending value for noise schedule
        beta_schedule="linear", # Type of noise schedule
        use_conditioning=True,  # Whether to use conditioning
        condition_dim=256,      # Dimension of the condition embedding
        num_atom_types=95,      # Number of atom types
        discretize_t=50,        # Timestep to start discretizing atom types
        use_argmax_for_discretization=False,  # Whether to use argmax or soft discretization
        learning_rate=1e-4,     # Learning rate
        weight_decay=1e-6,      # Weight decay for optimizer
        positions_loss_weight=1.0,  # Weight for positions loss
        lattice_loss_weight=1.0,    # Weight for lattice parameters loss
        atom_types_loss_weight=1.0, # Weight for atom types loss
        scheduler_step_size=100, # Learning rate scheduler step size
        scheduler_gamma=0.5,    # Learning rate decay factor
        max_atoms=50,           # Maximum number of atoms in a unit cell
        device="cuda",          # Device to use
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # Create diffusion model
        self.model = CrystalDiffusionModel(self.hparams)

        # Create diffusion process
        self.diffusion = CrystalDiffusionProcess(self.hparams)

    def forward(self, x, t, condition=None):
        """Forward pass of the model."""
        return self.model(x, t, condition)

    def training_step(self, batch, batch_idx):
        """Training step."""
        crystal = batch["crystal"]
        condition = batch.get("condition")

        batch_size = crystal["positions"].shape[0]
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)

        loss_dict = self.diffusion.p_losses(
            self.model, crystal, t, condition=condition
        )

        for key, value in loss_dict.items():
            self.log(f"train/{key}_loss", value, on_step=True, on_epoch=True, prog_bar=True)

        return loss_dict["total"]

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        crystal = batch["crystal"]
        condition = batch.get("condition")

        batch_size = crystal["positions"].shape[0]
        t = torch.randint(0, self.hparams.timesteps, (batch_size,), device=self.device)

        loss_dict = self.diffusion.p_losses(
            self.model, crystal, t, condition=condition
        )

        for key, value in loss_dict.items():
            self.log(f"val/{key}_loss", value, on_epoch=True, prog_bar=True)

        return loss_dict["total"]

    def on_before_optimizer_step(self, optimizer):
        """Log gradient norms before optimizer step."""
        norms = grad_norm(self, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
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

    @torch.no_grad()
    def generate(self, batch_size=1, num_atoms=None, condition=None):
        """
        Generate new crystal structures.

        Args:
            batch_size: Number of structures to generate
            num_atoms: Number of atoms per structure (defaults to max_atoms)
            condition: Optional condition embedding for property guidance

        Returns:
            Dictionary with generated crystal structures
        """
        if num_atoms is None:
            num_atoms = self.hparams.max_atoms

        shape = (batch_size, num_atoms)

        return self.diffusion.p_sample_loop(self.model, shape, condition)

    @torch.no_grad()
    def generate_with_property(self, condition=None, num_samples=5, num_atoms=None):
        """
        Generate crystal structures with optional property conditioning.

        Args:
            condition: Optional conditioning tensor for property guidance
            num_samples: Number of structures to generate
            num_atoms: Number of atoms per structure (defaults to max_atoms)

        Returns:
            Generated crystal structures
        """
        generated_crystals = self.generate(
            batch_size=num_samples,
            num_atoms=num_atoms,
            condition=condition
        )

        return generated_crystals
