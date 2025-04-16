import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import math
from torch.distributions import VonMises, Normal
from typing import Dict, Any

from mattermake.models.hct_base import HierarchicalCrystalTransformerBase
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HierarchicalCrystalTransformer(pl.LightningModule):
    """
    Hierarchical Crystal Transformer with dedicated encoders/projectors,
    cross-attention for atom decoding, and distributional outputs for
    lattice (Normal) and coordinates (Von Mises). Handles space group
    as a categorical distribution.

    Assumes PAD=0, START=-1, END=-2 token scheme and valid indices >= 1
    in the preprocessed data.
    """

    def __init__(
        self,
        # --- Vocabularies & Indices ---
        element_vocab_size: int = 100,  # Size of composition count vector input
        sg_vocab_size: int = 231,  # 1-230 -> map to 0-230 for embedding/pred (0=PAD/unused?)
        wyckoff_vocab_size: int = 200,  # Max wyckoff index + 1 (assuming indices >= 1)
        pad_idx: int = 0,
        start_idx: int = -1,
        end_idx: int = -2,
        # --- Model Dimensions ---
        d_model: int = 256,
        sg_embed_dim: int = 64,  # Embedding dim for SG before projection
        type_embed_dim: int = 64,
        wyckoff_embed_dim: int = 64,
        # --- Transformer Params ---
        nhead: int = 8,
        num_comp_encoder_layers: int = 2,  # Encoder for composition
        num_atom_decoder_layers: int = 4,  # Decoder for atom sequence
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # --- Control Flags ---
        condition_lattice_on_sg: bool = True,
        # --- Optimizer & Loss Params ---
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        sg_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,  # NLL weight
        type_loss_weight: float = 1.0,
        wyckoff_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,  # NLL weight
        eps: float = 1e-6,  # Small value for stability (e.g., variance)
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        log.info("Initializing HierarchicalCrystalTransformer")
        
        # Initialize the base transformer model
        log.info(f"Creating base model with d_model={d_model}, nhead={nhead}")
        self.model = HierarchicalCrystalTransformerBase(
            element_vocab_size=element_vocab_size,
            sg_vocab_size=sg_vocab_size,
            wyckoff_vocab_size=wyckoff_vocab_size,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            d_model=d_model,
            sg_embed_dim=sg_embed_dim,
            type_embed_dim=type_embed_dim,
            wyckoff_embed_dim=wyckoff_embed_dim,
            nhead=nhead,
            num_comp_encoder_layers=num_comp_encoder_layers,
            num_atom_decoder_layers=num_atom_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            condition_lattice_on_sg=condition_lattice_on_sg,
            eps=eps,
            **kwargs
        )

        # --- Loss Functions (Modified for NLL) ---
        log.info("Setting up loss functions")
        # Note: Target mapping for SG loss needs to align with sg_head output range
        self.sg_loss = nn.CrossEntropyLoss()  # Assumes head predicts logits for 0..N
        # Lattice/Coord loss is now NLL, calculated in training_step
        self.type_loss = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_idx)
        self.wyckoff_loss = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_idx)
        # Coord loss needs target coord transformation: [0,1) -> [-pi, pi)
        log.info("HierarchicalCrystalTransformer initialization complete")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass using the base model"""
        return self.model(batch)

    def _map_indices_for_embedding(
        self, indices: torch.Tensor, is_type: bool
    ) -> torch.Tensor:
        """Maps PAD, START, END indices to non-negative indices for embedding lookup."""
        return self.model._map_indices_for_embedding(indices, is_type)

    def calculate_loss(
        self, predictions: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Calculates the combined loss using NLL for continuous variables."""
        losses = {}
        total_loss = torch.tensor(
            0.0, device=self.device
        )  # Ensure loss is on correct device

        # --- Targets ---
        # Map SG target 1-230 -> 0-229 (assuming head predicts this range)
        sg_target = batch["spacegroup"].squeeze(1) - 1
        lattice_target = batch["lattice"]  # (b, 6)
        # Atom targets: [Atom1, ..., AtomN, END] (exclude START)
        atom_types_target_raw = batch["atom_types"][:, 1:]  # (b, T-1)
        atom_wyckoffs_target_raw = batch["atom_wyckoffs"][:, 1:]  # (b, T-1)
        atom_coords_target = batch["atom_coords"][:, 1:, :]  # (b, T-1, 3) in [0, 1)
        atom_mask_target = batch["atom_mask"][:, 1:]  # (b, T-1)

        # Map target indices for discrete losses (map -1, -2 -> positive embedding indices)
        type_target_mapped = self._map_indices_for_embedding(
            atom_types_target_raw, is_type=True
        )
        wyckoff_target_mapped = self._map_indices_for_embedding(
            atom_wyckoffs_target_raw, is_type=False
        )

        # --- SG Loss (Cross Entropy) ---
        # Ensure target indices are valid for the logits shape
        loss_sg = self.sg_loss(predictions["sg_logits"], sg_target)
        losses["sg_loss"] = loss_sg * self.hparams.sg_loss_weight
        total_loss += losses["sg_loss"]

        # --- Lattice Loss (NLL Normal) ---
        lattice_mean = predictions["lattice_mean"]
        lattice_log_var = predictions["lattice_log_var"]
        # Ensure variance is positive and stable
        lattice_var = torch.exp(lattice_log_var) + self.hparams.eps
        lattice_std = torch.sqrt(lattice_var)
        # Use independent Normal distributions
        lattice_dist = Normal(lattice_mean, lattice_std)
        nll_lattice = -lattice_dist.log_prob(lattice_target)  # (b, 6)
        # Average NLL over batch and 6 parameters
        nll_lattice = nll_lattice.mean()
        losses["lattice_nll"] = nll_lattice * self.hparams.lattice_loss_weight
        total_loss += losses["lattice_nll"]

        # --- Atom Type Loss (Cross Entropy) ---
        type_logits_flat = predictions["type_logits"].reshape(
            -1, self.model.effective_type_vocab_size
        )
        type_target_flat = type_target_mapped.reshape(-1)
        loss_type = self.type_loss(
            type_logits_flat, type_target_flat
        )  # Handles ignore_index=PAD
        losses["type_loss"] = loss_type * self.hparams.type_loss_weight
        total_loss += losses["type_loss"]

        # --- Atom Wyckoff Loss (Cross Entropy) ---
        wyckoff_logits_flat = predictions["wyckoff_logits"].reshape(
            -1, self.model.effective_wyckoff_vocab_size
        )
        wyckoff_target_flat = wyckoff_target_mapped.reshape(-1)
        loss_wyckoff = self.wyckoff_loss(
            wyckoff_logits_flat, wyckoff_target_flat
        )  # Handles ignore_index=PAD
        losses["wyckoff_loss"] = loss_wyckoff * self.hparams.wyckoff_loss_weight
        total_loss += losses["wyckoff_loss"]

        # --- Coordinate Loss (NLL VonMises) ---
        coord_loc = predictions["coord_loc"]  # (b, T-1, 3) in [-pi, pi]
        coord_concentration = predictions["coord_concentration"]  # (b, T-1, 3) > 0
        # Transform target coordinates [0, 1) -> angles [-pi, pi)
        target_coords_angle = (
            atom_coords_target * 2 * math.pi
        ) - math.pi  # (b, T-1, 3)

        # Create VonMises distribution
        # Ensure concentration is detached if needed, though usually not required for log_prob
        coord_dist = VonMises(loc=coord_loc, concentration=coord_concentration)
        # Calculate log probability (NLL = -log_prob)
        nll_coord_unmasked = -coord_dist.log_prob(target_coords_angle)  # (b, T-1, 3)

        # Mask the NLL based on valid target atoms (exclude padding)
        mask_expanded = atom_mask_target.unsqueeze(-1).float()  # (b, T-1, 1)
        nll_coord_masked = nll_coord_unmasked * mask_expanded  # (b, T-1, 3)

        # Average over non-masked elements (valid atoms and coordinates)
        num_valid = mask_expanded.sum() * 3  # Multiply by 3 for x, y, z dimensions
        nll_coord = (
            nll_coord_masked.sum() / num_valid
            if num_valid > 0
            else torch.tensor(0.0, device=self.device)
        )
        losses["coord_nll"] = nll_coord * self.hparams.coord_loss_weight
        total_loss += losses["coord_nll"]

        losses["total_loss"] = total_loss
        return losses

    # --- Lightning Steps (training_step, validation_step, test_step) ---
    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        log.debug(f"Training batch {batch_idx}")
        predictions = self(batch)
        losses = self.calculate_loss(predictions, batch)
        batch_size = batch["composition"].size(0)
        log_opts = {
            "on_step": True,
            "on_epoch": True,
            "prog_bar": False,
            "batch_size": batch_size,
        }
        self.log(
            "train/total_loss",
            losses["total_loss"],
            prog_bar=True,
            batch_size=batch_size,
        )
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items() if k != "total_loss"},
            **log_opts,
        )
        if batch_idx % 100 == 0:
            log.info(f"Training batch {batch_idx}, loss: {losses['total_loss']:.4f}")
        return losses["total_loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        log.debug(f"Validation batch {batch_idx}")
        predictions = self(batch)
        losses = self.calculate_loss(predictions, batch)
        batch_size = batch["composition"].size(0)
        log_opts = {"on_step": False, "on_epoch": True, "batch_size": batch_size}
        self.log_dict({f"val/{k}": v for k, v in losses.items()}, **log_opts)
        if batch_idx % 100 == 0:
            log.info(f"Validation batch {batch_idx}, loss: {losses['total_loss']:.4f}")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        log.debug(f"Test batch {batch_idx}")
        predictions = self(batch)
        losses = self.calculate_loss(predictions, batch)
        batch_size = batch["composition"].size(0)
        log_opts = {"on_step": False, "on_epoch": True, "batch_size": batch_size}
        self.log_dict({f"test/{k}": v for k, v in losses.items()}, **log_opts)
        if batch_idx % 100 == 0:
            log.info(f"Test batch {batch_idx}, loss: {losses['total_loss']:.4f}")

    # --- Optimizer ---
    def configure_optimizers(self) -> optim.Optimizer:
        log.info(f"Configuring optimizer with lr={self.hparams.learning_rate}, weight_decay={self.hparams.weight_decay}")
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        # Optional: Add LR scheduler
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10)
        # return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val/total_loss"}}
        return optimizer

    # --- Generation (Sampling from Distributions) ---
    @torch.no_grad()
    def generate(
        self,
        composition: torch.Tensor,  # (1, V_elem)
        max_atoms: int = 50,
        # --- Sampling Strategy Params ---
        sg_sampling_mode: str = "sample",  # 'sample' or 'argmax'
        lattice_sampling_mode: str = "sample",  # 'sample' or 'mean'
        atom_discrete_sampling_mode: str = "sample",  # 'sample' or 'argmax'
        coord_sampling_mode: str = "sample",  # 'sample' or 'mode' ('loc')
        temperature: float = 1.0,  # For categorical sampling
        # top_k / top_p could be added here
    ) -> Dict[str, Any]:
        """Autoregressive generation (sampling)."""
        log.info(f"Generating crystal structure with temperature={temperature}, max_atoms={max_atoms}")
        log.info(f"Sampling modes: SG={sg_sampling_mode}, lattice={lattice_sampling_mode}, atoms={atom_discrete_sampling_mode}, coords={coord_sampling_mode}")
        return self.model.generate(
            composition=composition,
            max_atoms=max_atoms,
            sg_sampling_mode=sg_sampling_mode,
            lattice_sampling_mode=lattice_sampling_mode,
            atom_discrete_sampling_mode=atom_discrete_sampling_mode,
            coord_sampling_mode=coord_sampling_mode,
            temperature=temperature,
        )