import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from torch.distributions import Normal
from typing import Dict, Any, Optional, List

from mattermake.models.modular_crystal_transformer_base import (
    ModularCrystalTransformerBase,
)
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ModularCrystalTransformer(LightningModule):
    """
    PyTorch Lightning module for the Fully Modular Crystal Transformer.
    Handles training, validation, testing, loss calculation, and generation.
    """

    def __init__(
        self,
        # --- Vocabularies & Indices ---
        element_vocab_size: int = 100,
        sg_vocab_size: int = 231,
        pad_idx: int = 0,
        start_idx: int = -1,
        end_idx: int = -2,
        # --- Model Dimensions ---
        d_model: int = 256,
        type_embed_dim: int = 64,
        # --- Transformer Params ---
        nhead: int = 8,
        num_atom_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # --- Encoder/Decoder Configuration ---
        encoder_configs: Dict[str, Dict[str, Any]] = None,
        decoder_configs: Dict[str, Dict[str, Any]] = None,
        # --- Encoding/Decoding Order ---
        encoding_order: List[str] = None,
        decoding_order: List[str] = None,
        # --- Optimizer & Loss Params ---
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        sg_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        eps: float = 1e-6,
        **kwargs,  # Catch any extra hparams
    ):
        super().__init__()
        self.save_hyperparameters()
        log.info("Initializing Fully Modular Crystal Transformer")

        # Initialize the base transformer model
        log.info(f"Creating modular base model with d_model={d_model}, nhead={nhead}")
        self.model = ModularCrystalTransformerBase(
            element_vocab_size=element_vocab_size,
            sg_vocab_size=sg_vocab_size,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            d_model=d_model,
            type_embed_dim=type_embed_dim,
            nhead=nhead,
            num_atom_decoder_layers=num_atom_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            encoder_configs=encoder_configs,
            decoder_configs=decoder_configs,
            encoding_order=encoding_order,
            decoding_order=decoding_order,
            eps=eps,
            **kwargs,
        )

        # --- Loss Functions ---
        log.info("Setting up loss functions")
        self.sg_loss = nn.CrossEntropyLoss()
        self.type_loss = nn.CrossEntropyLoss(ignore_index=self.hparams.pad_idx)
        # Lattice/Coord NLL loss calculated inline
        log.info("Fully Modular Crystal Transformer initialization complete")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass using the base model"""
        return self.model(batch)

    # Add back a simplified mapping function specifically for loss calculation
    def _map_indices_for_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps special token indices to embedding indices for loss calculation."""
        # This is needed because the loss expects mapped indices
        element_vocab_size = self.hparams.element_vocab_size
        start_embed_idx = element_vocab_size + 1  # START token
        end_embed_idx = element_vocab_size + 2  # END token
        pad_idx = self.hparams.pad_idx
        start_idx = self.hparams.start_idx
        end_idx = self.hparams.end_idx

        indices = indices.long()
        mapped_indices = indices.clone()
        mapped_indices[indices == start_idx] = start_embed_idx
        mapped_indices[indices == end_idx] = end_embed_idx
        mapped_indices[indices == pad_idx] = pad_idx  # Keep pad_idx the same
        return mapped_indices

    def calculate_loss(
        self, predictions: Dict[str, Any], batch: Dict[str, Any]
    ) -> Dict[str, torch.Tensor]:
        """Calculates the combined loss using NLL for continuous variables."""
        losses = {}
        total_loss = torch.tensor(0.0, device=self.device)

        # --- Targets ---
        sg_target = batch["spacegroup"].squeeze(1) - 1
        lattice_target = batch["lattice"]
        atom_types_target_raw = batch["atom_types"][:, 1:]
        # atom_wyckoffs_target_raw REMOVED
        atom_coords_target = batch["atom_coords"][:, 1:, :]
        atom_mask_target = batch["atom_mask"][:, 1:]

        # Map target indices for discrete losses
        type_target_mapped = self._map_indices_for_embedding(
            atom_types_target_raw  # is_type=True is default
        )
        # wyckoff_target_mapped REMOVED

        # --- SG Loss ---
        if "sg_logits" in predictions:
            loss_sg = self.sg_loss(predictions["sg_logits"], sg_target.long())
            losses["sg_loss"] = loss_sg * self.hparams.sg_loss_weight
            total_loss += losses["sg_loss"]
        else:
            log.warning("sg_logits not found in predictions, skipping SG loss.")

        # --- Lattice Loss ---
        if "lattice_mean" in predictions and "lattice_log_var" in predictions:
            try:
                lattice_mean = predictions["lattice_mean"]
                lattice_log_var = predictions["lattice_log_var"]

                # Add extra safeguards against NaN values
                if (
                    torch.isnan(lattice_mean).any()
                    or torch.isnan(lattice_log_var).any()
                ):
                    log.warning(
                        "Found NaN values in lattice parameters, applying additional stabilization"
                    )
                    lattice_mean = torch.nan_to_num(
                        lattice_mean, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    lattice_log_var = torch.nan_to_num(
                        lattice_log_var, nan=-1.0, posinf=2.0, neginf=-20.0
                    )

                # Stricter constraints for numerical stability
                lattice_mean = torch.clamp(lattice_mean, min=-10.0, max=10.0)
                lattice_log_var = torch.clamp(lattice_log_var, min=-10.0, max=2.0)

                # Ensure numerical stability
                lattice_var = torch.exp(lattice_log_var) + self.hparams.eps
                lattice_std = torch.sqrt(lattice_var).clamp(
                    min=self.hparams.eps, max=5.0
                )

                lattice_dist = Normal(lattice_mean, lattice_std)
                log_probs = lattice_dist.log_prob(lattice_target)

                # Clamp extreme values in log probabilities
                log_probs = torch.clamp(log_probs, min=-20.0, max=20.0)

                nll_lattice = -log_probs.mean()
                # Final safety clamp
                nll_lattice = torch.clamp(nll_lattice, min=0.0, max=10.0)

                losses["lattice_nll"] = nll_lattice * self.hparams.lattice_loss_weight
                total_loss += losses["lattice_nll"]
            except Exception as e:
                log.warning(f"Error in lattice loss calculation: {e}")
                # Provide a default loss that won't break training
                losses["lattice_nll"] = torch.tensor(
                    1.0, device=self.device, requires_grad=True
                )
                total_loss += losses["lattice_nll"]
        else:
            log.warning("Lattice predictions not found, skipping Lattice NLL loss.")

        # --- Atom Type Loss ---
        if "type_logits" in predictions:
            type_logits_flat = predictions["type_logits"].reshape(
                -1, self.model.effective_type_vocab_size
            )
            type_target_flat = type_target_mapped.reshape(-1)
            loss_type = self.type_loss(type_logits_flat, type_target_flat)
            losses["type_loss"] = loss_type * self.hparams.type_loss_weight
            total_loss += losses["type_loss"]
        else:
            log.warning("type_logits not found in predictions, skipping Type loss.")

        # --- Atom Wyckoff Loss REMOVED ---

        # --- Coordinate Loss (using new parameters) ---
        if "coord_mean" in predictions and "coord_log_var" in predictions:
            # We now directly receive mean in [0,1] range and log_variance
            coord_mean = predictions["coord_mean"]
            coord_log_var = predictions["coord_log_var"]

            # Convert log_var to variance for calculations
            coord_variance = torch.exp(coord_log_var) + self.hparams.eps

            # Ensure values are within reasonable ranges
            coord_mean = torch.clamp(coord_mean, min=0.0, max=1.0)
            coord_variance = torch.clamp(coord_variance, min=self.hparams.eps, max=0.1)

            try:
                # Use simple MSE loss weighted by precision - more stable than full NLL
                # This treats coordinates as predictions with uncertainty
                coord_squared_error = (coord_mean - atom_coords_target).pow(2)

                # Account for the periodic nature of fractional coords
                # If error > 0.5, it's shorter to go the other way around
                periodic_mask = coord_squared_error > 0.25  # (0.5)²
                coord_squared_error[periodic_mask] = 1.0 - torch.sqrt(
                    coord_squared_error[periodic_mask]
                )
                coord_squared_error[periodic_mask] = coord_squared_error[
                    periodic_mask
                ].pow(2)

                # Weight errors by precision
                weighted_error = coord_squared_error / coord_variance

                # Add log variance term (from Gaussian NLL) to prevent variance collapse
                loss_terms = weighted_error + torch.log(coord_variance)

                # Apply mask and compute mean
                mask_expanded = atom_mask_target.unsqueeze(-1).to(loss_terms.dtype)
                masked_loss = loss_terms * mask_expanded
                num_valid = mask_expanded.sum()

                if num_valid > 0:
                    coord_loss = masked_loss.sum() / (num_valid * 3)
                    # Safety clamp
                    coord_loss = torch.clamp(coord_loss, min=0.0, max=20.0)
                else:
                    coord_loss = torch.tensor(0.0, device=self.device)

                losses["coord_loss"] = coord_loss * self.hparams.coord_loss_weight
                total_loss += losses["coord_loss"]

            except Exception as e:
                log.warning(f"Error in coordinate loss calculation: {e}")
                # Provide a default loss that won't break training
                losses["coord_loss"] = torch.tensor(
                    1.0, device=self.device, requires_grad=True
                )
                total_loss += losses["coord_loss"]
        else:
            log.warning(
                "coord_mean or coord_log_var not found in predictions, skipping coordinate loss calculation."
            )

        losses["total_loss"] = total_loss
        return losses

    # --- Lightning Steps (training_step, validation_step, test_step) ---
    def step(self, batch: Dict[str, Any], batch_idx: int, stage: str):
        log.debug(f"{stage} batch {batch_idx}")
        try:
            # Forward pass - check for NaNs in inputs first
            for key, value in batch.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    log.warning(f"Found NaN values in input batch['{key}']")

            predictions = self(batch)

            # Check for NaNs in predictions before loss calculation
            for key, value in predictions.items():
                if isinstance(value, torch.Tensor) and torch.isnan(value).any():
                    log.warning(
                        f"Found NaN values in predictions['{key}']. Applying nan_to_num"
                    )
                    predictions[key] = torch.nan_to_num(value, nan=0.0)

            losses = self.calculate_loss(predictions, batch)
            batch_size = batch["composition"].size(0)
            log_opts = {
                "on_step": stage == "train",
                "on_epoch": True,
                "prog_bar": stage == "train",
                "batch_size": batch_size,
                "sync_dist": True,
            }
            self.log(
                f"{stage}/total_loss",
                losses["total_loss"],
                prog_bar=log_opts["prog_bar"],
                batch_size=batch_size,
                sync_dist=True,
            )
            self.log_dict(
                {f"{stage}/{k}": v for k, v in losses.items() if k != "total_loss"},
                on_step=log_opts["on_step"],
                on_epoch=log_opts["on_epoch"],
                batch_size=batch_size,
                sync_dist=True,
            )
            if batch_idx % 100 == 0:
                log.info(
                    f"{stage.capitalize()} batch {batch_idx}, loss: {losses['total_loss']:.4f}"
                )
            return losses["total_loss"]
        except Exception as e:
            log.error(
                f"Error during {stage}_step at batch {batch_idx}: {e}", exc_info=True
            )
            # Return a small but non-zero tensor to avoid breaking the training loop
            return torch.tensor(1e-5, device=self.device, requires_grad=True)

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        return self.step(batch, batch_idx, "train")

    def validation_step(self, batch: Dict[str, Any], batch_idx: int):
        self.step(batch, batch_idx, "val")

    def test_step(self, batch: Dict[str, Any], batch_idx: int):
        self.step(batch, batch_idx, "test")

    # --- Optimizer ---
    def configure_optimizers(self):
        log.info(
            f"Configuring optimizer with lr={self.hparams.learning_rate}, weight_decay={self.hparams.weight_decay}"
        )
        optimizer = optim.AdamW(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        # Add learning rate scheduler for better stability
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # Reduce LR by half when plateauing
            patience=2,  # Wait 2 epochs before reducing
            verbose=True,
            min_lr=1e-6,  # Don't go below this learning rate
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/total_loss",  # Monitor validation loss
                "interval": "epoch",  # Apply after each epoch
                "frequency": 1,  # Apply every epoch
            },
        }

    # --- Generation (Sampling from Distributions) ---
    @torch.no_grad()
    def generate(
        self,
        # --- Required Inputs for Encoders ---
        composition: torch.Tensor,
        # --- Optional Inputs for other potential encoders ---
        spacegroup: Optional[torch.Tensor] = None,
        # ... add other conditioning inputs as needed
        # --- Generation Parameters ---
        max_atoms: int = 50,
        sg_sampling_mode: str = "sample",
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",
        coord_sampling_mode: str = "sample",
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Autoregressive generation (sampling) using the modular base model."""
        log.info(f"Generating crystal structure with max_atoms={max_atoms}")
        log.info(
            f"Sampling modes: SG={sg_sampling_mode}, lattice={lattice_sampling_mode}, atoms={atom_discrete_sampling_mode}, coords={coord_sampling_mode}"
        )

        if hasattr(self.model, "generate") and callable(self.model.generate):
            # Pass only non-Wyckoff relevant args to base model generate
            return self.model.generate(
                composition=composition,
                spacegroup=spacegroup,
                max_atoms=max_atoms,
                sg_sampling_mode=sg_sampling_mode,
                lattice_sampling_mode=lattice_sampling_mode,
                atom_discrete_sampling_mode=atom_discrete_sampling_mode,
                coord_sampling_mode=coord_sampling_mode,
                temperature=temperature,
            )
        else:
            raise NotImplementedError(
                "Base model does not have a generate method implemented."
            )
