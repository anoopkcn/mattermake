import torch
import torch.nn as nn
import torch.optim as optim
from lightning.pytorch import LightningModule
from torch.distributions import Normal
from typing import Dict, Any, Optional, List

from mattermake.models.modular_hierarchical_crystal_transformer_base import (
    ModularHierarchicalCrystalTransformerBase,
)
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class ModularHierarchicalCrystalTransformer(LightningModule):
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
        encoder_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        decoder_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        # --- Encoding/Decoding Order ---
        encoding_order: Optional[List[str]] = None,
        decoding_order: Optional[List[str]] = None,
        # --- Optimizer & Loss Params ---
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        sg_loss_weight: float = 1.0,
        lattice_loss_weight: float = 1.0,
        wyckoff_loss_weight: float = 1.0,
        type_loss_weight: float = 1.0,
        coord_loss_weight: float = 1.0,
        wyckoff_vocab_size: Optional[int] = None,
        eps: float = 1e-6,
        **kwargs,  # Catch any extra hparams
    ):
        super().__init__()

        # Auto-calculate Wyckoff vocab size if not provided
        if wyckoff_vocab_size is None:
            from mattermake.data.components.wyckoff_interface import (
                get_effective_wyckoff_vocab_size,
            )

            wyckoff_vocab_size = get_effective_wyckoff_vocab_size()

        self.save_hyperparameters()
        log.info("Initializing Fully Modular Crystal Transformer")

        # Initialize the base transformer model
        log.info(f"Creating modular base model with d_model={d_model}, nhead={nhead}")
        self.model = ModularHierarchicalCrystalTransformerBase(
            element_vocab_size=element_vocab_size,
            sg_vocab_size=sg_vocab_size,
            wyckoff_vocab_size=wyckoff_vocab_size,
            pad_idx=pad_idx,
            start_idx=start_idx,
            end_idx=end_idx,
            d_model=d_model,
            type_embed_dim=type_embed_dim,
            nhead=nhead,
            num_atom_decoder_layers=num_atom_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            encoder_configs={} if encoder_configs is None else encoder_configs,
            decoder_configs={} if decoder_configs is None else decoder_configs,
            encoding_order=[] if encoding_order is None else encoding_order,
            decoding_order=[] if decoding_order is None else decoding_order,
            eps=eps,
            **kwargs,
        )

        # --- Loss Functions ---
        log.info("Setting up loss functions")
        self.sg_loss = nn.CrossEntropyLoss()
        self.type_loss = nn.CrossEntropyLoss(
            ignore_index=getattr(self.hparams, "pad_idx", 0)
        )
        self.wyckoff_loss = nn.CrossEntropyLoss(ignore_index=0)  # 0 = padding
        # Lattice/Coord NLL loss calculated inline
        log.info("Fully Modular Crystal Transformer initialization complete")

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Forward pass using the base model"""
        return self.model(batch)

    # Add back a simplified mapping function specifically for loss calculation
    def _map_indices_for_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps special token indices to embedding indices for loss calculation."""
        # This is needed because the loss expects mapped indices
        element_vocab_size = getattr(self.hparams, "element_vocab_size", 100)
        start_embed_idx = element_vocab_size + 1  # START token
        end_embed_idx = element_vocab_size + 2  # END token
        pad_idx = getattr(self.hparams, "pad_idx", 0)
        start_idx = getattr(self.hparams, "start_idx", -1)
        end_idx = getattr(self.hparams, "end_idx", -2)

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
            # Add bounds checking for space group target indices to prevent CUDA assertion errors
            sg_vocab_size = predictions["sg_logits"].size(-1)
            if sg_target.numel() > 0:
                max_sg_target = sg_target.max().item()
                min_sg_target = sg_target.min().item()
                if max_sg_target >= sg_vocab_size or min_sg_target < 0:
                    log.warning(
                        f"Space group target indices out of bounds. Min: {min_sg_target}, Max: {max_sg_target}, Vocab size: {sg_vocab_size}"
                    )
                    sg_target = torch.clamp(sg_target, 0, sg_vocab_size - 1)

            loss_sg = self.sg_loss(predictions["sg_logits"], sg_target.long())
            losses["sg_loss"] = loss_sg * getattr(self.hparams, "sg_loss_weight", 1.0)
            total_loss += losses["sg_loss"]
        else:
            log.warning("sg_logits not found in predictions, skipping SG loss.")

        # --- Lattice Matrix Loss ---
        if (
            "lattice_matrix_mean" in predictions
            and "lattice_matrix_log_var" in predictions
        ):
            try:
                lattice_matrix_mean = predictions["lattice_matrix_mean"]
                lattice_matrix_log_var = predictions["lattice_matrix_log_var"]
                lattice_target_matrix = batch["lattice"]

                # If the target is still in the old format (6 parameters), convert it to matrix format
                if lattice_target_matrix.shape[-1] == 6:
                    # Borrow the conversion function from the encoder
                    # Note: in a production setting, this should be moved to a shared utility
                    lattice_encoder = self.model.encoders["lattice"]
                    # Handle conversion through direct attribute access
                    if hasattr(lattice_encoder, "_convert_params_to_matrix"):
                        convert_fn = getattr(
                            lattice_encoder, "_convert_params_to_matrix"
                        )
                        lattice_target_matrix = convert_fn(lattice_target_matrix)
                    else:
                        log.warning(
                            "Lattice encoder missing conversion function, using raw tensor"
                        )
                elif (
                    lattice_target_matrix.shape[-1] != 9
                    and lattice_target_matrix.dim() == 3
                    and lattice_target_matrix.shape[1:] == (3, 3)
                ):
                    # It's a 3x3 matrix, flatten it
                    lattice_target_matrix = lattice_target_matrix.reshape(
                        lattice_target_matrix.size(0), -1
                    )
                elif lattice_target_matrix.shape[-1] != 9:
                    raise ValueError(
                        f"Unexpected lattice target shape: {lattice_target_matrix.shape}. Expected (B, 9) or (B, 6) or (B, 3, 3)."
                    )

                # Add extra safeguards against NaN values
                if (
                    torch.isnan(lattice_matrix_mean).any()
                    or torch.isnan(lattice_matrix_log_var).any()
                ):
                    log.warning(
                        "Found NaN values in lattice matrix, applying additional stabilization"
                    )
                    lattice_matrix_mean = torch.nan_to_num(
                        lattice_matrix_mean, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                    lattice_matrix_log_var = torch.nan_to_num(
                        lattice_matrix_log_var, nan=-1.0, posinf=2.0, neginf=-20.0
                    )

                # Stricter constraints for numerical stability
                lattice_matrix_mean = torch.clamp(
                    lattice_matrix_mean, min=-10.0, max=10.0
                )
                lattice_matrix_log_var = torch.clamp(
                    lattice_matrix_log_var, min=-10.0, max=2.0
                )

                # Ensure numerical stability
                eps_value = getattr(self.hparams, "eps", 1e-6)
                lattice_matrix_var = torch.exp(lattice_matrix_log_var) + eps_value
                lattice_matrix_std = torch.sqrt(lattice_matrix_var).clamp(
                    min=eps_value, max=5.0
                )

                # Calculate the NLL loss for the lattice matrix
                lattice_matrix_dist = Normal(lattice_matrix_mean, lattice_matrix_std)
                log_probs = lattice_matrix_dist.log_prob(lattice_target_matrix)

                # Clamp extreme values in log probabilities
                log_probs = torch.clamp(log_probs, min=-20.0, max=20.0)

                # Mean across batch and all 9 matrix elements
                nll_lattice = -torch.mean(log_probs)
                # Final safety clamp
                nll_lattice = torch.clamp(nll_lattice, min=0.0, max=10.0)

                losses["lattice_matrix_nll"] = nll_lattice * getattr(
                    self.hparams, "lattice_loss_weight", 1.0
                )
                total_loss += losses["lattice_matrix_nll"]
            except Exception as e:
                log.warning(f"Error in lattice matrix loss calculation: {e}")
                # Provide a default loss that won't break training
                losses["lattice_matrix_nll"] = torch.tensor(
                    1.0, device=self.device, requires_grad=True
                )
                total_loss += losses["lattice_matrix_nll"]
        elif "lattice_mean" in predictions and "lattice_log_var" in predictions:
            # Handle legacy format for backward compatibility
            log.warning(
                "Found legacy lattice parameter format. Consider updating the model."
            )
            try:
                # Similar calculation, but using the old keys
                lattice_mean = predictions["lattice_mean"]
                lattice_log_var = predictions["lattice_log_var"]
                lattice_target = batch["lattice"]

                # Safety processing (NaN checks, clamping, etc.)
                # [Omitted - same as above but for lattice_mean/lattice_log_var]

                # Calculate NLL loss
                eps_value = getattr(self.hparams, "eps", 1e-6)
                lattice_var = torch.exp(lattice_log_var) + eps_value
                lattice_std = torch.sqrt(lattice_var).clamp(min=eps_value, max=5.0)
                lattice_dist = Normal(lattice_mean, lattice_std)
                log_probs = lattice_dist.log_prob(lattice_target)
                log_probs = torch.clamp(log_probs, min=-20.0, max=20.0)
                nll_lattice = -log_probs.mean()
                nll_lattice = torch.clamp(nll_lattice, min=0.0, max=10.0)

                losses["lattice_nll"] = nll_lattice * getattr(
                    self.hparams, "lattice_loss_weight", 1.0
                )
                total_loss += losses["lattice_nll"]
            except Exception as e:
                log.warning(f"Error in legacy lattice loss calculation: {e}")
                losses["lattice_nll"] = torch.tensor(
                    1.0, device=self.device, requires_grad=True
                )
                total_loss += losses["lattice_nll"]
        else:
            log.warning(
                "Neither lattice matrix nor legacy lattice predictions found, skipping Lattice loss."
            )

        # --- Atom Type Loss ---
        if "type_logits" in predictions:
            # Get the effective vocabulary size from model attributes
            if hasattr(self.model, "effective_type_vocab_size"):
                effective_vocab_size = getattr(self.model, "effective_type_vocab_size")
            else:
                # Fallback to default (element_vocab_size + 3 for PAD, START, END)
                effective_vocab_size = (
                    getattr(self.hparams, "element_vocab_size", 100) + 3
                )

            type_logits_flat = predictions["type_logits"].reshape(
                -1, effective_vocab_size
            )
            type_target_flat = type_target_mapped.reshape(-1)

            # Add bounds checking for target indices to prevent CUDA assertion errors
            if type_target_flat.numel() > 0:
                max_target = type_target_flat.max().item()
                min_target = type_target_flat.min().item()
                if max_target >= effective_vocab_size or min_target < 0:
                    log.warning(
                        f"Atom type target indices out of bounds. Min: {min_target}, Max: {max_target}, Vocab size: {effective_vocab_size}"
                    )
                    type_target_flat = torch.clamp(
                        type_target_flat, 0, effective_vocab_size - 1
                    )

            loss_type = self.type_loss(type_logits_flat, type_target_flat)
            losses["type_loss"] = loss_type * getattr(
                self.hparams, "type_loss_weight", 1.0
            )
            total_loss += losses["type_loss"]
        else:
            log.warning("type_logits not found in predictions, skipping Type loss.")

        # --- Wyckoff Loss ---
        if "wyckoff_logits" in predictions and "wyckoff" in batch:
            wyckoff_target = batch["wyckoff"][:, 1:]  # Remove start token if present
            wyckoff_logits = predictions["wyckoff_logits"]

            # Flatten for loss calculation
            from mattermake.data.components.wyckoff_interface import (
                get_effective_wyckoff_vocab_size,
            )

            wyckoff_vocab_size = get_effective_wyckoff_vocab_size()
            wyckoff_logits_flat = wyckoff_logits.reshape(-1, wyckoff_vocab_size)
            wyckoff_target_flat = wyckoff_target.reshape(-1)

            # Map special tokens to valid vocabulary indices before bounds checking
            if wyckoff_target_flat.numel() > 0:
                # Map special tokens: START (-1) -> vocab_size - 2, END (-2) -> vocab_size - 1
                wyckoff_target_mapped = wyckoff_target_flat.clone()
                wyckoff_target_mapped = torch.where(
                    wyckoff_target_flat == -1,
                    wyckoff_vocab_size - 2,
                    wyckoff_target_mapped,
                )
                wyckoff_target_mapped = torch.where(
                    wyckoff_target_flat == -2,
                    wyckoff_vocab_size - 1,
                    wyckoff_target_mapped,
                )

                # Add bounds checking for wyckoff target indices to prevent CUDA assertion errors
                max_wyckoff_target = wyckoff_target_mapped.max().item()
                min_wyckoff_target = wyckoff_target_mapped.min().item()
                if max_wyckoff_target >= wyckoff_vocab_size or min_wyckoff_target < 0:
                    log.warning(
                        f"Wyckoff target indices out of bounds after mapping. Min: {min_wyckoff_target}, Max: {max_wyckoff_target}, Vocab size: {wyckoff_vocab_size}"
                    )
                    wyckoff_target_flat = torch.clamp(
                        wyckoff_target_mapped, 0, wyckoff_vocab_size - 1
                    )
                else:
                    wyckoff_target_flat = wyckoff_target_mapped

            # Create mask for valid Wyckoff positions based on space group
            if "spacegroup" in batch:
                from mattermake.data.components.wyckoff_interface import (
                    create_wyckoff_mask,
                )

                sg_numbers = batch["spacegroup"].squeeze()

                # Create mask and apply to logits
                mask = create_wyckoff_mask(sg_numbers, self.device)  # (B, vocab_size)

                # Expand mask to match logits shape
                seq_len = wyckoff_logits.size(1)
                expanded_mask = mask.unsqueeze(1).expand(
                    -1, seq_len, -1
                )  # (B, L, vocab_size)
                expanded_mask_flat = expanded_mask.reshape(-1, wyckoff_vocab_size)

                # Apply mask (set invalid positions to very negative values)
                wyckoff_logits_flat = wyckoff_logits_flat.masked_fill(
                    ~expanded_mask_flat, -1e9
                )

            # Calculate loss
            loss_wyckoff = self.wyckoff_loss(wyckoff_logits_flat, wyckoff_target_flat)
            losses["wyckoff_loss"] = loss_wyckoff * getattr(
                self.hparams, "wyckoff_loss_weight", 1.0
            )
            total_loss += losses["wyckoff_loss"]
        else:
            log.warning(
                "wyckoff_logits or wyckoff not found in predictions/batch, skipping Wyckoff loss."
            )

        # --- Coordinate Loss (using new parameters) ---
        if "coord_mean" in predictions and "coord_log_var" in predictions:
            # We now directly receive mean in [0,1] range and log_variance
            coord_mean = predictions["coord_mean"]
            coord_log_var = predictions["coord_log_var"]

            # Convert log_var to variance for calculations
            eps_value = getattr(self.hparams, "eps", 1e-6)
            coord_variance = torch.exp(coord_log_var) + eps_value

            # Ensure values are within reasonable ranges
            coord_mean = torch.clamp(coord_mean, min=0.0, max=1.0)
            coord_variance = torch.clamp(coord_variance, min=eps_value, max=0.1)

            try:
                # Initialize variables to prevent unbound errors
                mask_expanded = torch.ones(1, 1, 1, device=self.device)
                num_valid = torch.tensor(1.0, device=self.device)
                masked_mse = torch.tensor(0.01, device=self.device)

                # Move to CPU for validation if CUDA operations are failing
                try:
                    # Test if CUDA operations are working
                    _ = torch.tensor(1.0, device=self.device)
                    use_cuda = True
                except:
                    log.warning(
                        "CUDA operations failing, using CPU for coordinate loss"
                    )
                    use_cuda = False

                # Ensure coordinate targets are in valid range [0, 1]
                try:
                    atom_coords_target_clamped = torch.clamp(
                        atom_coords_target, 0.0, 1.0
                    )
                    coord_mean_clamped = torch.clamp(coord_mean, 0.0, 1.0)
                except Exception as clamp_error:
                    log.warning(f"Error in coordinate clamping: {clamp_error}")
                    # Use safer CPU operations
                    atom_coords_target_clamped = (
                        atom_coords_target.cpu()
                        .clamp(0.0, 1.0)
                        .to(atom_coords_target.device)
                    )
                    coord_mean_clamped = (
                        coord_mean.cpu().clamp(0.0, 1.0).to(coord_mean.device)
                    )

                # Validate coordinate shapes match
                if coord_mean_clamped.shape != atom_coords_target_clamped.shape:
                    log.warning(
                        f"Coordinate shape mismatch: pred {coord_mean_clamped.shape} vs target {atom_coords_target_clamped.shape}"
                    )
                    # Reshape to match the smaller dimension
                    min_seq_len = min(
                        coord_mean_clamped.shape[1], atom_coords_target_clamped.shape[1]
                    )
                    coord_mean_clamped = coord_mean_clamped[:, :min_seq_len, :]
                    atom_coords_target_clamped = atom_coords_target_clamped[
                        :, :min_seq_len, :
                    ]

                # Use simple MSE loss weighted by precision - more stable than full NLL
                # This treats coordinates as predictions with uncertainty
                try:
                    coord_squared_error = (
                        coord_mean_clamped - atom_coords_target_clamped
                    ).pow(2)
                except Exception as mse_error:
                    log.warning(f"Error in MSE calculation: {mse_error}")
                    # Fallback to CPU calculation
                    coord_mean_cpu = coord_mean_clamped.cpu()
                    coord_target_cpu = atom_coords_target_clamped.cpu()
                    coord_squared_error = (
                        (coord_mean_cpu - coord_target_cpu)
                        .pow(2)
                        .to(coord_mean_clamped.device)
                    )

                # Validate the error tensor using CPU operations to avoid CUDA issues
                try:
                    error_cpu = coord_squared_error.cpu()
                    has_nan = torch.isnan(error_cpu).any().item()
                    has_inf = torch.isinf(error_cpu).any().item()
                    if has_nan or has_inf:
                        log.warning(
                            "NaN or Inf detected in coordinate squared error, cleaning..."
                        )
                        coord_squared_error = torch.nan_to_num(
                            coord_squared_error, nan=0.0, posinf=1.0, neginf=0.0
                        )
                except Exception as validation_error:
                    log.warning(f"Error in tensor validation: {validation_error}")
                    # Force clean the tensor
                    coord_squared_error = torch.nan_to_num(
                        coord_squared_error, nan=0.0, posinf=1.0, neginf=0.0
                    )

                # Account for the periodic nature of fractional coords
                # If error > 0.5, it's shorter to go the other way around
                try:
                    periodic_mask = coord_squared_error > 0.25  # (0.5)²

                    # Safe periodic boundary handling
                    if periodic_mask.any():
                        try:
                            sqrt_values = torch.sqrt(
                                torch.clamp(
                                    coord_squared_error[periodic_mask], min=1e-8
                                )
                            )
                            periodic_corrections = (1.0 - sqrt_values).pow(2)
                            coord_squared_error = (
                                coord_squared_error.clone()
                            )  # Ensure we can modify in-place
                            coord_squared_error[periodic_mask] = periodic_corrections
                        except Exception as pe:
                            log.warning(f"Error in periodic boundary correction: {pe}")
                            # Skip periodic correction if it fails
                            pass
                except Exception as periodic_error:
                    log.warning(
                        f"Error in periodic boundary calculation: {periodic_error}"
                    )
                    # Skip periodic correction entirely

                # Validate mask dimensions and data
                try:
                    if atom_mask_target.shape != coord_squared_error.shape[:2]:
                        log.warning(
                            f"Mask shape mismatch: {atom_mask_target.shape} vs {coord_squared_error.shape[:2]}"
                        )
                        # Resize mask to match coordinate error shape
                        batch_size, seq_len = coord_squared_error.shape[:2]
                        if atom_mask_target.shape[1] >= seq_len:
                            atom_mask_target = atom_mask_target[:batch_size, :seq_len]
                        else:
                            padding = torch.zeros(
                                batch_size,
                                seq_len - atom_mask_target.shape[1],
                                device=atom_mask_target.device,
                                dtype=atom_mask_target.dtype,
                            )
                            atom_mask_target = torch.cat(
                                [atom_mask_target, padding], dim=1
                            )

                    # Ensure mask is valid (no NaN, proper dtype)
                    atom_mask_target = torch.nan_to_num(
                        atom_mask_target, nan=0.0
                    ).bool()
                except Exception as mask_error:
                    log.warning(f"Error in mask processing: {mask_error}")
                    # Create a simple valid mask
                    batch_size, seq_len = coord_squared_error.shape[:2]
                    atom_mask_target = torch.ones(
                        batch_size,
                        seq_len,
                        device=coord_squared_error.device,
                        dtype=torch.bool,
                    )

                # Just calculate plain MSE loss for coordinates
                try:
                    mask_expanded = atom_mask_target.unsqueeze(-1).to(
                        coord_squared_error.dtype
                    )

                    # Clamp coordinate error to prevent numerical issues
                    coord_squared_error = torch.clamp(
                        coord_squared_error, min=0.0, max=4.0
                    )  # Max possible error is 2.0^2

                    masked_mse = coord_squared_error * mask_expanded
                    num_valid = mask_expanded.sum()
                except Exception as loss_calc_error:
                    log.warning(f"Error in loss calculation setup: {loss_calc_error}")
                    # Use CPU for final calculation
                    try:
                        coord_error_cpu = coord_squared_error.cpu().clamp(0.0, 4.0)
                        mask_cpu = (
                            atom_mask_target.cpu()
                            .unsqueeze(-1)
                            .to(coord_error_cpu.dtype)
                        )
                        masked_mse = (coord_error_cpu * mask_cpu).to(
                            coord_squared_error.device
                        )
                        num_valid = mask_cpu.sum().to(coord_squared_error.device)
                    except Exception as final_error:
                        log.warning(f"Final fallback failed: {final_error}")
                        # Return a safe default loss
                        losses["coord_loss"] = torch.tensor(
                            0.01, device=self.device, requires_grad=True
                        )
                        total_loss += losses["coord_loss"]
                        # Continue to rest of function instead of returning early

                # Optionally add a variance regularization term, but don't let it make the total loss negative
                # log_var_reg = torch.log(coord_variance) * mask_expanded
                # This is causing the negative loss, so we'll use a different approach

                if num_valid > 0:
                    # Use simple MSE loss initially (guaranteed to be positive)
                    coord_loss = masked_mse.sum() / (num_valid * 3)

                    # Add a variance regularization term that won't make loss negative
                    # This encourages reasonably small but non-zero variances
                    var_too_small = (coord_variance < 0.01).float() * mask_expanded
                    var_too_large = (coord_variance > 0.05).float() * mask_expanded
                    var_penalty = var_too_small.sum() + var_too_large.sum()

                    if var_penalty.item() > 0:
                        # Small penalty for inappropriate variances
                        coord_loss = coord_loss + 0.01 * var_penalty / num_valid

                    # coord_loss_raw = coord_loss.clone()  # For logging
                    # Safety clamp for extreme values
                    coord_loss = torch.clamp(coord_loss, min=0.0001, max=20.0)

                    # # CRITICAL DIAGNOSTIC PRINTS
                    # print("\n============= COORDINATE LOSS DIAGNOSTICS ===============")
                    # print(f"Raw coord loss (before clamp): {coord_loss_raw.item():.10f}")
                    # print(f"Clamped coord loss: {coord_loss.item():.10f}")
                    # print(f"coord_loss_weight from config: {self.hparams.coord_loss_weight}")
                    # print(f"Final weighted coord loss: {(coord_loss * self.hparams.coord_loss_weight).item():.10f}")
                    # print(f"Valid atoms: {num_valid.item()} of {mask_expanded.numel()} positions")
                    # print(f"Coord variance stats: min={coord_variance.min().item():.6f}, max={coord_variance.max().item():.6f}")
                    # print(f"MSE error stats: min={coord_squared_error.min().item():.6f}, max={coord_squared_error.max().item():.6f}")

                    # # Check if variances are too large, causing weighted error to disappear
                    # if coord_variance.max() > 0.05:
                    #     print("WARNING: Coordinate variances are very large, causing weighted errors to be very small!")
                    # print("=======================================================\n")
                else:
                    coord_loss = torch.tensor(0.0, device=self.device)
                    print(
                        "WARNING: No valid atoms found for coordinate loss calculation (num_valid=0)!"
                    )

                # Force a minimum loss value for diagnostic purposes
                coord_loss_weighted = coord_loss * getattr(
                    self.hparams, "coord_loss_weight", 1.0
                )

                # # For debugging: make sure we get some visible loss contribution
                # if self.hparams.coord_loss_weight == 0.0:
                #     print("ERROR: coord_loss_weight is set to ZERO in config!")

                # Force coordinate loss to be at least 0.01 regardless of weight (diagnostic only)
                forced_min_loss = torch.tensor(
                    0.01, device=self.device, requires_grad=True
                )
                losses["coord_loss"] = torch.max(coord_loss_weighted, forced_min_loss)

                # Store the original total for comparison
                # old_total = total_loss.clone()
                total_loss += losses["coord_loss"]

                # Show the impact on total loss
                # print(
                #     f"Total loss before coord: {old_total.item():.4f}, after: {total_loss.item():.4f} (delta: {(total_loss - old_total).item():.4f})"
                # )

            except Exception as e:
                log.warning(f"Error in coordinate loss calculation: {e}")
                log.warning(
                    f"coord_mean shape: {coord_mean.shape if 'coord_mean' in locals() else 'undefined'}"
                )
                log.warning(
                    f"atom_coords_target shape: {atom_coords_target.shape if 'atom_coords_target' in locals() else 'undefined'}"
                )
                log.warning(
                    f"atom_mask_target shape: {atom_mask_target.shape if 'atom_mask_target' in locals() else 'undefined'}"
                )

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
    def configure_optimizers(self):  # type: ignore
        lr = getattr(self.hparams, "learning_rate", 1e-4)
        wd = getattr(self.hparams, "weight_decay", 0.01)

        log.info(f"Configuring optimizer with lr={lr}, weight_decay={wd}")
        optimizer = optim.AdamW(
            self.parameters(),
            lr=lr,
            weight_decay=wd,
        )

        # Add learning rate scheduler for better stability
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,  # Reduce LR by half when plateauing
            patience=2,  # Wait 2 epochs before reducing
            cooldown=0,
            min_lr=1e-6,  # Don't go below this learning rate
        )

        # Return in a dictionary format that Lightning supports
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
        # --- Primary Inputs for Encoders (at the same hierarchical level) ---
        composition: torch.Tensor,  # Required input
        spacegroup: Optional[
            torch.Tensor
        ] = None,  # Optional but treated as primary input when provided
        # --- Generation Parameters ---
        max_atoms: int = 50,
        sg_sampling_mode: str = "sample",  # Only used when spacegroup is None
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",
        coord_sampling_mode: str = "sample",
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """Autoregressive generation (sampling) using the modular base model.

        The model treats both composition and space group (when provided) as primary inputs
        at the same hierarchical level. If space group is not provided, it will be predicted
        based on the composition.
        """
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
