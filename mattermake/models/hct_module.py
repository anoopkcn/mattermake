import torch
from typing import Any, Dict, List, Optional
from lightning.pytorch import LightningModule

from mattermake.models.components.hct import (
    HierarchicalCrystalTransformer,
)
from mattermake.models.components.hct_utils.config import (
    HierarchicalCrystalTransformerConfig,
)

from mattermake.utils.pylogger import get_pylogger


class HierarchicalCrystalTransformerModule(LightningModule):
    """LightningModule for training and inference with HierarchicalCrystalTransformer"""

    def __init__(
        self,
        vocab_size: int = 2000,
        hidden_size: int = 768,
        num_hidden_layers: int = 12,
        num_attention_heads: int = 12,
        intermediate_size: int = 3072,
        hidden_dropout_prob: float = 0.1,
        attention_probs_dropout_prob: float = 0.1,
        max_position_embeddings: int = 512,
        type_vocab_size: int = 7,
        learning_rate: float = 1e-4,
        weight_decay: float = 0.01,
        warmup_steps: int = 1000,
        adam_epsilon: float = 1e-8,
        use_curriculum: bool = False,
        composition_curriculum_epochs: int = 5,
        space_group_curriculum_epochs: int = 5,
        lattice_curriculum_epochs: int = 5,
        composition_layers: int = 3,
        space_group_layers: int = 2,
        lattice_layers: int = 3,
        atom_layers: int = 6,
        integration_layers: int = 2,
        coordinate_embedding_dim: int = 32,  # Parameter for coordinate embedding dimension
        mixture_density_loss_weight: float = 0.5,  # Weight for mixture density network losses
        lattice_mixture_components: int = 5,  # Number of components for lattice parameter MoG
        coord_mixture_components: int = 5,  # Number of components for coordinate MoVM
        apply_wyckoff_constraints: bool = False,  # Whether to apply Wyckoff position constraints
        use_combined_wyckoff_tokens: bool = True,  # Whether to use combined Wyckoff tokens
        tokenizer_config: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.custom_logger = get_pylogger(__name__)

        # Initialize model
        config = HierarchicalCrystalTransformerConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=intermediate_size,
            hidden_dropout_prob=hidden_dropout_prob,
            attention_probs_dropout_prob=attention_probs_dropout_prob,
            max_position_embeddings=max_position_embeddings,
            type_vocab_size=type_vocab_size,
            composition_layers=composition_layers,
            space_group_layers=space_group_layers,
            lattice_layers=lattice_layers,
            atom_layers=atom_layers,
            integration_layers=integration_layers,
            use_curriculum=use_curriculum,
            composition_curriculum_epochs=composition_curriculum_epochs,
            space_group_curriculum_epochs=space_group_curriculum_epochs,
            lattice_curriculum_epochs=lattice_curriculum_epochs,
            coordinate_embedding_dim=coordinate_embedding_dim,
            lattice_mixture_components=lattice_mixture_components,
            coord_mixture_components=coord_mixture_components,
            apply_wyckoff_constraints=apply_wyckoff_constraints,
            use_combined_wyckoff_tokens=use_combined_wyckoff_tokens,
        )

        self.model = HierarchicalCrystalTransformer(config)

        # Store token mapping for generation
        self.tokenizer_config = tokenizer_config

        # Track current epoch for curriculum learning
        self.current_curriculum_epoch = 0

    def on_save_checkpoint(self, checkpoint):
        """Save tokenizer config with checkpoint"""
        if self.tokenizer_config:
            checkpoint["tokenizer_config"] = self.tokenizer_config

    def on_load_checkpoint(self, checkpoint):
        """Load tokenizer config from checkpoint"""
        if "tokenizer_config" in checkpoint:
            self.tokenizer_config = checkpoint["tokenizer_config"]

    def forward(self, batch):
        """Forward pass"""
        return self.model(
            input_ids=batch["input_ids"],
            segment_ids=batch["segment_ids"],
            attention_mask=batch.get("attention_mask"),
            labels=batch.get("target_ids"),
            use_causal_mask=True,
        )

    def training_step(self, batch, batch_idx):
        """Training step"""
        outputs = self(batch)
        loss = outputs["loss"]

        # Get batch size for logging
        batch_size = len(batch["input_ids"])

        self.log(
            "train_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log component-specific losses if available
        if "space_group_loss" in outputs:
            self.log(
                "train_sg_loss",
                outputs["space_group_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        if "lattice_loss" in outputs:
            self.log(
                "train_lattice_loss",
                outputs["lattice_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        if "element_loss" in outputs:
            self.log(
                "train_element_loss",
                outputs["element_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        if "wyckoff_loss" in outputs:
            self.log(
                "train_wyckoff_loss",
                outputs["wyckoff_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
        if "coordinate_loss" in outputs:
            self.log(
                "train_coordinate_loss",
                outputs["coordinate_loss"],
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        outputs = self(batch)

        # Check for NaNs/Infs in raw outputs first (optional but helpful)
        for key, value in outputs.items():
            if isinstance(value, torch.Tensor) and (
                torch.isnan(value).any() or torch.isinf(value).any()
            ):
                self.custom_logger.warning(
                    f"NaN/Inf detected in validation output key: {key}"
                )

        # Check main loss
        if (
            "loss" not in outputs
            or not isinstance(outputs["loss"], torch.Tensor)
            or torch.isnan(outputs["loss"]).any()
            or torch.isinf(outputs["loss"]).any()
        ):
            loss = torch.tensor(
                0.0, device=self.device
            )  # Dummy loss if main loss is bad
            self.custom_logger.warning(
                "Validation loss is NaN/Inf or missing, using 0.0"
            )
        else:
            loss = outputs["loss"]

        # Get batch size for logging
        batch_size = len(batch["input_ids"])

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
            batch_size=batch_size,
        )

        # Log component-specific validation losses if available, with NaN/Inf checks
        if (
            "space_group_loss" in outputs
            and isinstance(outputs["space_group_loss"], torch.Tensor)
            and not (
                torch.isnan(outputs["space_group_loss"]).any()
                or torch.isinf(outputs["space_group_loss"]).any()
            )
        ):
            self.log(
                "val_sg_loss",
                outputs["space_group_loss"],
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,  # Keep prog_bar for consistency with original code
                batch_size=batch_size,
            )
        elif (
            "space_group_loss" in outputs
        ):  # Only log warning if key exists but value is bad
            self.custom_logger.warning("val_sg_loss is NaN/Inf or not a Tensor")

        if (
            "lattice_loss" in outputs
            and isinstance(outputs["lattice_loss"], torch.Tensor)
            and not (
                torch.isnan(outputs["lattice_loss"]).any()
                or torch.isinf(outputs["lattice_loss"]).any()
            )
        ):
            self.log(
                "val_lattice_loss",
                outputs["lattice_loss"],
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,  # Keep prog_bar
                batch_size=batch_size,
            )
        elif "lattice_loss" in outputs:
            self.custom_logger.warning("val_lattice_loss is NaN/Inf or not a Tensor")

        if (
            "element_loss" in outputs
            and isinstance(outputs["element_loss"], torch.Tensor)
            and not (
                torch.isnan(outputs["element_loss"]).any()
                or torch.isinf(outputs["element_loss"]).any()
            )
        ):
            self.log(
                "val_element_loss",
                outputs["element_loss"],
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,  # Keep prog_bar
                batch_size=batch_size,
            )
        elif "element_loss" in outputs:
            self.custom_logger.warning("val_element_loss is NaN/Inf or not a Tensor")

        if (
            "wyckoff_loss" in outputs
            and isinstance(outputs["wyckoff_loss"], torch.Tensor)
            and not (
                torch.isnan(outputs["wyckoff_loss"]).any()
                or torch.isinf(outputs["wyckoff_loss"]).any()
            )
        ):
            self.log(
                "val_wyckoff_loss",
                outputs["wyckoff_loss"],
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,  # Keep prog_bar
                batch_size=batch_size,
            )
        elif "wyckoff_loss" in outputs:
            self.custom_logger.warning("val_wyckoff_loss is NaN/Inf or not a Tensor")

        if (
            "coordinate_loss" in outputs
            and isinstance(outputs["coordinate_loss"], torch.Tensor)
            and not (
                torch.isnan(outputs["coordinate_loss"]).any()
                or torch.isinf(outputs["coordinate_loss"]).any()
            )
        ):
            self.log(
                "val_coordinate_loss",
                outputs["coordinate_loss"],
                on_epoch=True,
                sync_dist=True,
                prog_bar=True,  # Keep prog_bar
                batch_size=batch_size,
            )
        elif "coordinate_loss" in outputs:
            self.custom_logger.warning("val_coordinate_loss is NaN/Inf or not a Tensor")

        # Calculate token-level accuracy (should be robust to NaN losses)
        # Check if logits are valid before proceeding
        if (
            "logits" in outputs
            and isinstance(outputs["logits"], torch.Tensor)
            and not (
                torch.isnan(outputs["logits"]).any()
                or torch.isinf(outputs["logits"]).any()
            )
        ):
            logits = outputs["logits"]
            labels = batch["target_ids"]

            # Shift for teacher forcing
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            # Get predictions
            preds = torch.argmax(shift_logits, dim=-1)

            # Mask out padding tokens
            valid_mask = (shift_labels != -100) & (
                shift_labels != 2
            )  # Ignore -100 and pad token
            correct = (preds == shift_labels) & valid_mask

            # Calculate accuracy
            valid_sum = valid_mask.sum().float()
            if valid_sum > 0:
                accuracy = correct.sum().float() / valid_sum
                self.log(
                    "val_accuracy",
                    accuracy,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.custom_logger.warning(
                    "No valid tokens found for accuracy calculation in this batch."
                )
                self.log(
                    "val_accuracy",
                    0.0,
                    on_epoch=True,
                    prog_bar=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )  # Log 0 if no valid tokens

            # Calculate accuracy by segment type
            segment_ids = batch["segment_ids"]
            shift_segments = segment_ids[:, 1:].contiguous()

            # Composition accuracy
            comp_mask = (
                shift_segments == self.model.config.SEGMENT_COMPOSITION
            ) & valid_mask
            comp_mask_sum = comp_mask.sum().float()
            if comp_mask_sum > 0:
                comp_correct = ((preds == shift_labels) & comp_mask).sum().float()
                comp_accuracy = comp_correct / comp_mask_sum
                self.log(
                    "val_comp_accuracy",
                    comp_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_comp_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # Space group accuracy
            sg_mask = (
                shift_segments == self.model.config.SEGMENT_SPACE_GROUP
            ) & valid_mask
            sg_mask_sum = sg_mask.sum().float()
            if sg_mask_sum > 0:
                sg_correct = ((preds == shift_labels) & sg_mask).sum().float()
                sg_accuracy = sg_correct / sg_mask_sum
                self.log(
                    "val_sg_accuracy",
                    sg_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_sg_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # Lattice accuracy
            lattice_mask = (
                shift_segments == self.model.config.SEGMENT_LATTICE
            ) & valid_mask
            lattice_mask_sum = lattice_mask.sum().float()
            if lattice_mask_sum > 0:
                lattice_correct = ((preds == shift_labels) & lattice_mask).sum().float()
                lattice_accuracy = lattice_correct / lattice_mask_sum
                self.log(
                    "val_lattice_accuracy",
                    lattice_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_lattice_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # Atom accuracy (element, wyckoff, coordinate)
            atom_mask = (
                (shift_segments == self.model.config.SEGMENT_ELEMENT)
                | (shift_segments == self.model.config.SEGMENT_WYCKOFF)
                | (shift_segments == self.model.config.SEGMENT_COORDINATE)
            ) & valid_mask
            atom_mask_sum = atom_mask.sum().float()
            if atom_mask_sum > 0:
                atom_correct = ((preds == shift_labels) & atom_mask).sum().float()
                atom_accuracy = atom_correct / atom_mask_sum
                self.log(
                    "val_atom_accuracy",
                    atom_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_atom_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            # Break down atom accuracy into element, Wyckoff, and coordinate accuracies
            element_mask = (
                shift_segments == self.model.config.SEGMENT_ELEMENT
            ) & valid_mask
            element_mask_sum = element_mask.sum().float()
            if element_mask_sum > 0:
                element_correct = ((preds == shift_labels) & element_mask).sum().float()
                element_accuracy = element_correct / element_mask_sum
                self.log(
                    "val_element_accuracy",
                    element_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_element_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            wyckoff_mask = (
                shift_segments == self.model.config.SEGMENT_WYCKOFF
            ) & valid_mask
            wyckoff_mask_sum = wyckoff_mask.sum().float()
            if wyckoff_mask_sum > 0:
                wyckoff_correct = ((preds == shift_labels) & wyckoff_mask).sum().float()
                wyckoff_accuracy = wyckoff_correct / wyckoff_mask_sum
                self.log(
                    "val_wyckoff_accuracy",
                    wyckoff_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_wyckoff_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

            coordinate_mask = (
                shift_segments == self.model.config.SEGMENT_COORDINATE
            ) & valid_mask
            coordinate_mask_sum = coordinate_mask.sum().float()
            if coordinate_mask_sum > 0:
                coordinate_correct = (
                    ((preds == shift_labels) & coordinate_mask).sum().float()
                )
                coordinate_accuracy = coordinate_correct / coordinate_mask_sum
                self.log(
                    "val_coordinate_accuracy",
                    coordinate_accuracy,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )
            else:
                self.log(
                    "val_coordinate_accuracy",
                    0.0,
                    on_epoch=True,
                    sync_dist=True,
                    batch_size=batch_size,
                )

        else:
            # Log warnings and default values if logits are invalid
            self.custom_logger.warning(
                "Logits are NaN/Inf or missing. Skipping accuracy calculation."
            )
            # Log default 0 for all accuracy metrics
            self.log(
                "val_accuracy",
                0.0,
                on_epoch=True,
                prog_bar=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_comp_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_sg_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_lattice_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_atom_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_element_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_wyckoff_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )
            self.log(
                "val_coordinate_accuracy",
                0.0,
                on_epoch=True,
                sync_dist=True,
                batch_size=batch_size,
            )

        return loss  # Return the potentially corrected loss

    def test_step(self, batch, batch_idx):
        """Test step"""
        outputs = self(batch)
        loss = outputs["loss"]

        # Get batch size for logging
        batch_size = len(batch["input_ids"])

        self.log(
            "test_loss", loss, on_epoch=True, sync_dist=True, batch_size=batch_size
        )
        return loss

    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler with curriculum learning"""
        # Filter parameters that require gradients
        parameters = [p for p in self.parameters() if p.requires_grad]

        # Create optimizer with weight decay
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
            eps=self.hparams.adam_epsilon,
        )

        # Create scheduler with curriculum learning if enabled
        if self.hparams.use_curriculum:

            def curriculum_lambda(epoch):
                # Basic warmup
                base_lr = min(1.0, epoch / self.hparams.warmup_steps)

                # Track current epoch for curriculum
                self.current_curriculum_epoch = epoch

                # Apply curriculum scheduling
                if epoch < self.hparams.composition_curriculum_epochs:
                    # Focus on learning composition tokens
                    self.model.set_active_modules(["composition"])
                elif (
                    epoch
                    < self.hparams.composition_curriculum_epochs
                    + self.hparams.space_group_curriculum_epochs
                ):
                    # Add space group learning
                    self.model.set_active_modules(["composition", "space_group"])
                elif (
                    epoch
                    < self.hparams.composition_curriculum_epochs
                    + self.hparams.space_group_curriculum_epochs
                    + self.hparams.lattice_curriculum_epochs
                ):
                    # Add lattice parameter learning
                    self.model.set_active_modules(
                        ["composition", "space_group", "lattice"]
                    )
                else:
                    # Full model training
                    self.model.set_active_modules(
                        ["composition", "space_group", "lattice", "atoms"]
                    )

                return base_lr

            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, curriculum_lambda)
        else:
            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer, lambda epoch: min(1.0, epoch / self.hparams.warmup_steps)
            )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def on_train_epoch_start(self):
        """Update model's active modules based on curriculum at the start of each epoch"""
        if self.hparams.use_curriculum:
            epoch = self.current_epoch

            if epoch < self.hparams.composition_curriculum_epochs:
                self.model.set_active_modules(["composition"])
                self.custom_logger.info(
                    f"Epoch {epoch}: Training composition modules only"
                )
            elif (
                epoch
                < self.hparams.composition_curriculum_epochs
                + self.hparams.space_group_curriculum_epochs
            ):
                self.model.set_active_modules(["composition", "space_group"])
                self.custom_logger.info(
                    f"Epoch {epoch}: Training composition and space group modules"
                )
            elif (
                epoch
                < self.hparams.composition_curriculum_epochs
                + self.hparams.space_group_curriculum_epochs
                + self.hparams.lattice_curriculum_epochs
            ):
                self.model.set_active_modules(["composition", "space_group", "lattice"])
                self.custom_logger.info(
                    f"Epoch {epoch}: Training composition, space group, and lattice modules"
                )
            else:
                self.model.set_active_modules(
                    ["composition", "space_group", "lattice", "atoms"]
                )
                self.custom_logger.info(f"Epoch {epoch}: Training all modules")

    def generate_structure(
        self,
        start_tokens: Optional[torch.Tensor] = None,
        constraints: Optional[Dict[str, Any]] = None,
        max_length: int = 512,
        temperature: float = 0.8,
        top_k: Optional[int] = 40,
        top_p: Optional[float] = 0.9,
        repetition_penalty: float = 1.2,
        num_return_sequences: int = 1,
        verbose: bool = False,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Generate crystal structures

        Args:
            start_tokens: Optional starting tokens (defaults to BOS token)
            constraints: Dictionary of constraints for generation
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: If set, sample from top k most likely tokens
            top_p: If set, sample from tokens with cumulative probability >= top_p
            num_return_sequences: Number of sequences to generate
            verbose: Whether to print verbose output during generation

        Returns:
            List of generated structures with their token sequences
        """
        self.model.eval()
        device = next(self.model.parameters()).device

        # Prepare BOS token if start_tokens not provided
        if start_tokens is None:
            # Assuming BOS token is 0
            start_tokens = torch.tensor([[0]], dtype=torch.long, device=device)
            start_segments = torch.tensor([[0]], dtype=torch.long, device=device)
        else:
            # If start_tokens provided, ensure they're on the right device
            start_tokens = start_tokens.to(device)
            if "start_segments" in kwargs:
                start_segments = kwargs["start_segments"].to(device)
            else:
                # Default to special token segment
                start_segments = torch.zeros_like(start_tokens)

        # Expand for multiple sequences
        if num_return_sequences > 1:
            start_tokens = start_tokens.expand(num_return_sequences, -1)
            start_segments = start_segments.expand(num_return_sequences, -1)

        # If we have token mappings from the tokenizer, add them to constraints
        if self.tokenizer_config and constraints is None:
            constraints = {}

        if self.tokenizer_config and "token_id_maps" not in constraints:
            constraints["token_id_maps"] = {
                "idx_to_token": self.tokenizer_config.get("idx_to_token", {}),
                "token_to_idx": self.tokenizer_config.get("token_to_idx", {}),
            }

        # Generate sequences
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=start_tokens,
                segment_ids=start_segments,
                constraints=constraints,
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                eos_token_id=1,  # Assuming EOS token is 1
                pad_token_id=2,  # Assuming PAD token is 2
                verbose=verbose,
            )

        # Parse generated sequences
        generated_structures = []
        for i in range(outputs["sequences"].size(0)):
            seq = outputs["sequences"][i].cpu().tolist()
            segments = outputs["segment_ids"][i].cpu().tolist()

            # Get continuous predictions
            continuous_data = {}
            # Always use continuous predictions
            # Extract continuous lattice parameters if available
            if (
                "continuous_lattice_lengths" in outputs
                and "continuous_lattice_angles" in outputs
            ):
                # Get the values for this sequence
                lengths = outputs["continuous_lattice_lengths"][i].cpu()
                angles = outputs["continuous_lattice_angles"][i].cpu()
                continuous_data["lattice_params"] = {
                    "lengths": lengths,
                    "angles": angles,
                }

            # Extract continuous fractional coordinates if available
            if "continuous_fractional_coords" in outputs:
                coords = outputs["continuous_fractional_coords"][i].cpu()
                continuous_data["fractional_coords"] = coords

            # Decode to structure (using a helper function, implementation depends on tokenizer)
            if hasattr(self, "decode_to_structure"):
                structure = self.decode_to_structure(
                    seq, segments, continuous_data=continuous_data
                )
            else:
                # Basic decoding
                structure = self._basic_structure_decoding(
                    seq, segments, continuous_data=continuous_data
                )

            generated_structures.append(structure)

        return generated_structures

    def _basic_structure_decoding(self, sequence, segments, continuous_data=None):
        """Basic decoding of token sequence to structure representation

        Args:
            sequence: List of token IDs
            segments: List of segment IDs
            continuous_data: Optional dictionary containing continuous predictions
                             for lattice parameters and fractional coordinates
        """
        # Initialize components
        composition = {}
        space_group = None
        lattice_params = []
        atoms = []

        # Use continuous lattice parameters if available
        has_continuous_lattice = False
        has_continuous_coords = False

        # Check explicit continuous prediction flags if available
        if continuous_data:
            has_continuous_lattice = continuous_data.get(
                "has_continuous_lattice", False
            )
            has_continuous_coords = continuous_data.get("has_continuous_coords", False)

        # Process continuous lattice parameters if available
        if continuous_data and (
            "lattice_params" in continuous_data
            or (
                "continuous_lattice_lengths" in continuous_data
                and "continuous_lattice_angles" in continuous_data
            )
        ):
            # Handle both formats for backward compatibility
            if "lattice_params" in continuous_data:
                lattice_data = continuous_data["lattice_params"]
                if "lengths" in lattice_data and "angles" in lattice_data:
                    # Use the predicted continuous values
                    if (
                        isinstance(lattice_data["lengths"], torch.Tensor)
                        and lattice_data["lengths"].numel() >= 3
                    ):
                        # Extract a, b, c from lengths tensor
                        a, b, c = lattice_data["lengths"][:3].tolist()
                        lattice_params.extend([a, b, c])
                        has_continuous_lattice = True

                    if (
                        isinstance(lattice_data["angles"], torch.Tensor)
                        and lattice_data["angles"].numel() >= 3
                    ):
                        # Extract alpha, beta, gamma from angles tensor
                        alpha, beta, gamma = lattice_data["angles"][:3].tolist()
                        lattice_params.extend([alpha, beta, gamma])
                        has_continuous_lattice = True
            else:
                # Direct continuous prediction tensors
                if isinstance(
                    continuous_data.get("continuous_lattice_lengths"), torch.Tensor
                ):
                    lengths = continuous_data["continuous_lattice_lengths"]
                    if lengths.numel() >= 3:
                        # Get the last prediction for this sequence
                        a, b, c = (
                            lengths[-3:].tolist()
                            if lengths.dim() == 1
                            else lengths[0, :3].tolist()
                        )
                        lattice_params.extend([a, b, c])
                        has_continuous_lattice = True

                if isinstance(
                    continuous_data.get("continuous_lattice_angles"), torch.Tensor
                ):
                    angles = continuous_data["continuous_lattice_angles"]
                    if angles.numel() >= 3:
                        # Get the last prediction for this sequence
                        alpha, beta, gamma = (
                            angles[-3:].tolist()
                            if angles.dim() == 1
                            else angles[0, :3].tolist()
                        )
                        lattice_params.extend([alpha, beta, gamma])
                        has_continuous_lattice = True

        # Track current atom being processed
        current_element = None
        current_wyckoff = None
        current_coords = []

        # Create segment mappings if we have tokenizer config
        idx_to_token = {}
        if self.tokenizer_config and "idx_to_token" in self.tokenizer_config:
            idx_to_token = self.tokenizer_config["idx_to_token"]

        # Process tokens and segment IDs
        for i, (token_id, segment_id) in enumerate(zip(sequence, segments)):
            # Skip special tokens
            if token_id <= 2:  # BOS, EOS, PAD
                continue

            # Get token name
            token_name = idx_to_token.get(token_id, f"UNK-{token_id}")

            # Process based on segment type
            if segment_id == self.model.config.SEGMENT_COMPOSITION:
                # Parse composition token (format: COMP_El_count)
                if token_name.startswith("COMP_"):
                    parts = token_name.split("_")
                    if len(parts) >= 3:
                        element = parts[1]
                        try:
                            count = int(parts[2])
                            composition[element] = composition.get(element, 0) + count
                        except ValueError:
                            pass

            elif segment_id == self.model.config.SEGMENT_SPACE_GROUP:
                # Parse space group token (format: SG_num)
                if token_name.startswith("SG_"):
                    try:
                        space_group = int(token_name[3:])
                    except ValueError:
                        pass

            elif segment_id == self.model.config.SEGMENT_LATTICE:
                # Parse lattice parameter token (format: LAT_bin)
                if token_name.startswith("LAT_"):
                    try:
                        bin_idx = int(token_name[4:])
                        # Convert bin index to actual value
                        if len(lattice_params) < 3:
                            # Length parameter (a, b, c) in Angstroms
                            value = bin_idx * (
                                100.0 / 100
                            )  # Assuming lattice_bins = 100
                        else:
                            # Angle parameter (alpha, beta, gamma) in degrees
                            value = bin_idx * (
                                180.0 / 100
                            )  # Assuming lattice_bins = 100
                        lattice_params.append(value)
                    except ValueError:
                        pass

            elif segment_id == self.model.config.SEGMENT_ELEMENT:
                # If we were processing an atom, save it before starting a new one
                if current_element is not None and len(current_coords) > 0:
                    atoms.append(
                        {
                            "element": current_element,
                            "wyckoff": current_wyckoff,
                            "coords": current_coords,
                        }
                    )
                    current_coords = []

                # Parse element token (format: ELEM_symbol)
                if token_name.startswith("ELEM_"):
                    current_element = token_name[5:]
                    current_wyckoff = None

            elif segment_id == self.model.config.SEGMENT_WYCKOFF:
                # Parse Wyckoff token (format: WYCK_letter)
                if token_name.startswith("WYCK_"):
                    current_wyckoff = token_name[5:]

            elif segment_id == self.model.config.SEGMENT_COORDINATE:
                # Parse coordinate token (format: COORD_value)
                if token_name.startswith("COORD_"):
                    try:
                        value = float(token_name[6:])
                        current_coords.append(value)
                    except ValueError:
                        pass

        # Use continuous fractional coordinates if available
        # This implementation simplifies the coordinate handling and would need to be
        # expanded in a real application to properly align atoms and coordinates
        if continuous_data and (
            "fractional_coords" in continuous_data
            or "continuous_fractional_coords" in continuous_data
        ):
            # Handle both formats for backward compatibility
            if "fractional_coords" in continuous_data:
                coords_tensor = continuous_data["fractional_coords"]
            else:
                coords_tensor = continuous_data["continuous_fractional_coords"]

            if isinstance(coords_tensor, torch.Tensor) and coords_tensor.size(-1) == 3:
                # We have proper coordinate predictions (num_atoms, 3)
                has_continuous_coords = True

                # Ensure we have some atoms defined even if none found from discrete tokens
                if (
                    len(atoms) == 0
                    and has_continuous_lattice
                    and "elements" in continuous_data
                ):
                    # Try to create atoms from predicted elements if available
                    elements = continuous_data["elements"]
                    if isinstance(elements, list) and len(elements) > 0:
                        # Create basic atoms with predicted elements
                        for i, elem in enumerate(elements):
                            atoms.append(
                                {
                                    "element": elem,
                                    "wyckoff": "a",  # Default Wyckoff position
                                    "coords": [],  # Will be filled below
                                }
                            )
                    else:
                        # Create generic atoms if no elements predicted
                        num_coords = (
                            coords_tensor.size(0)
                            if coords_tensor.dim() == 2
                            else coords_tensor.size(0) // 3
                        )
                        for i in range(num_coords):
                            atoms.append(
                                {
                                    "element": "X",  # Generic element
                                    "wyckoff": "a",  # Default Wyckoff position
                                    "coords": [],  # Will be filled below
                                }
                            )

                # Update atoms with continuous coordinates
                if len(atoms) > 0:
                    # Prepare coordinates with robust shape handling
                    if coords_tensor.dim() == 1:
                        # Handle flat tensor (num_atoms * 3)
                        if coords_tensor.size(0) % 3 == 0:
                            coords_tensor = coords_tensor.view(
                                -1, 3
                            )  # Reshape to (num_atoms, 3)
                        else:
                            print(
                                f"Warning: Coordinates tensor size {coords_tensor.size(0)} is not divisible by 3"
                            )
                            # Create a default set of coordinates
                            coords_tensor = (
                                torch.zeros(
                                    (len(atoms), 3), device=coords_tensor.device
                                )
                                + 0.5
                            )
                    elif coords_tensor.dim() > 2:
                        # Handle unexpected higher-dimensional tensor
                        print(
                            f"Warning: Unexpected coordinate tensor shape: {coords_tensor.shape}"
                        )
                        # Flatten to 2D by averaging across extra dimensions if possible
                        if coords_tensor.size(-1) == 3:
                            # Average across all dims except the last
                            orig_shape = coords_tensor.shape
                            new_shape = (-1, 3)
                            try:
                                coords_tensor = coords_tensor.reshape(new_shape)
                            except RuntimeError as e:
                                print(f"Error reshaping coordinates: {e}")
                                print(
                                    f"Original shape: {orig_shape}, Attempted: {new_shape}, Total elements: {coords_tensor.numel()}"
                                )
                                # Fall back to zeros
                                coords_tensor = (
                                    torch.zeros(
                                        (len(atoms), 3), device=coords_tensor.device
                                    )
                                    + 0.5
                                )
                        else:
                            # Can't handle this shape
                            print(
                                f"Can't process coordinate tensor with shape {coords_tensor.shape}"
                            )
                            coords_tensor = (
                                torch.zeros(
                                    (len(atoms), 3), device=coords_tensor.device
                                )
                                + 0.5
                            )

                    # Limit to the number of atoms we have
                    num_atoms = min(len(atoms), coords_tensor.size(0))
                    for i in range(num_atoms):
                        # Convert tensor coordinates to list
                        atoms[i]["coords"] = coords_tensor[i].tolist()

                    # If we have more predicted coordinates than atoms, create generic atoms
                    if coords_tensor.size(0) > len(atoms) and has_continuous_lattice:
                        for i in range(len(atoms), coords_tensor.size(0)):
                            atoms.append(
                                {
                                    "element": "X",  # Generic element
                                    "wyckoff": "a",  # Default Wyckoff position
                                    "coords": coords_tensor[i].tolist(),
                                }
                            )

        # Add the last atom if it exists and we're not using continuous coordinates
        if (
            not has_continuous_coords
            and current_element is not None
            and len(current_coords) > 0
        ):
            atoms.append(
                {
                    "element": current_element,
                    "wyckoff": current_wyckoff,
                    "coords": current_coords,
                }
            )

        # Build structure dictionary
        structure = {
            "tokens": sequence,
            "segments": segments,
            "composition": composition,
            "space_group": space_group,
            "lattice_params": lattice_params,
            "atoms": atoms,
            "used_continuous_lattice": has_continuous_lattice,
            "used_continuous_coords": has_continuous_coords,
        }

        # Try to build pymatgen structure if possible
        try:
            from pymatgen.core import Structure, Lattice

            if len(lattice_params) >= 6:
                a, b, c, alpha, beta, gamma = lattice_params[:6]
                lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

                species = []
                coords = []

                for atom in atoms:
                    element = atom["element"]
                    atom_coords = atom["coords"]

                    # Ensure we have 3 coordinates
                    if len(atom_coords) < 3:
                        atom_coords = atom_coords + [0.0] * (3 - len(atom_coords))
                    elif len(atom_coords) > 3:
                        atom_coords = atom_coords[:3]

                    species.append(element)
                    coords.append(atom_coords)

                if species and coords:
                    pmg_structure = Structure(lattice, species, coords)
                    structure["pmg_structure"] = pmg_structure
        except (ImportError, Exception) as e:
            structure["structure_error"] = str(e)

        return structure

    def on_validation_epoch_end(self):
        """Generate example structures at the end of validation epoch"""
        if self.current_epoch % 5 == 0:  # Generate every 5 epochs
            try:
                # Generate a few examples
                structures = self.generate_structure(
                    num_return_sequences=2, temperature=0.7
                )

                # Log generation samples
                self.custom_logger.info(
                    f"\n=== Generated samples (epoch {self.current_epoch}) ==="
                )
                for i, structure in enumerate(structures):
                    self.custom_logger.info(f"Sample {i + 1}:")
                    # Log structure details
                    self.custom_logger.info(f"Composition: {structure['composition']}")
                    self.custom_logger.info(f"Space group: {structure['space_group']}")
                    self.custom_logger.info(
                        f"Lattice parameters: {structure['lattice_params']}"
                    )
                    self.custom_logger.info(
                        f"Number of atoms: {len(structure['atoms'])}"
                    )

                    # Log a sample of atoms
                    if structure["atoms"]:
                        self.custom_logger.info("First few atoms:")
                        for j, atom in enumerate(structure["atoms"][:3]):
                            self.custom_logger.info(
                                f"  {atom['element']} at position {atom['coords']} (Wyckoff: {atom['wyckoff']})"
                            )

                        if len(structure["atoms"]) > 3:
                            self.custom_logger.info(
                                f"  ... and {len(structure['atoms']) - 3} more atoms"
                            )

            except Exception as e:
                self.custom_logger.error(f"Error generating samples: {str(e)}")
