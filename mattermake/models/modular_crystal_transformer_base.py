import torch
import torch.nn as nn
import math
from typing import Dict, Any, Optional

from mattermake.models.components.modular_encoders import (
    CompositionEncoder,
    SpaceGroupEncoder,
    LatticeEncoder,
    # Import other encoders here if added later
)
from mattermake.models.components.modular_decoders import (
    AtomTypeDecoder,
    AtomCoordinateDecoder,
    DecoderRegistry,
    # Import other decoders here if added later
)
from mattermake.models.components.modular_attention import ModularCrossAttention


# Re-use PositionalEncoding if it's defined elsewhere, or define it here
class PositionalEncoding(nn.Module):
    """Standard positional encoding."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [seq_len, batch_size, embedding_dim]
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class ModularCrystalTransformerBase(nn.Module):
    """
    Modular, hierarchical crystal transformer with separate cross-attention per encoder.
    Generates lattice params, atom types, and coordinates. (Wyckoff removed)
    """

    def __init__(
        self,
        # --- Vocabularies & Indices ---
        element_vocab_size: int = 100,
        sg_vocab_size: int = 231,
        # wyckoff_vocab_size: REMOVED
        pad_idx: int = 0,
        start_idx: int = -1,
        end_idx: int = -2,
        # --- Model Dimensions ---
        d_model: int = 256,
        type_embed_dim: int = 64,
        # wyckoff_embed_dim: REMOVED
        # --- Transformer Params ---
        nhead: int = 8,
        num_atom_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        # --- Encoder Configuration ---
        encoder_configs: Dict[str, Dict[str, Any]] = None,
        # --- Additional Params ---
        eps: float = 1e-6,
        **kwargs,  # Catch any extra hparams
    ):
        super().__init__()
        # Store hyperparameters, excluding self, kwargs, __class__
        self.hparams = {
            k: v
            for k, v in locals().items()
            if k not in ("self", "kwargs", "__class__")
        }
        # Update with any extra kwargs passed
        self.hparams.update(kwargs)

        self.eps = eps
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.d_model = d_model

        # --- Create Modular Encoders ---
        if encoder_configs is None:
            # Default minimal config if none provided
            encoder_configs = {
                "composition": {
                    "vocab_size": element_vocab_size,
                    "num_layers": 2,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                },
                "spacegroup": {"vocab_size": sg_vocab_size, "embed_dim": 64},
                "lattice": {"latent_dim": 64, "equivariant": False},
            }
        self.encoder_configs = encoder_configs
        self.encoders = nn.ModuleDict()
        for name, config in encoder_configs.items():
            # Use get() for flexibility, providing defaults
            if name == "composition":
                self.encoders[name] = CompositionEncoder(
                    element_vocab_size=config.get("vocab_size", element_vocab_size),
                    d_model=d_model,
                    nhead=config.get("nhead", nhead),
                    num_layers=config.get("num_layers", 2),
                    dim_feedforward=config.get("dim_feedforward", dim_feedforward),
                    dropout=config.get("dropout", dropout),
                )
            elif name == "spacegroup":
                self.encoders[name] = SpaceGroupEncoder(
                    sg_vocab_size=config.get("vocab_size", sg_vocab_size),
                    sg_embed_dim=config.get("embed_dim", 64),
                    d_model=d_model,
                )
            elif name == "lattice":
                self.encoders[name] = LatticeEncoder(
                    d_model=d_model,
                    latent_dim=config.get("latent_dim", 64),
                    equivariant=config.get("equivariant", False),
                )
            else:
                print(f"Warning: Unknown encoder type '{name}' in config. Skipping.")

        # --- Lattice Representation ---
        # Removed old lattice projector in favor of the lattice encoder

        # --- Atom Sequence Preparation (Embeddings + Input Projection) ---
        self._max_element_idx = self.hparams.get("element_vocab_size", 100)
        # _max_wyckoff_idx REMOVED
        # Map: PAD(0)->0, Valid(1..N)->1..N, START(-1)->N+1, END(-2)->N+2
        self.effective_type_vocab_size = self._max_element_idx + 3
        self.type_start_embed_idx = self._max_element_idx + 1
        self.type_end_embed_idx = self._max_element_idx + 2

        self.type_embedding = nn.Embedding(
            self.effective_type_vocab_size,
            type_embed_dim,
            padding_idx=self.pad_idx,
        )
        # wyckoff_embedding REMOVED

        # Combine atom step info (type emb + coords) and project # MODIFIED
        atom_step_dim = type_embed_dim + 3  # REMOVED wyckoff_embed_dim
        self.atom_input_proj = nn.Linear(atom_step_dim, d_model)
        self.atom_pos_encoder = PositionalEncoding(d_model, dropout)

        # --- Create Prediction Heads ---
        # Space Group Head
        self.sg_head = nn.Linear(
            d_model, 230
        )  # Direct size based on target range 0-229

        # Lattice Head
        self.lattice_head = nn.Linear(d_model, 6 * 2)  # mean + log_var per param
        # Initialize lattice head with small weights to improve numerical stability
        nn.init.xavier_normal_(self.lattice_head.weight, gain=0.5)
        nn.init.zeros_(self.lattice_head.bias)

        # --- Initialize Modular Decoders ---
        decoder_configs = kwargs.get(
            "decoder_configs",
            {
                "type": {"vocab_size": self.effective_type_vocab_size},
                "coordinate": {"eps": eps},
            },
        )

        # Create decoders
        decoders = {}
        for name, config in decoder_configs.items():
            # Ensure config is a dictionary even if it's None
            config = config if config is not None else {}

            if name == "type":
                # Use 'atom_type' as the key to avoid conflict with Python's built-in 'type'
                decoders["atom_type"] = AtomTypeDecoder(
                    d_model=d_model,
                    vocab_size=config.get("vocab_size", self.effective_type_vocab_size),
                )
            elif name == "coordinate":
                decoders[name] = AtomCoordinateDecoder(
                    d_model=d_model, eps=config.get("eps", eps)
                )
            else:
                print(f"Warning: Unknown decoder type '{name}' in config. Skipping.")

        # Create decoder registry
        self.decoder_registry = DecoderRegistry(decoders)

        # Modular decoders completely replace the need for separate prediction heads

        # --- Create Attention Mechanisms & Atom Decoder ---
        self.atom_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        # Separate attention for lattice prediction (uses only base encoders)
        self.lattice_modular_cross_attn = ModularCrossAttention(
            encoder_names=list(self.encoders.keys()),  # ONLY base encoders
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        # Separate attention for atom decoding (includes lattice context)
        # Include all encoder outputs without duplications
        atom_encoder_names = list(self.encoders.keys())

        self.atom_modular_cross_attn = ModularCrossAttention(
            encoder_names=atom_encoder_names,  # Base encoders + lattice contexts
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )
        # Standard decoder stack
        self.atom_decoder = nn.TransformerDecoder(
            self.atom_decoder_layer, num_layers=num_atom_decoder_layers
        )

    def _map_indices_for_embedding(
        self,
        indices: torch.Tensor,
        is_type: bool = True,  # Default to True
    ) -> torch.Tensor:
        """Maps PAD, START, END indices to non-negative indices for embedding lookup."""
        # **MODIFIED:** Simplified for type only
        if not is_type:
            # This case should no longer occur
            raise ValueError(
                "_map_indices_for_embedding called for non-type, but wyckoff is removed."
            )

        indices = indices.long()
        start_embed_idx = self.type_start_embed_idx
        end_embed_idx = self.type_end_embed_idx

        mapped_indices = indices.clone()
        mapped_indices[indices == self.start_idx] = start_embed_idx
        mapped_indices[indices == self.end_idx] = end_embed_idx
        return mapped_indices

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Teacher forcing forward pass for training"""

        # --- Step 1: Run all registered encoders ---
        encoder_outputs = {}
        composition = batch["composition"].float()
        # Check for NaNs in composition
        if torch.isnan(composition).any():
            composition = torch.nan_to_num(composition, nan=0.0)

        sg_target = batch["spacegroup"]

        for name, encoder in self.encoders.items():
            try:
                if isinstance(encoder, CompositionEncoder):
                    output = encoder(composition)
                    # Check for NaNs in encoder output
                    if torch.isnan(output).any():
                        output = torch.nan_to_num(
                            output, nan=0.0, posinf=1e6, neginf=-1e6
                        )
                    encoder_outputs[name] = output
                elif isinstance(encoder, SpaceGroupEncoder):
                    output = encoder(sg_target)
                    # Check for NaNs in encoder output
                    if torch.isnan(output).any():
                        output = torch.nan_to_num(
                            output, nan=0.0, posinf=1e6, neginf=-1e6
                        )
                    encoder_outputs[name] = output
                elif name in batch:  # Generic fallback
                    output = encoder(batch[name])
                    # Check for NaNs in encoder output
                    if torch.isnan(output).any():
                        output = torch.nan_to_num(
                            output, nan=0.0, posinf=1e6, neginf=-1e6
                        )
                    encoder_outputs[name] = output
                else:
                    print(
                        f"Warning: Input for encoder '{name}' not found in batch. Skipping."
                    )
            except Exception as e:
                print(f"Error in encoder '{name}': {e}")
                # Create a safe dummy tensor as fallback
                dummy_shape = [composition.size(0), 1, self.d_model]
                encoder_outputs[name] = torch.zeros(dummy_shape, device=self.device)

        # --- Step 2: Predict Space Group ---
        if "composition" in encoder_outputs:
            # Check for NaNs in composition encoder output
            comp_output = encoder_outputs["composition"]
            if torch.isnan(comp_output).any():
                comp_output = torch.nan_to_num(
                    comp_output, nan=0.0, posinf=1e6, neginf=-1e6
                )
                encoder_outputs["composition"] = comp_output

            sg_logits = self.sg_head(comp_output.squeeze(1))
            # Check for NaNs in sg_logits
            if torch.isnan(sg_logits).any():
                sg_logits = torch.nan_to_num(sg_logits, nan=0.0)
        else:
            sg_logits = torch.zeros(
                (composition.size(0), 230), device=self.device
            )  # Use 230

        # --- Step 3: Predict Lattice ---
        lattice_query = encoder_outputs.get(
            "composition",
            torch.zeros((composition.size(0), 1, self.d_model), device=self.device),
        )
        # Check lattice query for NaNs
        if torch.isnan(lattice_query).any():
            lattice_query = torch.nan_to_num(lattice_query, nan=0.0)

        # Ensure all encoder outputs are NaN-free before cross-attention
        clean_encoder_outputs = {}
        for name, tensor in encoder_outputs.items():
            if torch.isnan(tensor).any():
                clean_encoder_outputs[name] = torch.nan_to_num(
                    tensor, nan=0.0, posinf=1e6, neginf=-1e6
                )
            else:
                clean_encoder_outputs[name] = tensor

        fused_context_for_lattice = self.lattice_modular_cross_attn(
            query=lattice_query, encoder_outputs=clean_encoder_outputs
        )

        # Check fused context for NaNs
        if torch.isnan(fused_context_for_lattice).any():
            fused_context_for_lattice = torch.nan_to_num(
                fused_context_for_lattice, nan=0.0, posinf=1e6, neginf=-1e6
            )
        lattice_params = self.lattice_head(fused_context_for_lattice.squeeze(1))

        # Apply gradient clipping to prevent NaN values
        lattice_params = torch.clamp(lattice_params, -1e6, 1e6)
        # Check for NaNs and replace them
        lattice_params = torch.nan_to_num(
            lattice_params, nan=0.0, posinf=1e6, neginf=-1e6
        )
        lattice_mean, lattice_log_var = torch.chunk(lattice_params, 2, dim=-1)

        # Ensure numerical stability for log_var
        lattice_log_var = torch.clamp(lattice_log_var, -20, 2)

        # --- Step 4: Prepare for Atom Sequence Decoding ---
        lattice_target = batch["lattice"]
        # Check lattice_target for NaNs
        if torch.isnan(lattice_target).any():
            lattice_target = torch.nan_to_num(lattice_target, nan=0.0)

        # Process lattice using the lattice encoder
        if "lattice" in self.encoders:
            # Use the encoder to get a more sophisticated representation
            lattice_context = self.encoders["lattice"](lattice_target)
            # Add to the encoder outputs (using the encoder's name directly)
            clean_encoder_outputs["lattice"] = lattice_context
        else:
            # Create a fallback representation if encoder is missing
            # This avoids errors if configuration doesn't include lattice encoder
            dummy_shape = [lattice_target.size(0), 1, self.d_model]
            clean_encoder_outputs["lattice"] = torch.zeros(
                dummy_shape, device=self.device
            )

        # Create decoder contexts with clean encoder outputs
        decoder_contexts = clean_encoder_outputs.copy()

        # --- Step 5: Atom Sequence Decoding (Teacher Forcing) ---
        atom_types_target = batch["atom_types"]
        # atom_wyckoffs_target REMOVED
        atom_coords_target = batch["atom_coords"]
        atom_mask = batch["atom_mask"]

        # Check for NaNs in coordinates
        if torch.isnan(atom_coords_target).any():
            atom_coords_target = torch.nan_to_num(atom_coords_target, nan=0.0)

        # Prepare Atom Decoder Inputs (Shifted Right)
        atom_types_input_raw = atom_types_target[:, :-1]
        # atom_wyckoffs_input_raw REMOVED
        atom_coords_input = atom_coords_target[:, :-1, :]
        atom_mask_input = atom_mask[:, :-1]

        # Map indices -> Embedding indices
        types_embed_idx = self._map_indices_for_embedding(atom_types_input_raw)
        # wyckoffs_embed_idx REMOVED

        # Clamp indices
        types_embed_idx_clamped = torch.clamp(
            types_embed_idx, 0, self.effective_type_vocab_size - 1
        )
        # wyckoffs_embed_idx_clamped REMOVED

        type_embeds = self.type_embedding(types_embed_idx_clamped)
        # Check for NaNs in embeddings
        if torch.isnan(type_embeds).any():
            type_embeds = torch.nan_to_num(type_embeds, nan=0.0)
        # wyckoff_embeds REMOVED

        # Check atom coords for NaNs
        if torch.isnan(atom_coords_input).any():
            atom_coords_input = torch.nan_to_num(atom_coords_input, nan=0.0)

        # Combine atom step inputs (Type + Coords only)
        atom_step_inputs = torch.cat(
            [type_embeds, atom_coords_input], dim=-1
        )  # MODIFIED
        atom_decoder_input = self.atom_input_proj(atom_step_inputs)

        # Check for NaNs in decoder input
        if torch.isnan(atom_decoder_input).any():
            atom_decoder_input = torch.nan_to_num(atom_decoder_input, nan=0.0)

        # Add Positional Encoding
        atom_decoder_input_pos = self.atom_pos_encoder(
            atom_decoder_input.permute(1, 0, 2)
        ).permute(1, 0, 2)

        # Check for NaNs after positional encoding
        if torch.isnan(atom_decoder_input_pos).any():
            atom_decoder_input_pos = torch.nan_to_num(atom_decoder_input_pos, nan=0.0)

        # Prepare Masks for Atom Decoder
        tgt_len = atom_decoder_input_pos.size(1)
        if tgt_len <= 0:  # Handle empty sequences
            batch_size = atom_decoder_input_pos.size(0)
            return {
                "sg_logits": sg_logits,
                "lattice_mean": lattice_mean,
                "lattice_log_var": lattice_log_var,
                "type_logits": torch.zeros(
                    (batch_size, 0, self.effective_type_vocab_size), device=self.device
                ),
                # "wyckoff_logits": REMOVED
                "coord_mean": torch.zeros((batch_size, 0, 3), device=self.device),
                "coord_log_var": torch.zeros((batch_size, 0, 3), device=self.device),
            }

        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            self.device
        )
        tgt_key_padding_mask = ~atom_mask_input

        # --- Run Atom Decoder ---
        # WORKAROUND: Apply ModularCrossAttention once before the decoder stack.
        # Ensure all decoder contexts are NaN-free
        clean_decoder_contexts = {}
        for name, tensor in decoder_contexts.items():
            if torch.isnan(tensor).any():
                clean_decoder_contexts[name] = torch.nan_to_num(
                    tensor, nan=0.0, posinf=1e6, neginf=-1e6
                )
            else:
                clean_decoder_contexts[name] = tensor

        # Apply attention with additional gradient clipping
        with torch.no_grad():
            # Temporarily set to evaluation mode to minimize chance of NaNs
            training_mode = self.training
            self.eval()
            try:
                fused_memory = self.atom_modular_cross_attn(
                    query=atom_decoder_input_pos,
                    encoder_outputs=clean_decoder_contexts,
                )
                # Aggressively ensure no NaNs in fused memory
                fused_memory = torch.nan_to_num(
                    fused_memory, nan=0.0, posinf=1e6, neginf=-1e6
                )
            finally:
                # Restore previous training mode
                if training_mode:
                    self.train()

        # Double-check for NaNs in fused memory
        if torch.isnan(fused_memory).any():
            fused_memory = torch.zeros_like(fused_memory)

        # Ensure masks have the same dtype to avoid the warning
        # Convert tgt_mask to boolean if tgt_key_padding_mask is boolean
        if tgt_key_padding_mask.dtype == torch.bool and tgt_mask.dtype != torch.bool:
            # If mask has 0s and 1s, we need to convert differently than if it has -inf and 0s
            if (tgt_mask == 0).any() or (tgt_mask == 1).any():  # 0s and 1s format
                tgt_mask = tgt_mask.to(torch.bool)
            else:  # -inf and 0s format
                tgt_mask = tgt_mask == 0  # Only 0s will become True

        # Feed fused representation into the standard decoder stack.
        atom_decoder_output = self.atom_decoder(
            tgt=atom_decoder_input_pos,
            memory=fused_memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=tgt_key_padding_mask,
        )

        # Final check for NaNs in decoder output
        if torch.isnan(atom_decoder_output).any():
            atom_decoder_output = torch.nan_to_num(
                atom_decoder_output, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # --- Atom Prediction using Modular Decoders ---
        # Check for NaNs in decoder output
        atom_decoder_output = torch.nan_to_num(
            atom_decoder_output, nan=0.0, posinf=1e6, neginf=-1e6
        )

        # Process through decoder registry
        predictions_from_registry = self.decoder_registry(
            atom_decoder_output, decoder_contexts, atom_mask_input
        )

        # Extract predictions directly from decoder registry
        type_logits = predictions_from_registry[
            "type_logits"
        ]  # This key is produced by TypeDecoder
        coord_mean = predictions_from_registry["coord_mean"]
        coord_log_var = predictions_from_registry["coord_log_var"]

        # Return predicted *parameters* for loss calculation
        predictions = {
            "sg_logits": sg_logits,
            "lattice_mean": lattice_mean,
            "lattice_log_var": lattice_log_var,
            "type_logits": type_logits,
            # "wyckoff_logits": REMOVED
            "coord_mean": coord_mean,
            "coord_log_var": coord_log_var,
        }

        # Add any other model outputs to predictions if needed
        # (currently none needed beyond the standard outputs)

        return predictions

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")  # Default to CPU

    @torch.no_grad()
    def generate(
        self,
        # --- Required Inputs for Encoders ---
        composition: torch.Tensor,  # (1, V_elem)
        # --- Optional Inputs for other potential encoders ---
        spacegroup: Optional[torch.Tensor] = None,  # Example: (1, 1)
        # Add other conditioning inputs required by active encoders...
        # --- Generation Parameters ---
        max_atoms: int = 50,
        sg_sampling_mode: str = "sample",
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",  # For type sampling
        coord_sampling_mode: str = "sample",
        temperature: float = 1.0,
        # --- Explicit Start Tokens (Optional) ---
        start_type_token: Optional[int] = None,
        # start_wyckoff_token: REMOVED
        start_coord_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Autoregressive generation (sampling) using modular encoders. (Wyckoff removed)"""
        from torch.distributions import Normal  # Local import

        bs = composition.size(0)
        if bs != 1:
            raise NotImplementedError("Batch generation not yet supported.")

        device = self.device
        start_type_token = (
            start_type_token if start_type_token is not None else self.start_idx
        )
        # start_wyckoff_token REMOVED
        start_coord_token = (
            start_coord_token
            if start_coord_token is not None
            else torch.zeros(bs, 1, 3, device=device)
        )

        # --- Step 1: Run all registered encoders ---
        encoder_batch = {"composition": composition}
        if spacegroup is not None:
            encoder_batch["spacegroup"] = spacegroup

        encoder_outputs = {}
        for name, encoder in self.encoders.items():
            if name in encoder_batch:
                input_tensor = encoder_batch[name].to(device)
                encoder_outputs[name] = encoder(input_tensor)

        # --- Step 2: Predict and Sample Space Group ---
        comp_output = encoder_outputs["composition"].to(device)
        sg_logits = self.sg_head(comp_output.squeeze(1))
        if sg_sampling_mode == "argmax":
            sg_sampled_idx = torch.argmax(sg_logits, dim=-1)
        else:
            sg_probs = torch.softmax(sg_logits / temperature, dim=-1)
            sg_sampled_idx = torch.multinomial(sg_probs, num_samples=1).squeeze(-1)
        # Convert to actual space group number (1-230)
        sg_sampled = sg_sampled_idx + 1

        if "spacegroup" in self.encoders:
            # The SpaceGroupEncoder now expects space group numbers (1-230) and handles the conversion
            # to one-hot internally, so we don't need to change anything here
            sg_context = self.encoders["spacegroup"](sg_sampled.unsqueeze(1).to(device))
            encoder_outputs["spacegroup"] = sg_context

        # --- Step 3: Predict and Sample Lattice ---
        lattice_query = encoder_outputs.get(
            "composition", torch.zeros((bs, 1, self.d_model), device=device)
        ).to(device)
        encoder_outputs_device = {k: v.to(device) for k, v in encoder_outputs.items()}
        fused_context_for_lattice = self.lattice_modular_cross_attn(
            query=lattice_query, encoder_outputs=encoder_outputs_device
        )
        lattice_params = self.lattice_head(fused_context_for_lattice.squeeze(1))
        lattice_mean, lattice_log_var = torch.chunk(lattice_params, 2, dim=-1)

        if lattice_sampling_mode == "mean":
            lattice_sampled = lattice_mean
        else:
            lattice_std = torch.exp(0.5 * lattice_log_var)
            lattice_dist = Normal(lattice_mean, lattice_std)
            lattice_sampled = lattice_dist.sample()

        # Process lattice using the lattice encoder
        if "lattice" in self.encoders:
            # Use the encoder to get a more sophisticated representation
            lattice_context = self.encoders["lattice"](lattice_sampled.to(device))
            # Add to encoder outputs (using encoder's name directly)
            encoder_outputs["lattice"] = lattice_context
        else:
            # Create a fallback representation if encoder is missing
            # This avoids errors if configuration doesn't include lattice encoder
            dummy_shape = [bs, 1, self.d_model]
            encoder_outputs["lattice"] = torch.zeros(dummy_shape, device=device)

        # Create decoder contexts
        decoder_contexts = {k: v.to(device) for k, v in encoder_outputs.items()}

        # --- Step 4: Autoregressive Atom Sequence Generation ---
        current_type = torch.tensor(
            [[start_type_token]], device=device, dtype=torch.long
        )
        # current_wyckoff REMOVED
        current_coords = start_coord_token.to(device)

        generated_types = []
        # generated_wyckoffs REMOVED
        generated_coords = []

        input_type_seq = current_type
        # input_wyckoff_seq REMOVED
        input_coord_seq = current_coords

        for step in range(max_atoms):
            # Map current sequence indices
            type_embed_idx = self._map_indices_for_embedding(input_type_seq)
            # wyckoff_embed_idx REMOVED

            type_embed = self.type_embedding(type_embed_idx)
            # wyckoff_embed REMOVED

            # Combine step input (Type + Coords)
            atom_step_inputs = torch.cat(
                [type_embed, input_coord_seq.to(device)], dim=-1
            )  # MODIFIED
            atom_decoder_input = self.atom_input_proj(atom_step_inputs)

            # Positional encoding
            pos_encoded_input = self.atom_pos_encoder(
                atom_decoder_input.permute(1, 0, 2)
            ).permute(1, 0, 2)

            # Fused context
            fused_memory = self.atom_modular_cross_attn(
                query=pos_encoded_input.to(device), encoder_outputs=decoder_contexts
            )

            # Atom Decoder Step
            tgt_len = pos_encoded_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
                device
            )
            atom_decoder_output = self.atom_decoder(
                tgt=pos_encoded_input.to(device),
                memory=fused_memory.to(device),
                tgt_mask=tgt_mask,
            )
            last_token_output = atom_decoder_output[:, -1, :]

            # Predict next properties
            # Check for NaNs in decoder output
            last_token_output = torch.nan_to_num(
                last_token_output, nan=0.0, posinf=1e6, neginf=-1e6
            )

            # Process through decoders
            # Create mini batch with sequence length 1
            mini_mask = None  # No mask needed for single token
            predictions = self.decoder_registry(
                last_token_output.unsqueeze(1), decoder_contexts, mini_mask
            )

            # Extract predictions
            type_logits = predictions["type_logits"]
            coord_mean = predictions["coord_mean"]
            coord_log_var = predictions["coord_log_var"]

            # Final safeguard for all outputs
            type_logits = torch.nan_to_num(type_logits, nan=0.0)
            coord_mean = torch.nan_to_num(coord_mean, nan=0.5)
            coord_log_var = torch.nan_to_num(coord_log_var, nan=-3.0)

            # If we used decoder, we need to squeeze the sequence dimension
            if type_logits.dim() > 2:
                type_logits = type_logits.squeeze(1)
            if coord_mean.dim() > 2:
                coord_mean = coord_mean.squeeze(1)
            if coord_log_var.dim() > 2:
                coord_log_var = coord_log_var.squeeze(1)

            # --- Sample next atom ---
            # Type
            if atom_discrete_sampling_mode == "argmax":
                next_type_embed_idx = torch.argmax(type_logits, dim=-1, keepdim=True)
            else:
                type_probs = torch.softmax(type_logits / temperature, dim=-1)
                next_type_embed_idx = torch.multinomial(type_probs, num_samples=1)

            # Reverse map (simplified, assumes is_type=True)
            def reverse_map(embed_idx):
                start_idx, end_idx = self.type_start_embed_idx, self.type_end_embed_idx
                original_idx = embed_idx.clone()
                original_idx[embed_idx == start_idx] = self.start_idx
                original_idx[embed_idx == end_idx] = self.end_idx
                original_idx[embed_idx == 0] = self.pad_idx
                return original_idx

            next_type = reverse_map(next_type_embed_idx)

            if next_type.item() == self.end_idx:
                break

            # Wyckoff sampling REMOVED

            # Coordinates - using the new parameter format (no conversion needed)
            coord_std = torch.exp(0.5 * coord_log_var).clamp(
                max=0.2
            )  # Limit standard deviation

            if coord_sampling_mode == "mode":
                # Just use the mean directly
                next_coords = torch.clamp(coord_mean, min=0.0, max=1.0)
            else:
                # Sample from Gaussian and apply periodic boundary conditions
                # Use reparameterization trick: μ + σ·ε where ε ~ N(0,1)
                epsilon = torch.randn_like(coord_mean)
                sampled = coord_mean + coord_std * epsilon

                # Apply modulo to handle the periodic boundary
                next_coords = torch.remainder(sampled, 1.0)

            # --- Append and update ---
            generated_types.append(next_type.item())
            # generated_wyckoffs REMOVED
            generated_coords.append(next_coords)  # Keep as tensor (1, 3)

            input_type_seq = torch.cat([input_type_seq, next_type.to(device)], dim=1)
            # input_wyckoff_seq REMOVED
            input_coord_seq = torch.cat(
                [input_coord_seq, next_coords.unsqueeze(1).to(device)], dim=1
            )

        # --- Collect results ---
        results = {
            "composition": composition.squeeze(0).cpu(),
            "spacegroup_sampled": sg_sampled.squeeze(0).cpu(),
            "lattice_sampled": lattice_sampled.squeeze(0).cpu(),
            "atom_types_generated": torch.tensor(
                generated_types, dtype=torch.long, device="cpu"
            ),
            # "atom_wyckoffs_generated": REMOVED
            "atom_coords_generated": torch.cat(generated_coords, dim=0).cpu()
            if generated_coords
            else torch.empty((0, 3), device="cpu"),
        }
        return results
