import torch
import torch.nn as nn
from typing import Dict, Any, Optional, List

from mattermake.models.components.modular_encoders import (
    CompositionEncoder,
    SpaceGroupEncoder,
    LatticeEncoder,
    AtomTypeEncoder,
    AtomCoordinateEncoder,
)
from mattermake.models.components.modular_decoders import (
    SpaceGroupDecoder,
    LatticeDecoder,
    AtomTypeDecoder,
    AtomCoordinateDecoder,
    OrderedDecoderRegistry,
)

from mattermake.models.components.modular_attention import ModularCrossAttention
from mattermake.models.components.positional_embeddings import RotaryPositionalEmbedding


class ModularCrystalTransformerBase(nn.Module):
    """
    Fully modular crystal transformer with separate encoders and decoders for all components.
    Generates lattice params, atom types, and coordinates with dedicated modular components.
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
            # Default config if none provided
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
                "atom_type": {
                    "element_vocab_size": element_vocab_size,
                    "type_embed_dim": type_embed_dim,
                    "num_layers": 2,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                    "dropout": dropout,
                    "pad_idx": pad_idx,
                    "start_idx": start_idx,
                    "end_idx": end_idx,
                },
                "coordinate": {
                    "hidden_dim": 128,
                    "num_layers": 2,
                    "nhead": nhead,
                    "dim_feedforward": dim_feedforward,
                },
            }
        self.encoder_configs = encoder_configs

        # Store the encoding order - default is alphabetical if not specified
        if encoding_order is None:
            encoding_order = sorted(encoder_configs.keys())
        self.encoding_order = encoding_order

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
                    has_conditioning=config.get("has_conditioning", False),
                )
            elif name == "lattice":
                self.encoders[name] = LatticeEncoder(
                    d_model=d_model,
                    latent_dim=config.get("latent_dim", 64),
                    equivariant=config.get("equivariant", False),
                    has_conditioning=config.get("has_conditioning", False),
                )
            elif name == "atom_type":
                self.encoders[name] = AtomTypeEncoder(
                    element_vocab_size=config.get(
                        "element_vocab_size", element_vocab_size
                    ),
                    type_embed_dim=config.get("type_embed_dim", type_embed_dim),
                    d_model=d_model,
                    nhead=config.get("nhead", nhead),
                    num_layers=config.get("num_layers", 2),
                    dim_feedforward=config.get("dim_feedforward", dim_feedforward),
                    dropout=config.get("dropout", dropout),
                    pad_idx=config.get("pad_idx", pad_idx),
                    start_idx=config.get("start_idx", start_idx),
                    end_idx=config.get("end_idx", end_idx),
                    has_conditioning=config.get("has_conditioning", False),
                )
            elif name == "coordinate":
                self.encoders[name] = AtomCoordinateEncoder(
                    d_model=d_model,
                    hidden_dim=config.get("hidden_dim", 128),
                    num_layers=config.get("num_layers", 2),
                    nhead=config.get("nhead", nhead),
                    dim_feedforward=config.get("dim_feedforward", dim_feedforward),
                    dropout=config.get("dropout", dropout),
                    has_conditioning=config.get("has_conditioning", False),
                )
            else:
                print(f"Warning: Unknown encoder type '{name}' in config. Skipping.")

        # --- Create Rotary Positional Encoding for sequence processing ---
        self.atom_pos_encoder = RotaryPositionalEmbedding(d_model, dropout)

        # --- Store encoder dependencies ---
        self.encoder_dependencies = {}
        for name, config in encoder_configs.items():
            if "depends_on" in config:
                self.encoder_dependencies[name] = config["depends_on"]

        # --- Validate encoding order against dependencies ---
        self._validate_encoding_order()

        # --- Compute effective vocabulary size for atom types ---
        self._max_element_idx = self.hparams.get("element_vocab_size", 100)
        self.effective_type_vocab_size = self._max_element_idx + 3  # PAD, START, END

        # --- Initialize Modular Decoders ---
        if decoder_configs is None:
            # Default decoder configuration
            decoder_configs = {
                "spacegroup": {"sg_vocab_size": 230},  # 1-230 range
                "lattice": {},  # Default parameters
                "atom_type": {"vocab_size": self.effective_type_vocab_size},
                "coordinate": {"eps": eps},
            }

        # Create decoders
        decoders = {}
        for name, config in decoder_configs.items():
            # Ensure config is a dictionary even if it's None
            config = config if config is not None else {}

            if name == "spacegroup":
                decoders[name] = SpaceGroupDecoder(
                    d_model=d_model,
                    sg_vocab_size=config.get("sg_vocab_size", 230),
                )
            elif name == "lattice":
                decoders[name] = LatticeDecoder(
                    d_model=d_model,
                )
            elif name == "atom_type":
                decoders[name] = AtomTypeDecoder(
                    d_model=d_model,
                    vocab_size=config.get("vocab_size", self.effective_type_vocab_size),
                )
            elif name == "coordinate":
                decoders[name] = AtomCoordinateDecoder(
                    d_model=d_model, eps=config.get("eps", eps)
                )
            else:
                print(f"Warning: Unknown decoder type '{name}' in config. Skipping.")

        # Default decoding order if not specified
        if decoding_order is None:
            decoding_order = ["spacegroup", "lattice", "atom_type", "coordinate"]

        # Create ordered decoder registry
        self.decoder_registry = OrderedDecoderRegistry(decoders, order=decoding_order)

        # --- Create Attention Mechanisms & Atom Decoder ---
        # Separate attention for different prediction stages
        self.global_modular_cross_attn = ModularCrossAttention(
            encoder_names=list(
                filter(
                    lambda x: x not in ["atom_type", "coordinate"], self.encoders.keys()
                )
            ),
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Atom decoding uses all encoder contexts
        self.atom_decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.atom_modular_cross_attn = ModularCrossAttention(
            encoder_names=list(self.encoders.keys()),
            d_model=d_model,
            nhead=nhead,
            dropout=dropout,
        )

        # Standard decoder stack
        self.atom_decoder = nn.TransformerDecoder(
            self.atom_decoder_layer, num_layers=num_atom_decoder_layers
        )

    # The index mapping is now fully handled by the AtomTypeEncoder
    # No need for a separate method

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Teacher forcing forward pass for training with fully modular components"""

        # --- Step 1: Process all inputs and run encoders ---
        encoder_outputs = {}
        batch_size = batch["composition"].size(0)
        device = self.device

        # --- 1a. Process primary input: composition ---
        composition = batch["composition"].float()
        if torch.isnan(composition).any():
            composition = torch.nan_to_num(composition, nan=0.0)

        # --- 1b. Process primary input: space group ---
        # Note: Composition and space group are at the same hierarchical level
        sg_target = batch["spacegroup"]  # Values should be 1-230

        # --- 1c. Process secondary/dependent property: lattice matrix ---
        # (depends on composition and space group)
        # IMPORTANT: Assumes batch["lattice"] now contains the 3x3 matrix or flattened 9 elements
        lattice_matrix_target = batch["lattice"]
        if torch.isnan(lattice_matrix_target).any():
            lattice_matrix_target = torch.nan_to_num(lattice_matrix_target, nan=0.0)

        # --- 1d. Process atom sequence inputs ---
        atom_types_target = batch["atom_types"]
        atom_coords_target = batch["atom_coords"]
        atom_mask = batch["atom_mask"]

        if torch.isnan(atom_coords_target).any():
            atom_coords_target = torch.nan_to_num(atom_coords_target, nan=0.0)

        # --- 1e. Run encoders in specified order ---
        for name in self.encoding_order:
            if name not in self.encoders:
                continue

            encoder = self.encoders[name]
            try:
                # Check if this encoder has dependencies
                condition_context = None
                if (
                    name in self.encoder_dependencies
                    and self.encoder_dependencies[name]
                ):
                    # Gather outputs from dependency encoders
                    dep_contexts = []
                    for dep_name in self.encoder_dependencies[name]:
                        if dep_name in encoder_outputs:
                            dep_contexts.append(encoder_outputs[dep_name])

                    # Fuse contexts if we have any
                    if dep_contexts:
                        condition_context = self._fuse_contexts(dep_contexts)

                # Call the encoder with appropriate inputs and conditioning
                if name == "composition":
                    output = encoder(composition, condition_context)
                elif name == "spacegroup":
                    output = encoder(sg_target, condition_context)
                elif name == "lattice":  # Now encodes the matrix
                    output = encoder(lattice_matrix_target, condition_context)
                elif name == "atom_type":
                    # Use teacher forcing (shifted right for decoder inputs)
                    atom_types_input = atom_types_target[:, :-1]
                    atom_mask_input = atom_mask[:, :-1]
                    output = encoder(
                        atom_types_input, atom_mask_input, condition_context
                    )
                elif name == "coordinate":
                    # Use teacher forcing (shifted right for decoder inputs)
                    atom_coords_input = atom_coords_target[:, :-1]
                    atom_mask_input = atom_mask[:, :-1]
                    output = encoder(
                        atom_coords_input, atom_mask_input, condition_context
                    )
                elif name in batch:  # Generic fallback for other encoders
                    output = encoder(batch[name], condition_context)
                else:
                    print(
                        f"Warning: Input for encoder '{name}' not found in batch. Skipping."
                    )
                    continue

                # Check for NaNs in encoder output
                if torch.isnan(output).any():
                    output = torch.nan_to_num(output, nan=0.0, posinf=1e6, neginf=-1e6)

                encoder_outputs[name] = output
            except Exception as e:
                print(f"Error in encoder '{name}': {e}")
                # Create safe fallback
                seq_len = (
                    1
                    if name not in ["atom_type", "coordinate"]
                    else atom_types_target.size(1) - 1
                )
                dummy_shape = [batch_size, seq_len, self.d_model]
                encoder_outputs[name] = torch.zeros(dummy_shape, device=device)

        # --- Step 2: Global Encoding ---
        # Create a clean copy of encoder outputs (without atom-level encoders)
        global_encoder_outputs = {}
        for name, tensor in encoder_outputs.items():
            if name not in ["atom_type", "coordinate"]:
                if torch.isnan(tensor).any():
                    global_encoder_outputs[name] = torch.nan_to_num(
                        tensor, nan=0.0, posinf=1e6, neginf=-1e6
                    )
                else:
                    global_encoder_outputs[name] = tensor

        # --- Step 3: Run Space Group and Lattice Decoders ---
        # Get composition context for space group prediction
        comp_context = global_encoder_outputs.get(
            "composition", torch.zeros((batch_size, 1, self.d_model), device=device)
        )

        # Run space group decoder with composition context
        sg_decoder = self.decoder_registry.decoders["spacegroup"]
        sg_outputs = sg_decoder(comp_context, global_encoder_outputs, None)

        # Create the global context by fusing encoder contexts
        # This context is now used for lattice matrix prediction
        global_context = self.global_modular_cross_attn(
            query=comp_context, encoder_outputs=global_encoder_outputs
        )

        # Ensure global context has no NaNs
        if torch.isnan(global_context).any():
            global_context = torch.nan_to_num(
                global_context, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # Run lattice decoder (predicts matrix distribution) with fused global context
        lattice_decoder = self.decoder_registry.decoders["lattice"]
        # lattice_outputs will contain 'lattice_matrix_mean' and 'lattice_matrix_log_var'
        lattice_outputs = lattice_decoder(global_context, global_encoder_outputs, None)

        # --- Step 4: Atom Sequence Decoding (Teacher Forcing) ---
        # Prepare atom decoder input by combining atom type and coord encodings
        atom_type_encoded = encoder_outputs.get(
            "atom_type",
            torch.zeros(
                (batch_size, atom_types_target.size(1) - 1, self.d_model), device=device
            ),
        )

        coord_encoded = encoder_outputs.get(
            "coordinate",
            torch.zeros(
                (batch_size, atom_coords_target.size(1) - 1, self.d_model),
                device=device,
            ),
        )

        # Project the encoded representations together
        atom_decoder_input = (atom_type_encoded + coord_encoded) / 2.0

        # Apply Rotary Positional Embeddings
        atom_decoder_input_pos = self.atom_pos_encoder(atom_decoder_input)

        # Combine all encoder outputs for decoder context
        all_contexts = {}
        all_contexts.update(global_encoder_outputs)

        # Check shape and ensure no NaNs
        for name, tensor in all_contexts.items():
            if torch.isnan(tensor).any():
                all_contexts[name] = torch.nan_to_num(
                    tensor, nan=0.0, posinf=1e6, neginf=-1e6
                )

        # Prepare masks for atom decoder
        tgt_len = atom_decoder_input_pos.size(1)
        if tgt_len <= 0:  # Handle empty sequences
            batch_size = atom_decoder_input_pos.size(0)
            return {
                **sg_outputs,
                **lattice_outputs,
                "type_logits": torch.zeros(
                    (batch_size, 0, self.effective_type_vocab_size), device=device
                ),
                "coord_mean": torch.zeros((batch_size, 0, 3), device=device),
                "coord_log_var": torch.zeros((batch_size, 0, 3), device=device),
            }

        # Create masks for causal attention
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(device)
        atom_mask_input = atom_mask[:, :-1]  # Shifted right for teacher forcing
        tgt_key_padding_mask = ~atom_mask_input

        # Apply cross-attention to combine contexts
        fused_context = self.atom_modular_cross_attn(
            query=atom_decoder_input_pos, encoder_outputs=all_contexts
        )

        # Ensure no NaNs in fused context
        if torch.isnan(fused_context).any():
            fused_context = torch.nan_to_num(
                fused_context, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # Ensure masks have the same dtype
        if tgt_key_padding_mask.dtype == torch.bool and tgt_mask.dtype != torch.bool:
            if (tgt_mask == 0).any() or (tgt_mask == 1).any():  # 0s and 1s format
                tgt_mask = tgt_mask.to(torch.bool)
            else:  # -inf and 0s format
                tgt_mask = tgt_mask == 0  # Only 0s will become True

        # Run atom decoder
        atom_decoder_output = self.atom_decoder(
            tgt=atom_decoder_input_pos,
            memory=fused_context,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=None,  # No padding in memory
        )

        # Ensure no NaNs in decoder output
        if torch.isnan(atom_decoder_output).any():
            atom_decoder_output = torch.nan_to_num(
                atom_decoder_output, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # --- Step 5: Run Atom Type and Coordinate Decoders ---
        # Get atom type and coordinate decoders
        atom_type_decoder = self.decoder_registry.decoders["atom_type"]
        coord_decoder = self.decoder_registry.decoders["coordinate"]

        # Run the decoders
        atom_type_outputs = atom_type_decoder(
            atom_decoder_output, all_contexts, atom_mask_input
        )
        coord_outputs = coord_decoder(
            atom_decoder_output, all_contexts, atom_mask_input
        )

        # --- Step 6: Combine All Outputs ---
        predictions = {
            **sg_outputs,  # sg_logits
            **lattice_outputs,  # lattice_mean, lattice_log_var
            **atom_type_outputs,  # type_logits
            **coord_outputs,  # coord_mean, coord_log_var
        }

        return predictions

    def _validate_encoding_order(self):
        """Validate that encoding_order respects all encoder dependencies"""
        # Create a set of processed encoders to check dependencies against
        processed = set()

        for name in self.encoding_order:
            # Skip if encoder doesn't exist or isn't in dependencies
            if name not in self.encoders or name not in self.encoder_dependencies:
                processed.add(name)
                continue

            # Check if all dependencies have been processed before this encoder
            deps = self.encoder_dependencies[name]
            for dep in deps:
                if dep not in processed:
                    raise ValueError(
                        f"Encoder '{name}' depends on '{dep}' but appears before it "
                        f"in the encoding order. Please update encoding_order to respect dependencies."
                    )

            processed.add(name)

    def _fuse_contexts(self, contexts: List[torch.Tensor]) -> torch.Tensor:
        """Fuse multiple context tensors by averaging them.

        Args:
            contexts: List of context tensors to fuse

        Returns:
            Fused context tensor
        """
        if not contexts:
            return None

        # For now, just average the contexts
        # Can be enhanced with learned fusion later
        return torch.stack(contexts).mean(dim=0)

    @property
    def device(self):
        try:
            return next(self.parameters()).device
        except StopIteration:
            return torch.device("cpu")  # Default to CPU

    @torch.no_grad()
    def generate(
        self,
        # --- Primary Inputs for Encoders (at the same hierarchical level) ---
        composition: torch.Tensor,  # (1, V_elem) - Required input
        spacegroup: Optional[
            torch.Tensor
        ] = None,  # (1, 1) - Optional but treated as primary input when provided
        # --- Generation Parameters ---
        max_atoms: int = 50,
        sg_sampling_mode: str = "sample",  # Only used when spacegroup is None
        lattice_sampling_mode: str = "sample",
        atom_discrete_sampling_mode: str = "sample",  # For type sampling
        coord_sampling_mode: str = "sample",
        temperature: float = 1.0,
        # --- Explicit Start Tokens (Optional) ---
        start_type_token: Optional[int] = None,
        start_coord_token: Optional[torch.Tensor] = None,
    ) -> Dict[str, Any]:
        """Autoregressive generation (sampling) using the fully modular architecture"""
        from torch.distributions import Normal

        bs = composition.size(0)
        if bs != 1:
            raise NotImplementedError("Batch generation not yet supported.")

        device = self.device
        start_type_token = (
            start_type_token if start_type_token is not None else self.start_idx
        )
        start_coord_token = (
            start_coord_token
            if start_coord_token is not None
            else torch.zeros(bs, 1, 3, device=device)
        )

        # --- Step 1: Run all global encoders (composition, optionally space group) ---
        global_encoder_outputs = {}

        # Process encoders in the specified encoding order to respect dependencies
        for name in self.encoding_order:
            # Skip if not a global encoder or not available
            if name not in self.encoders or name in ["atom_type", "coordinate"]:
                continue

            # Check if this encoder has dependencies
            condition_context = None
            if name in self.encoder_dependencies and self.encoder_dependencies[name]:
                # Gather outputs from dependency encoders
                dep_contexts = []
                for dep_name in self.encoder_dependencies[name]:
                    if dep_name in global_encoder_outputs:
                        dep_contexts.append(global_encoder_outputs[dep_name])

                # Fuse contexts if we have any
                if dep_contexts:
                    condition_context = self._fuse_contexts(dep_contexts)

            # Process each encoder type with appropriate inputs
            if name == "composition":
                comp_output = self.encoders[name](
                    composition.to(device), condition_context
                )
                global_encoder_outputs["composition"] = comp_output
            elif name == "spacegroup":
                # Only encode if provided
                if spacegroup is not None:
                    sg_output = self.encoders[name](
                        spacegroup.to(device), condition_context
                    )
                    global_encoder_outputs["spacegroup"] = sg_output
            elif name == "lattice":
                # We'll encode lattice after we predict it below
                pass
            else:
                # Generic fallback for any other global encoders
                print(
                    f"Warning: Encoder '{name}' not handled specifically in generate. Skipping."
                )

        # --- Step 2: Process Space Group (use provided or generate) ---module
        # Space group is treated as a primary input at the same hierarchical level as composition
        # When provided directly, we use it; when not provided, we predict it from composition
        if spacegroup is None:
            # Space group not provided - predict it from composition
            # Get composition context for space group prediction
            comp_context = global_encoder_outputs.get(
                "composition", torch.zeros((bs, 1, self.d_model), device=device)
            )

            # Run space group decoder
            sg_decoder = self.decoder_registry.decoders["spacegroup"]
            sg_outputs = sg_decoder(comp_context, global_encoder_outputs, None)
            sg_logits = sg_outputs["sg_logits"]

            # Sample space group
            if sg_sampling_mode == "argmax":
                sg_sampled_idx = torch.argmax(sg_logits, dim=-1)
            else:
                sg_probs = torch.softmax(sg_logits / temperature, dim=-1)
                sg_sampled_idx = torch.multinomial(sg_probs, num_samples=1).squeeze(-1)

            # Convert to actual space group number (1-230)
            sg_sampled = sg_sampled_idx + 1

            # Encode the sampled space group
            if "spacegroup" in self.encoders:
                sg_context = self.encoders["spacegroup"](
                    sg_sampled.unsqueeze(1).to(device)
                )
                global_encoder_outputs["spacegroup"] = sg_context
        else:
            # Use the provided space group directly
            sg_sampled = spacegroup

            # Ensure we've encoded this primary input if it wasn't done in Step 1
            if (
                "spacegroup" not in global_encoder_outputs
                and "spacegroup" in self.encoders
            ):
                sg_context = self.encoders["spacegroup"](sg_sampled.to(device))
                global_encoder_outputs["spacegroup"] = sg_context

        # --- Step 3: Predict and Sample Lattice ---
        # Create global context for lattice prediction by fusing encoder outputs
        comp_context = global_encoder_outputs.get(
            "composition", torch.zeros((bs, 1, self.d_model), device=device)
        )

        global_context = self.global_modular_cross_attn(
            query=comp_context.to(device),
            encoder_outputs={
                k: v.to(device) for k, v in global_encoder_outputs.items()
            },
        )

        # Ensure no NaNs in global context
        if torch.isnan(global_context).any():
            global_context = torch.nan_to_num(
                global_context, nan=0.0, posinf=1e6, neginf=-1e6
            )

        # Run lattice decoder (predicts matrix distribution)
        lattice_decoder = self.decoder_registry.decoders["lattice"]
        lattice_outputs = lattice_decoder(global_context, global_encoder_outputs, None)
        # These have shape (B, 9)
        lattice_matrix_mean = lattice_outputs["lattice_matrix_mean"]
        lattice_matrix_log_var = lattice_outputs["lattice_matrix_log_var"]

        # Sample lattice matrix elements
        if lattice_sampling_mode == "mean":
            lattice_matrix_sampled_flat = lattice_matrix_mean
        else:
            lattice_matrix_std = torch.exp(0.5 * lattice_matrix_log_var)
            # Ensure std dev is reasonable
            lattice_matrix_std = torch.clamp(lattice_matrix_std, min=1e-6, max=5.0)
            lattice_matrix_dist = Normal(lattice_matrix_mean, lattice_matrix_std)
            lattice_matrix_sampled_flat = lattice_matrix_dist.sample()

        # Encode the sampled lattice matrix
        if "lattice" in self.encoders:
            # Check if lattice encoder has dependencies
            condition_context = None
            if (
                "lattice" in self.encoder_dependencies
                and self.encoder_dependencies["lattice"]
            ):
                # Gather outputs from dependency encoders
                dep_contexts = []
                for dep_name in self.encoder_dependencies["lattice"]:
                    if dep_name in global_encoder_outputs:
                        dep_contexts.append(global_encoder_outputs[dep_name])

                # Fuse contexts if we have any
                if dep_contexts:
                    condition_context = self._fuse_contexts(dep_contexts)

            # Pass the sampled matrix (flattened) to the encoder
            lattice_context = self.encoders["lattice"](
                lattice_matrix_sampled_flat.to(device), condition_context
            )
            global_encoder_outputs["lattice"] = lattice_context

        # --- Step 4: Autoregressive Atom Sequence Generation ---
        # Initialize sequence with start tokens
        current_type = torch.tensor(
            [[start_type_token]], device=device, dtype=torch.long
        )
        current_coords = start_coord_token.to(device)

        # For storing generated outputs
        generated_types = []
        generated_coords = []

        # Initial input sequences
        input_type_seq = current_type
        input_coord_seq = current_coords

        for step in range(max_atoms):
            # --- 4a. Get encoder representations ---
            # For each step, we need to encode the current sequence
            # Note: We're not actually using the atom_type_encoder and coordinate_encoder
            # since we need embeddings for a partial sequence

            # Encode atom types directly using the encoder
            atom_type_encoder = self.encoders["atom_type"]

            # Map special tokens for the encoder (START/END/PAD)
            start_embed_idx = self._max_element_idx + 1  # START token
            end_embed_idx = self._max_element_idx + 2  # END token

            # Convert special tokens
            mapped_types = input_type_seq.clone()
            mapped_types[input_type_seq == self.start_idx] = start_embed_idx
            mapped_types[input_type_seq == self.end_idx] = end_embed_idx
            mapped_types[input_type_seq == self.pad_idx] = 0

            # Directly encode using the encoder's components
            effective_vocab = atom_type_encoder.effective_vocab_size
            mapped_types_clamped = torch.clamp(mapped_types, 0, effective_vocab - 1)
            type_embed = atom_type_encoder.type_embedding(mapped_types_clamped)
            type_encoded = atom_type_encoder.projection(type_embed)

            # Encode coordinates directly using the encoder
            coord_encoder = self.encoders["coordinate"]
            coord_encoded = coord_encoder.coords_mlp(input_coord_seq)

            # Combine type and coordinate encodings
            atom_decoder_input = (type_encoded + coord_encoded) / 2.0

            # Apply Rotary Positional Embeddings with step offset
            pos_encoded_input = self.atom_pos_encoder(atom_decoder_input, offset=step)

            # --- 4b. Create fused context for transformer decoder ---
            # Combine global encoder outputs for context
            decoder_contexts = {
                k: v.to(device) for k, v in global_encoder_outputs.items()
            }

            # Apply cross-attention to combine contexts
            fused_context = self.atom_modular_cross_attn(
                query=pos_encoded_input.to(device), encoder_outputs=decoder_contexts
            )

            # --- 4c. Run transformer decoder ---
            tgt_len = pos_encoded_input.size(1)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
                device
            )

            atom_decoder_output = self.atom_decoder(
                tgt=pos_encoded_input.to(device),
                memory=fused_context.to(device),
                tgt_mask=tgt_mask,
            )

            # Extract last token output for prediction
            last_token_output = atom_decoder_output[:, -1, :]

            # Ensure no NaNs in decoder output
            last_token_output = torch.nan_to_num(
                last_token_output, nan=0.0, posinf=1e6, neginf=-1e6
            )

            # --- 4d. Predict next atom properties with decoders ---
            # Get atom type and coordinate decoders
            atom_type_decoder = self.decoder_registry.decoders["atom_type"]
            coord_decoder = self.decoder_registry.decoders["coordinate"]

            # Run decoders on the last token output
            atom_type_outputs = atom_type_decoder(
                last_token_output.unsqueeze(1), decoder_contexts, None
            )
            coord_outputs = coord_decoder(
                last_token_output.unsqueeze(1), decoder_contexts, None
            )

            # Extract predictions
            type_logits = atom_type_outputs["type_logits"].squeeze(
                1
            )  # Remove seq_len dimension
            coord_mean = coord_outputs["coord_mean"].squeeze(1)
            coord_log_var = coord_outputs["coord_log_var"].squeeze(1)

            # Final safeguard for all outputs
            type_logits = torch.nan_to_num(type_logits, nan=0.0)
            coord_mean = torch.nan_to_num(coord_mean, nan=0.5)
            coord_log_var = torch.nan_to_num(coord_log_var, nan=-3.0)

            # --- 4e. Sample next atom type ---
            if atom_discrete_sampling_mode == "argmax":
                next_type_embed_idx = torch.argmax(type_logits, dim=-1, keepdim=True)
            else:
                type_probs = torch.softmax(type_logits / temperature, dim=-1)
                next_type_embed_idx = torch.multinomial(type_probs, num_samples=1)

            # Map embedding indices back to actual atom type indices
            # Need to recalculate the token indices here for proper mapping
            start_embed_idx = (
                self._max_element_idx + 1
            )  # START token index in vocabulary
            end_embed_idx = self._max_element_idx + 2  # END token index in vocabulary

            next_type = next_type_embed_idx.clone()
            next_type[next_type_embed_idx == start_embed_idx] = self.start_idx
            next_type[next_type_embed_idx == end_embed_idx] = self.end_idx
            next_type[next_type_embed_idx == 0] = self.pad_idx

            # Check for END token
            if next_type.item() == self.end_idx:
                break

            # --- 4f. Sample next coordinates ---
            coord_std = torch.exp(0.5 * coord_log_var).clamp(
                max=0.2
            )  # Limit standard deviation

            if coord_sampling_mode == "mode":
                # Just use the mean directly
                next_coords = torch.clamp(coord_mean, min=0.0, max=1.0)
            else:
                # Sample from Gaussian and apply periodic boundary conditions
                epsilon = torch.randn_like(coord_mean)
                sampled = coord_mean + coord_std * epsilon

                # Apply modulo to handle periodic boundary
                next_coords = torch.remainder(sampled, 1.0)

            # --- 4g. Append and update sequences ---
            generated_types.append(next_type.item())
            generated_coords.append(next_coords)  # Keep as tensor (1, 3)

            input_type_seq = torch.cat([input_type_seq, next_type.to(device)], dim=1)
            input_coord_seq = torch.cat(
                [input_coord_seq, next_coords.unsqueeze(1).to(device)], dim=1
            )

        # --- Step 5: Collect results ---
        # Reshape sampled lattice matrix back to 3x3
        lattice_matrix_sampled = lattice_matrix_sampled_flat.reshape(3, 3)
        results = {
            "composition": composition.squeeze(0).cpu(),
            "spacegroup_sampled": sg_sampled.squeeze(0).cpu(),
            "lattice_matrix_sampled": lattice_matrix_sampled.cpu(),  # Return 3x3 matrix
            "atom_types_generated": torch.tensor(
                generated_types, dtype=torch.long, device="cpu"
            ),
            "atom_coords_generated": torch.cat(generated_coords, dim=0).cpu()
            if generated_coords
            else torch.empty((0, 3), device="cpu"),
        }

        return results
