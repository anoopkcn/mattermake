import torch
import torch.nn as nn
import math
from torch.distributions import VonMises, Normal, Categorical
from typing import Dict, Any


class PositionalEncoding(nn.Module):
    """Standard Positional Encoding"""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 500):
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
        """x: Tensor, shape [seq_len, batch_size, embedding_dim]"""
        # x.size(0) is the sequence length
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


class HierarchicalCrystalTransformerBase(nn.Module):
    """
    Base class for Hierarchical Crystal Transformer with dedicated encoders/projectors,
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
        # --- Additional Params ---
        eps: float = 1e-6,  # Small value for stability (e.g., variance)
        **kwargs,
    ):
        super().__init__()
        self.hparams = {k: v for k, v in locals().items() if k != 'self' and k != 'kwargs' and k != '__class__'}
        self.hparams.update(kwargs)
        self.eps = eps  # Store eps

        # --- Input Processing / Encoders / Projectors ---

        # 1. Composition Encoder
        self.comp_processor = nn.Linear(self.hparams['element_vocab_size'], d_model)
        comp_encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.comp_encoder = nn.TransformerEncoder(
            comp_encoder_layer, self.hparams['num_comp_encoder_layers']
        )

        # 2. Space Group Representation
        # Embedding for SG (input 0-230 -> index 0-230)
        # Note: Ensure sg_vocab_size includes an index for 0 if used/needed.
        self.sg_embedding = nn.Embedding(
            self.hparams['sg_vocab_size'], self.hparams['sg_embed_dim']
        )
        # Project SG embedding to d_model
        self.sg_projector = nn.Linear(self.hparams['sg_embed_dim'], d_model)

        # 3. Lattice Representation
        # Project 6 lattice params to d_model
        self.lattice_projector = nn.Linear(6, d_model)

        # 4. Atom Sequence Preparation (Embeddings + Input Projection)
        # Calculate effective vocab sizes for atom embeddings including PAD, START, END
        # Assumes element_vocab_size = max_Z, wyckoff_vocab_size = max_valid_idx + 1
        # These calculations need careful verification based on actual data range & config values
        self._max_element_idx = self.hparams['element_vocab_size']  # Max Z (e.g., 100)
        self._max_wyckoff_idx = (
            self.hparams['wyckoff_vocab_size'] - 1
        )  # Max valid index (e.g., 199)
        # Map: PAD(0)->0, Valid(1..N)->1..N, START(-1)->N+1, END(-2)->N+2
        self.effective_type_vocab_size = (
            self._max_element_idx + 3
        )  # e.g., 100 + 3 = 103
        self.type_start_embed_idx = self._max_element_idx + 1
        self.type_end_embed_idx = self._max_element_idx + 2

        self.effective_wyckoff_vocab_size = (
            self._max_wyckoff_idx + 3
        )  # e.g., 199 + 3 = 202
        self.wyckoff_start_embed_idx = self._max_wyckoff_idx + 1
        self.wyckoff_end_embed_idx = self._max_wyckoff_idx + 2

        self.type_embedding = nn.Embedding(
            self.effective_type_vocab_size,
            self.hparams['type_embed_dim'],
            padding_idx=self.hparams['pad_idx'],
        )
        self.wyckoff_embedding = nn.Embedding(
            self.effective_wyckoff_vocab_size,
            self.hparams['wyckoff_embed_dim'],
            padding_idx=self.hparams['pad_idx'],
        )

        # Combine atom step info (type emb + wyckoff emb + coords) and project
        atom_step_dim = self.hparams['type_embed_dim'] + self.hparams['wyckoff_embed_dim'] + 3
        self.atom_input_proj = nn.Linear(atom_step_dim, d_model)
        self.atom_pos_encoder = PositionalEncoding(d_model, dropout)

        # 5. Atom Decoder (Attends to Comp, SG, Lattice context)
        atom_decoder_layer = nn.TransformerDecoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.atom_decoder = nn.TransformerDecoder(
            atom_decoder_layer, self.hparams['num_atom_decoder_layers']
        )

        # --- Prediction Heads (Modified for Distributions) ---
        # Predict SG logits (index 0-230 or 0-229?) -> Map to 1-230
        self.sg_head = nn.Linear(d_model, self.hparams['sg_vocab_size'])

        # Lattice Head: Predicts 6 means and 6 log_vars for diagonal Normal
        lattice_cond_dim = d_model
        if self.hparams['condition_lattice_on_sg']:
            # Context dim needs careful checking depending on how context is combined (cat/add)
            lattice_cond_dim += d_model  # Assuming concatenation for simplicity
            # Alternative: Use cross attention layer before the head
        self.lattice_head_combiner = nn.Linear(
            lattice_cond_dim, d_model
        )  # Optional combiner layer
        self.lattice_head = nn.Linear(d_model, 6 * 2)  # mean + log_var per param

        # Atom Heads
        self.type_head = nn.Linear(d_model, self.effective_type_vocab_size)
        self.wyckoff_head = nn.Linear(d_model, self.effective_wyckoff_vocab_size)
        # Coord Head: Predicts loc and concentration for Von Mises per x, y, z
        self.coord_head = nn.Linear(d_model, 3 * 2)  # (loc, concentration) per dim
        
    def _map_indices_for_embedding(
        self, indices: torch.Tensor, is_type: bool
    ) -> torch.Tensor:
        """Maps PAD, START, END indices to non-negative indices for embedding lookup."""
        start_embed_idx = (
            self.type_start_embed_idx if is_type else self.wyckoff_start_embed_idx
        )
        end_embed_idx = (
            self.type_end_embed_idx if is_type else self.wyckoff_end_embed_idx
        )

        mapped_indices = indices.clone()
        # Map special tokens first
        mapped_indices[indices == self.hparams['start_idx']] = start_embed_idx
        mapped_indices[indices == self.hparams['end_idx']] = end_embed_idx
        # Ensure PAD remains 0
        mapped_indices[indices == self.hparams['pad_idx']] = self.hparams['pad_idx']
        # Ensure valid indices (>=1) remain unchanged
        valid_mask = indices >= 1
        mapped_indices[valid_mask] = indices[valid_mask]
        return mapped_indices

    def forward(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Teacher forcing forward pass for training. Predicts distribution parameters."""
        composition = batch["composition"].float()
        sg_target = batch["spacegroup"]  # (b, 1), values 1-230
        lattice_target = batch["lattice"]
        atom_types_target = batch["atom_types"]
        atom_wyckoffs_target = batch["atom_wyckoffs"]
        atom_coords_target = batch["atom_coords"]  # Target coords in [0, 1)
        atom_mask = batch["atom_mask"]

        # --- Stage 1: Composition Encoding ---
        comp_input = self.comp_processor(composition).unsqueeze(1)
        comp_memory = self.comp_encoder(comp_input)  # (b, 1, d)

        # --- Stage 2: Space Group Prediction & Encoding ---
        # Predict SG logits based on Comp memory
        sg_logits = self.sg_head(comp_memory.squeeze(1))  # (b, V_sg)

        # Encode Target SG for conditioning downstream tasks
        # Map SG target 1-230 to an embedding index (e.g., 0-229 or 1-230)
        # Assuming sg_head predicts 0..N-1, map target 1..N -> 0..N-1
        sg_target_idx = sg_target.squeeze(1) - 1
        # Handle potential padding index if sg_vocab_size included it explicitly
        # sg_target_idx = sg_target.squeeze(1) # If head predicts 0..N and target is 0..N

        sg_target_embed = self.sg_embedding(sg_target_idx)  # (b, sg_emb_dim)
        sg_proj = self.sg_projector(sg_target_embed).unsqueeze(1)  # (b, 1, d)

        # --- Stage 3: Lattice Prediction & Encoding ---
        # Prepare context for lattice prediction
        if self.hparams['condition_lattice_on_sg']:
            # Combine Comp and SG context
            # Simple concatenation along feature dim for linear head
            combined_context_features = torch.cat(
                [comp_memory.squeeze(1), sg_proj.squeeze(1)], dim=-1
            )  # (b, d+d)
            # Optional: Project combined context before feeding to head
            lattice_context_combined = self.lattice_head_combiner(
                combined_context_features
            )  # (b, d)
        else:
            lattice_context_combined = comp_memory.squeeze(1)  # Use only comp context

        # Predict parameters for Normal distribution (mean, log_var)
        lattice_params = self.lattice_head(lattice_context_combined)  # (b, 12)
        lattice_mean, lattice_log_var = torch.chunk(
            lattice_params, 2, dim=-1
        )  # (b, 6), (b, 6)

        # Encode Target Lattice for conditioning atom decoder
        lattice_proj = self.lattice_projector(lattice_target).unsqueeze(1)  # (b, 1, d)

        # --- Stage 4: Atom Sequence Decoding ---
        # Prepare combined memory context for the Atom Decoder's cross-attention
        decoder_memory = torch.cat(
            [comp_memory, sg_proj, lattice_proj], dim=1
        )  # (b, 3, d)

        # Prepare Atom Decoder Inputs (Shifted Right)
        atom_types_input_raw = atom_types_target[:, :-1]
        atom_wyckoffs_input_raw = atom_wyckoffs_target[:, :-1]
        atom_coords_input = atom_coords_target[:, :-1, :]  # Use target coords for input
        atom_mask_input = atom_mask[:, :-1]  # Mask for the input sequence

        # Map indices -> Embedding indices
        types_embed_idx = self._map_indices_for_embedding(
            atom_types_input_raw, is_type=True
        )
        wyckoffs_embed_idx = self._map_indices_for_embedding(
            atom_wyckoffs_input_raw, is_type=False
        )

        type_embeds = self.type_embedding(types_embed_idx)  # (b, T-1, type_emb_dim)
        wyckoff_embeds = self.wyckoff_embedding(
            wyckoffs_embed_idx
        )  # (b, T-1, wyckoff_emb_dim)

        # Combine and project atom step inputs
        atom_step_inputs = torch.cat(
            [type_embeds, wyckoff_embeds, atom_coords_input], dim=-1
        )
        atom_decoder_input = self.atom_input_proj(atom_step_inputs)  # (b, T-1, d)

        # Add Positional Encoding
        # Permute for PositionalEncoding: (T-1, b, d)
        atom_decoder_input = self.atom_pos_encoder(
            atom_decoder_input.permute(1, 0, 2)
        ).permute(1, 0, 2)  # (b, T-1, d)

        # Prepare Masks for Atom Decoder
        tgt_len = atom_decoder_input.size(1)  # T-1
        
        # Check if we have a valid sequence length
        if tgt_len <= 0:
            # Handle the case where sequence length is invalid
            batch_size = atom_decoder_input.size(0)
            return {
                "sg_logits": self.sg_head(comp_memory.squeeze(1)),  # Still provide SG logits
                "lattice_mean": torch.zeros((batch_size, 6), device=self.device),
                "lattice_log_var": torch.zeros((batch_size, 6), device=self.device),
                "type_logits": torch.zeros((batch_size, 0, self.effective_type_vocab_size), device=self.device),
                "wyckoff_logits": torch.zeros((batch_size, 0, self.effective_wyckoff_vocab_size), device=self.device),
                "coord_loc": torch.zeros((batch_size, 0, 3), device=self.device),
                "coord_concentration": torch.zeros((batch_size, 0, 3), device=self.device),
            }
        
        tgt_mask = nn.Transformer.generate_square_subsequent_mask(tgt_len).to(
            self.device
        )  # Causal mask (T-1, T-1)
        tgt_key_padding_mask = ~atom_mask_input  # True where padded (b, T-1)
        memory_key_padding_mask = None  # Context length is fixed

        # Run Atom Decoder
        atom_decoder_output = self.atom_decoder(
            tgt=atom_decoder_input,  # (b, T-1, d)
            memory=decoder_memory,  # (b, 3, d) - Cross-attends to this context
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=memory_key_padding_mask,
        )  # (b, T-1, d)

        # --- Atom Prediction Heads ---
        type_logits = self.type_head(atom_decoder_output)
        wyckoff_logits = self.wyckoff_head(atom_decoder_output)
        # Predict parameters for Von Mises distribution (loc, concentration)
        coord_params = self.coord_head(atom_decoder_output)  # (b, T-1, 6)
        coord_loc, coord_concentration_raw = torch.chunk(
            coord_params, 2, dim=-1
        )  # (b, T-1, 3) each
        # Ensure loc is in [-pi, pi) and concentration > 0
        coord_loc = torch.tanh(coord_loc) * math.pi  # Map output to [-pi, pi]
        coord_concentration = (
            torch.nn.functional.softplus(coord_concentration_raw) + self.eps
        )  # κ > 0

        # Return predicted *parameters* for loss calculation
        predictions = {
            "sg_logits": sg_logits,  # (b, V_sg)
            "lattice_mean": lattice_mean,  # (b, 6)
            "lattice_log_var": lattice_log_var,  # (b, 6)
            "type_logits": type_logits,  # (b, T-1, V_type_eff)
            "wyckoff_logits": wyckoff_logits,  # (b, T-1, V_wyck_eff)
            "coord_loc": coord_loc,  # (b, T-1, 3) in [-pi, pi]
            "coord_concentration": coord_concentration,  # (b, T-1, 3) > 0
        }
        return predictions

    @property
    def device(self):
        return next(self.parameters()).device
        
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
        """Autoregressive generation for HCT (sampling)."""
        self.eval()
        device = self.device
        composition = composition.float().to(device)

        # --- Stage 1: Comp Encoding ---
        comp_input = self.comp_processor(composition).unsqueeze(1)
        comp_memory = self.comp_encoder(comp_input)

        # --- Stage 2: SG Prediction & Encoding ---
        sg_logits = self.sg_head(comp_memory.squeeze(1))
        if sg_sampling_mode == "sample":
            sg_dist = Categorical(logits=sg_logits / temperature)  # Apply temperature
            predicted_sg_idx = sg_dist.sample()  # Sampled index 0-229 (or 0-230)
        else:  # argmax
            predicted_sg_idx = torch.argmax(sg_logits, dim=-1)

        predicted_sg_embed = self.sg_embedding(predicted_sg_idx)
        sg_proj = self.sg_projector(predicted_sg_embed).unsqueeze(1)
        # Map back to 1-230 number (adjust if head predicts different range)
        predicted_sg_num = predicted_sg_idx + 1

        # --- Stage 3: Lattice Distribution & Sampling/Mean ---
        if self.hparams['condition_lattice_on_sg']:
            combined_context_features = torch.cat(
                [comp_memory.squeeze(1), sg_proj.squeeze(1)], dim=-1
            )
            lattice_context_combined = self.lattice_head_combiner(
                combined_context_features
            )
        else:
            lattice_context_combined = comp_memory.squeeze(1)
        lattice_params = self.lattice_head(lattice_context_combined)
        lattice_mean, lattice_log_var = torch.chunk(lattice_params, 2, dim=-1)

        if lattice_sampling_mode == "sample":
            lattice_std = torch.exp(0.5 * lattice_log_var) + self.eps
            lattice_dist = Normal(lattice_mean, lattice_std)
            sampled_or_mean_lattice = lattice_dist.sample()  # (1, 6)
        else:  # mean
            sampled_or_mean_lattice = lattice_mean  # (1, 6)

        # Encode the sampled/mean lattice for atom decoder context
        lattice_proj = self.lattice_projector(sampled_or_mean_lattice).unsqueeze(
            1
        )  # (1, 1, d)

        # --- Stage 4: Atom Generation ---
        decoder_memory = torch.cat(
            [comp_memory, sg_proj, lattice_proj], dim=1
        )  # (1, 3, d)
        # Initialize with START tokens/coords
        start_type_raw = torch.tensor(
            [[self.hparams['start_idx']]], dtype=torch.long, device=device
        )
        start_wyckoff_raw = torch.tensor(
            [[self.hparams['start_idx']]], dtype=torch.long, device=device
        )
        start_coords = torch.zeros(
            (1, 1, 3), dtype=torch.float, device=device
        )  # Placeholder

        current_types_embed_idx = self._map_indices_for_embedding(
            start_type_raw, is_type=True
        )
        current_wyckoffs_embed_idx = self._map_indices_for_embedding(
            start_wyckoff_raw, is_type=False
        )
        current_coords = start_coords  # Use placeholder START coords for first input

        generated_types_raw = []
        generated_wyckoffs_raw = []
        generated_coords = []  # Store sampled coords [0, 1)

        for _ in range(max_atoms):
            # Prepare decoder input
            type_embed = self.type_embedding(current_types_embed_idx)
            wyckoff_embed = self.wyckoff_embedding(current_wyckoffs_embed_idx)
            # Input uses previously generated/sampled coords
            atom_step_inputs = torch.cat(
                [type_embed, wyckoff_embed, current_coords], dim=-1
            )
            atom_decoder_input = self.atom_input_proj(atom_step_inputs)
            atom_decoder_input_pos = self.atom_pos_encoder(
                atom_decoder_input.permute(1, 0, 2)
            ).permute(1, 0, 2)

            current_len = atom_decoder_input_pos.size(1)
            # Check for valid sequence length
            if current_len <= 0:
                # If we somehow get an empty sequence, just return early with empty results
                return {
                    "predicted_sg": predicted_sg_num.item(),
                    "sampled_lattice": sampled_or_mean_lattice.squeeze(0).cpu().numpy(),
                    "generated_types": [],
                    "generated_wyckoffs": [],
                    "generated_coords": [],  # fractional [0, 1)
                }
                
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(current_len).to(
                device
            )

            # Run decoder
            atom_decoder_output = self.atom_decoder(
                tgt=atom_decoder_input_pos,
                memory=decoder_memory,
                tgt_mask=tgt_mask,
            )
            last_step_output = atom_decoder_output[:, -1, :]

            # Predict next step parameters/logits
            next_type_logits = self.type_head(last_step_output)
            next_wyckoff_logits = self.wyckoff_head(last_step_output)
            next_coord_params = self.coord_head(last_step_output)
            next_coord_loc, next_coord_conc_raw = torch.chunk(
                next_coord_params, 2, dim=-1
            )
            next_coord_loc = torch.tanh(next_coord_loc) * math.pi
            next_coord_conc = (
                torch.nn.functional.softplus(next_coord_conc_raw) + self.eps
            )

            # --- Sampling / Argmax for Discrete ---
            if atom_discrete_sampling_mode == "sample":
                type_dist = Categorical(logits=next_type_logits / temperature)
                next_type_embed_idx = type_dist.sample()
                wyckoff_dist = Categorical(logits=next_wyckoff_logits / temperature)
                next_wyckoff_embed_idx = wyckoff_dist.sample()
            else:  # argmax
                next_type_embed_idx = torch.argmax(next_type_logits, dim=-1)
                next_wyckoff_embed_idx = torch.argmax(next_wyckoff_logits, dim=-1)

            # --- Sampling / Mode for Coordinates ---
            if coord_sampling_mode == "sample":
                coord_dist = VonMises(next_coord_loc, next_coord_conc)
                sampled_coord_angle = coord_dist.sample()  # (1, 3) in [-pi, pi)
            else:  # mode ('loc')
                sampled_coord_angle = next_coord_loc  # (1, 3) in [-pi, pi)

            # Map sampled angle back to [0, 1)
            sampled_coord_frac = (sampled_coord_angle + math.pi) / (
                2 * math.pi
            )  # (1, 3)

            # --- Map back from embed index to raw index ---
            # (Using the same reverse_map function as before)
            def reverse_map(embed_idx, is_type):
                max_valid = self._max_element_idx if is_type else self._max_wyckoff_idx
                start_embed = (
                    self.type_start_embed_idx
                    if is_type
                    else self.wyckoff_start_embed_idx
                )
                end_embed = (
                    self.type_end_embed_idx if is_type else self.wyckoff_end_embed_idx
                )
                if embed_idx == start_embed:
                    return self.hparams['start_idx']
                if embed_idx == end_embed:
                    return self.hparams['end_idx']
                if embed_idx == self.hparams['pad_idx']:
                    return self.hparams['pad_idx']
                # Clamp embed_idx just in case prediction is out of bounds before check
                embed_idx_item = embed_idx.clamp(min=0).item()
                if 1 <= embed_idx_item <= max_valid:
                    return embed_idx_item
                print(
                    f"Warning: Unexpected embed index predicted: {embed_idx.item()}, mapping to END."
                )
                return self.hparams['end_idx']

            next_type_raw = reverse_map(next_type_embed_idx, is_type=True)
            next_wyckoff_raw = reverse_map(next_wyckoff_embed_idx, is_type=False)

            # Check for END token
            if next_type_raw == self.hparams['end_idx']:
                break

            # Store generated atom (use original indices and sampled frac coords)
            if (
                next_type_raw != self.hparams['start_idx']
                and next_type_raw != self.hparams['pad_idx']
            ):
                generated_types_raw.append(next_type_raw)
                generated_wyckoffs_raw.append(next_wyckoff_raw)
                generated_coords.append(sampled_coord_frac.squeeze(0).cpu().numpy())

            # Prepare input for the next step (append *embedding* indices and *sampled* coords)
            current_types_embed_idx = torch.cat(
                [current_types_embed_idx, next_type_embed_idx.unsqueeze(0)], dim=1
            )
            current_wyckoffs_embed_idx = torch.cat(
                [current_wyckoffs_embed_idx, next_wyckoff_embed_idx.unsqueeze(0)], dim=1
            )
            current_coords = torch.cat(
                [current_coords, sampled_coord_frac.unsqueeze(1)], dim=1
            )  # Use sampled frac coords

        return {
            "predicted_sg": predicted_sg_num.item(),
            "sampled_lattice": sampled_or_mean_lattice.squeeze(0).cpu().numpy(),
            "generated_types": generated_types_raw,
            "generated_wyckoffs": generated_wyckoffs_raw,
            "generated_coords": generated_coords,  # fractional [0, 1)
        }