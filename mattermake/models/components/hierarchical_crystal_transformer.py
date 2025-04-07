from dataclasses import dataclass
from typing import Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalCrystalTransformerConfig:
    """Configuration for the Hierarchical Crystal Transformer model"""

    vocab_size: int = 2000  # Expected to be set based on tokenizer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    type_vocab_size: int = 7  # Number of different segment types
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    # Crystal-specific parameters
    space_group_embedding_dim: int = 64
    element_embedding_dim: int = 64
    wyckoff_embedding_dim: int = 32
    lattice_embedding_dim: int = 32
    coordinate_embedding_dim: int = 32

    # Hierarchical architecture parameters
    composition_layers: int = 3
    space_group_layers: int = 2
    lattice_layers: int = 3
    atom_layers: int = 6
    integration_layers: int = 2

    # Cross-level attention parameters
    use_cross_attention: bool = True
    cross_attention_heads: int = 8

    # Curriculum learning parameters
    use_curriculum: bool = False
    composition_curriculum_epochs: int = 5
    space_group_curriculum_epochs: int = 5
    lattice_curriculum_epochs: int = 5

    # Specialized loss weighting
    composition_loss_weight: float = 1.2
    space_group_loss_weight: float = 1.0
    lattice_loss_weight: float = 1.0
    atom_loss_weight: float = 0.8

    # Segment IDs for reference
    SEGMENT_SPECIAL: int = 0
    SEGMENT_COMPOSITION: int = 1
    SEGMENT_SPACE_GROUP: int = 2
    SEGMENT_LATTICE: int = 3
    SEGMENT_ELEMENT: int = 4
    SEGMENT_WYCKOFF: int = 5
    SEGMENT_COORDINATE: int = 6


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with causal masking option"""

    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_size = config.hidden_size // config.num_attention_heads
        self.dropout = config.attention_probs_dropout_prob

        # Check if hidden size is divisible by num_heads
        if self.head_size * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {self.hidden_size} and {self.num_heads}"
            )

        # Query, Key, Value projections
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

        # Output projection
        self.out_proj = nn.Linear(self.hidden_size, self.hidden_size)

        self.dropout_layer = nn.Dropout(self.dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # If key_value_states is provided, use those for keys and values (cross-attention)
        if key_value_states is not None:
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            kv_seq_len = key_value_states.shape[1]
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            kv_seq_len = seq_len

        # Project queries
        q = self.q_proj(hidden_states)

        # Reshape to (batch_size, num_heads, seq_len, head_size)
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )

        # Scaled dot-product attention
        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        # Apply causal mask if requested
        if causal_mask:
            causal_mask_tensor = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask_tensor, float("-inf"))

        # Apply attention mask if provided
        if attention_mask is not None:
            # Adjust mask dimensions to match attention scores
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(attention_mask == 0, float("-inf"))

        # Softmax and dropout
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Calculate context vectors
        context = torch.matmul(attn_weights, v)

        # Reshape back
        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        # Output projection
        output = self.out_proj(context)

        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward networks"""

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)

        # Feed-forward network
        self.ff_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        # Dropout
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Self-attention with residual connection
        attn_output = self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            key_value_states=key_value_states,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

        # Feed-forward with residual connection
        ff_output = self.ff_net(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output

        return hidden_states


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for connecting different levels in the hierarchy"""

    def __init__(self, config):
        super().__init__()
        self.cross_attention = MultiHeadAttention(config)
        self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        # Apply cross-attention with residual connection
        context_output = self.cross_attention(
            self.norm(hidden_states),
            key_value_states=context_states,
            attention_mask=attention_mask,
        )
        return hidden_states + self.dropout(context_output)


class HierarchicalCrystalTransformer(nn.Module):
    """Hierarchical transformer for coarse-to-fine crystal structure generation"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token embeddings for all token types
        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=2,  # Assuming padding idx is 2
        )

        # Position embeddings
        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        # Segment/Token type embeddings
        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        # Layer normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Level-specific layers
        self.composition_encoder = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.composition_layers)]
        )
        self.space_group_encoder = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.space_group_layers)]
        )
        self.lattice_encoder = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.lattice_layers)]
        )
        self.atom_encoder = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.atom_layers)]
        )

        # Cross-level attention mechanisms (if enabled)
        if config.use_cross_attention:
            self.sg_from_comp_attention = CrossAttentionLayer(config)
            self.lattice_from_sg_attention = CrossAttentionLayer(config)
            self.atom_from_lattice_attention = CrossAttentionLayer(config)

        # Integration layers (process all tokens together at the end)
        self.integration_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.integration_layers)]
        )

        # Output head
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Specialized prediction heads for different token types
        self.space_group_head = nn.Linear(config.hidden_size, 230)  # 230 space groups
        self.lattice_param_head = nn.Linear(
            config.hidden_size, 6
        )  # 6 lattice parameters
        self.wyckoff_head = nn.Linear(
            config.hidden_size, 26
        )  # 26 Wyckoff letters (a-z)
        self.element_head = nn.Linear(
            config.hidden_size, 95
        )  # ~95 elements commonly found in crystals

        # Keep track of which modules are active (for curriculum learning)
        self.active_modules = ["composition", "space_group", "lattice", "atoms"]

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize the weights"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)

    def set_active_modules(self, module_list):
        """Set which modules are active (for curriculum learning)"""
        self.active_modules = module_list

    def get_position_ids(self, input_ids):
        """Generate position IDs based on input sequence"""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        return position_ids

    def forward(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_length = input_ids.size()

        # Create position IDs
        position_ids = self.get_position_ids(input_ids)

        # Get embeddings for tokens, positions, and segment types
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.token_type_embeddings(segment_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        # Create level masks based on segment IDs
        comp_mask = segment_ids == self.config.SEGMENT_COMPOSITION
        sg_mask = segment_ids == self.config.SEGMENT_SPACE_GROUP
        lattice_mask = segment_ids == self.config.SEGMENT_LATTICE
        atom_mask = (
            (segment_ids == self.config.SEGMENT_ELEMENT)
            | (segment_ids == self.config.SEGMENT_WYCKOFF)
            | (segment_ids == self.config.SEGMENT_COORDINATE)
        )

        # Initialize hidden states
        hidden_states = embeddings

        # Process composition tokens (always active)
        if "composition" in self.active_modules:
            # Process only composition tokens through composition-specific layers
            for layer in self.composition_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        # Process space group tokens with composition context
        if "space_group" in self.active_modules:
            # Use cross-attention to condition on composition
            if self.config.use_cross_attention:
                # Process space group tokens with composition context
                hidden_states = self.sg_from_comp_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                )

            # Process through space group layers
            for layer in self.space_group_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        # Process lattice tokens with space group and composition context
        if "lattice" in self.active_modules:
            # Use cross-attention to condition on prior tokens
            if self.config.use_cross_attention:
                # Get context from composition and space group tokens
                context_mask = (comp_mask | sg_mask).unsqueeze(1).unsqueeze(2)
                hidden_states = self.lattice_from_sg_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask,
                )

            # Process through lattice layers
            for layer in self.lattice_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        # Process atom tokens with all prior context
        if "atoms" in self.active_modules:
            # Use cross-attention to condition on all prior tokens
            if self.config.use_cross_attention:
                # Get context from composition, space group, and lattice tokens
                context_mask = (
                    (comp_mask | sg_mask | lattice_mask).unsqueeze(1).unsqueeze(2)
                )
                hidden_states = self.atom_from_lattice_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask,
                )

            # Create atom-specific attention mask
            atom_attention_mask = attention_mask.clone()
            if self.training:
                # Create a mask that slightly emphasizes atom tokens during attention
                atom_emphasis = 1.0 + 0.2 * atom_mask.float()
                atom_attention_mask = atom_attention_mask * atom_emphasis.unsqueeze(1)

            # Process through atom-specific layers
            for layer in self.atom_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=atom_attention_mask if self.training else attention_mask,
                    causal_mask=use_causal_mask,
                )

        # Final integration layers to process all tokens together
        for layer in self.integration_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=use_causal_mask,
            )

        # Calculate logits for next token prediction
        logits = self.lm_head(hidden_states)

        # Prepare output dictionary
        outputs = {"logits": logits, "hidden_states": hidden_states}

        # Calculate loss if labels are provided
        if labels is not None:
            # Apply specialized loss weighting based on token types
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_segments = segment_ids[:, 1:].contiguous()

            # Standard cross entropy loss
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1)
            )

            # Reshape token losses to batch_size x seq_length
            token_losses = token_losses.view(batch_size, -1)

            # Apply weights based on token types
            weighted_losses = token_losses.clone()

            # Apply composition loss weight
            comp_indices = (shift_segments == self.config.SEGMENT_COMPOSITION).float()
            weighted_losses = weighted_losses * (
                1.0 + (self.config.composition_loss_weight - 1.0) * comp_indices
            )

            # Apply space group loss weight
            sg_indices = (shift_segments == self.config.SEGMENT_SPACE_GROUP).float()
            weighted_losses = weighted_losses * (
                1.0 + (self.config.space_group_loss_weight - 1.0) * sg_indices
            )

            # Apply lattice loss weight
            lattice_indices = (shift_segments == self.config.SEGMENT_LATTICE).float()
            weighted_losses = weighted_losses * (
                1.0 + (self.config.lattice_loss_weight - 1.0) * lattice_indices
            )

            # Apply atom loss weight
            atom_indices = (
                (shift_segments == self.config.SEGMENT_ELEMENT)
                | (shift_segments == self.config.SEGMENT_WYCKOFF)
                | (shift_segments == self.config.SEGMENT_COORDINATE)
            ).float()
            weighted_losses = weighted_losses * (
                1.0 + (self.config.atom_loss_weight - 1.0) * atom_indices
            )

            # Calculate mean loss
            loss = weighted_losses.sum() / (
                weighted_losses.size(0) * weighted_losses.size(1)
            )
            outputs["loss"] = loss

            # Calculate specialized losses if needed
            if "space_group" in self.active_modules:
                sg_hidden = hidden_states[sg_mask]
                if sg_hidden.size(0) > 0:
                    sg_logits = self.space_group_head(sg_hidden)
                    outputs["space_group_logits"] = sg_logits
                    # We would need space group labels to compute this loss
                    # outputs["space_group_loss"] = ...

            if "lattice" in self.active_modules:
                lattice_hidden = hidden_states[lattice_mask]
                if lattice_hidden.size(0) > 0:
                    lattice_logits = self.lattice_param_head(lattice_hidden)
                    outputs["lattice_logits"] = lattice_logits
                    # We would need lattice parameter labels to compute this loss
                    # outputs["lattice_loss"] = ...
                    
            if "atoms" in self.active_modules:
                # Extract atom-related hidden states using the component masks
                element_mask = segment_ids == self.config.SEGMENT_ELEMENT
                wyckoff_mask = segment_ids == self.config.SEGMENT_WYCKOFF
                coordinate_mask = segment_ids == self.config.SEGMENT_COORDINATE
                
                # Process element predictions
                if element_mask.sum() > 0:
                    element_hidden = hidden_states[element_mask]
                    element_logits = self.element_head(element_hidden)
                    outputs["element_logits"] = element_logits
                
                # Process Wyckoff position predictions
                if wyckoff_mask.sum() > 0:
                    wyckoff_hidden = hidden_states[wyckoff_mask]
                    wyckoff_logits = self.wyckoff_head(wyckoff_hidden)
                    outputs["wyckoff_logits"] = wyckoff_logits
                
                # Process coordinate information (for enhanced positional understanding)
                if coordinate_mask.sum() > 0:
                    coordinate_hidden = hidden_states[coordinate_mask]
                    outputs["coordinate_hidden"] = coordinate_hidden
                    
                    # Calculate average coordinate embedding for structure analysis
                    outputs["avg_coordinate_embedding"] = coordinate_hidden.mean(dim=0)
                
                # Store atom-specific hidden states for downstream tasks
                if atom_mask.sum() > 0:
                    outputs["atom_hidden_states"] = hidden_states[atom_mask]

        return outputs

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        constraints: Dict[str, Any] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 2,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using autoregressive generation.
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        # Initialize attention mask
        attention_mask = torch.ones_like(input_ids)

        # Initialize storage for generated sequence
        generated_ids = input_ids.clone()
        generated_segments = segment_ids.clone()

        # Track if each sequence is finished
        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Initialize constraint handler (if provided)
        # constraint_handler = (
        #     CrystalConstraintHandler(constraints) if constraints else None
        # )
        constraint_handler = None
        # Generate tokens one by one
        while True:
            # Forward pass to get next token logits
            outputs = self.forward(
                input_ids=generated_ids,
                segment_ids=generated_segments,
                attention_mask=attention_mask,
                use_causal_mask=True,
            )

            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Apply constraints if available
            if constraint_handler is not None:
                # Get current position and segment type
                current_pos = generated_ids.shape[1] - 1
                token_type = generated_segments[:, -1].cpu().numpy()

                # Apply constraint masks to logits
                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        mask = constraint_handler.get_mask(
                            generated_ids[i], token_type[i], current_pos
                        )
                        if mask is not None:
                            # Apply mask (set invalid tokens to -inf)
                            next_token_logits[i, ~mask] = float("-inf")

            # Apply top-k filtering
            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                # Apply indices changes back to original logits tensor
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float("-inf")

            # Sample next token
            probs = F.softmax(next_token_logits, dim=-1)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Set to pad_token_id if sequence is finished
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                ~unfinished_sequences
            )

            # Determine segment ID for the next token
            # This requires knowledge of the token sequence and current context
            # A more realistic implementation would determine this based on the token type
            next_segments = self._predict_next_segment_id(
                generated_ids, generated_segments, next_tokens
            )

            # Add to generated sequence
            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )
            generated_segments = torch.cat([generated_segments, next_segments], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )

            # Update finished sequences
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (
                    next_tokens != eos_token_id
                )

            # Stop if all sequences are finished or max length is reached
            if unfinished_sequences.sum() == 0 or generated_ids.shape[1] >= max_length:
                break

        return {"sequences": generated_ids, "segment_ids": generated_segments}

    def _predict_next_segment_id(self, sequence_ids, segment_ids, next_tokens):
        """Predict the segment ID for the next token based on context"""
        batch_size = sequence_ids.shape[0]

        # Get the current segment ID (from the last position)
        current_segment = segment_ids[:, -1]

        # Initialize next segment IDs (default: keep the same segment)
        next_segments = current_segment.clone()

        for i in range(batch_size):
            # Get token ID and current segment
            token_id = next_tokens[i].item()
            curr_seg = current_segment[i].item()

            # Extract the current sequence
            seq = sequence_ids[i]
            segs = segment_ids[i]

            # Determine next segment based on current sequence state and token
            if curr_seg == self.config.SEGMENT_COMPOSITION:
                # If we see a COMP_SEP token after composition, switch to space group
                if token_id == 3:  # COMP_SEP_TOKEN
                    next_segments[i] = self.config.SEGMENT_SPECIAL
                # Stay in composition segment otherwise

            elif curr_seg == self.config.SEGMENT_SPECIAL:
                # After COMP_SEP token, we expect space group tokens
                comp_sep_positions = (seq == 3).nonzero().squeeze(-1)
                if (
                    len(comp_sep_positions) > 0
                    and segs[-1] == self.config.SEGMENT_SPECIAL
                ):
                    next_segments[i] = self.config.SEGMENT_SPACE_GROUP

            elif curr_seg == self.config.SEGMENT_SPACE_GROUP:
                # After space group, we expect lattice parameters
                next_segments[i] = self.config.SEGMENT_LATTICE

            elif curr_seg == self.config.SEGMENT_LATTICE:
                # After 6 lattice parameters, switch to elements
                lattice_positions = (segs == self.config.SEGMENT_LATTICE).sum().item()
                if (
                    lattice_positions >= 6
                ):  # 6 lattice parameters (a, b, c, alpha, beta, gamma)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

            elif curr_seg == self.config.SEGMENT_ELEMENT:
                # After element, expect Wyckoff position
                next_segments[i] = self.config.SEGMENT_WYCKOFF

            elif curr_seg == self.config.SEGMENT_WYCKOFF:
                # After Wyckoff, expect coordinates
                next_segments[i] = self.config.SEGMENT_COORDINATE

            elif curr_seg == self.config.SEGMENT_COORDINATE:
                # After 3 coordinates, switch back to element for next atom
                # Count consecutive coordinate tokens
                coord_count = 0
                for j in range(len(segs) - 1, -1, -1):
                    if segs[j] == self.config.SEGMENT_COORDINATE:
                        coord_count += 1
                    else:
                        break

                if coord_count % 3 == 0:  # After 3 coordinates (x, y, z)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

        return next_segments.unsqueeze(-1)
