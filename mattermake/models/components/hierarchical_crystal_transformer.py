from dataclasses import dataclass
from typing import Optional, Dict, Any
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class HierarchicalCrystalTransformerConfig:
    """Configuration for the Hierarchical Crystal Transformer model"""

    vocab_size: int = 2000  # set based on tokenizer
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_probs_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    type_vocab_size: int = 7  # Number of different segment types
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-12

    space_group_embedding_dim: int = 64
    element_embedding_dim: int = 64
    wyckoff_embedding_dim: int = 32
    lattice_embedding_dim: int = 32
    coordinate_embedding_dim: int = 32

    composition_layers: int = 3
    space_group_layers: int = 2
    lattice_layers: int = 3
    atom_layers: int = 6
    integration_layers: int = 2

    use_cross_attention: bool = True
    cross_attention_heads: int = 8

    use_curriculum: bool = False
    composition_curriculum_epochs: int = 5
    space_group_curriculum_epochs: int = 5
    lattice_curriculum_epochs: int = 5

    composition_loss_weight: float = 1.2
    space_group_loss_weight: float = 1.0
    lattice_loss_weight: float = 1.0
    atom_loss_weight: float = 0.8

    # Mode for predictions: "discrete" or "continuous"
    prediction_mode: str = "discrete"

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

        if self.head_size * self.num_heads != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads, got {self.hidden_size} and {self.num_heads}"
            )

        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size)

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

        if key_value_states is not None:
            k = self.k_proj(key_value_states)
            v = self.v_proj(key_value_states)
            kv_seq_len = key_value_states.shape[1]
        else:
            k = self.k_proj(hidden_states)
            v = self.v_proj(hidden_states)
            kv_seq_len = seq_len

        q = self.q_proj(hidden_states)

        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)
        k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )
        v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(
            1, 2
        )

        attn_scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(self.head_size)

        if causal_mask:
            causal_mask_tensor = torch.triu(
                torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                diagonal=1,
            )
            attn_scores = attn_scores.masked_fill(causal_mask_tensor, float("-inf"))

        if attention_mask is not None:
            scores_shape = (
                attn_scores.shape
            )  # [batch_size, num_heads, seq_len, kv_seq_len]

            if attention_mask.dim() == 2:
                batch_mask = attention_mask.bool().float()  # [batch_size, seq_len]

                new_mask = torch.ones(
                    (batch_size, scores_shape[1], scores_shape[2], scores_shape[3]),
                    dtype=torch.bool,
                    device=attention_mask.device,
                )

                for b in range(batch_size):
                    for i in range(min(batch_mask.shape[1], scores_shape[2])):
                        if batch_mask[b, i] == 0:
                            new_mask[b, :, i, :] = False

                attn_scores = attn_scores.masked_fill(~new_mask, float("-inf"))
            else:
                bool_mask = torch.zeros_like(attn_scores, dtype=torch.bool)

                if (
                    attention_mask.dim() >= 3
                    and bool_mask.dim() == attention_mask.dim()
                ):
                    for i in range(attention_mask.dim()):
                        if (
                            attention_mask.shape[i] == bool_mask.shape[i]
                            or attention_mask.shape[i] == 1
                        ):
                            continue
                        else:
                            bool_mask = bool_mask.fill_(True)
                            break
                    else:
                        bool_mask = attention_mask == 0

                attn_scores = attn_scores.masked_fill(bool_mask, float("-inf"))

        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        context = torch.matmul(attn_weights, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch_size, seq_len, self.hidden_size)
        )

        output = self.out_proj(context)

        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward networks"""

    def __init__(self, config):
        super().__init__()
        self.attention = MultiHeadAttention(config)

        self.ff_net = nn.Sequential(
            nn.Linear(config.hidden_size, config.intermediate_size),
            nn.GELU(),
            nn.Linear(config.intermediate_size, config.hidden_size),
            nn.Dropout(config.hidden_dropout_prob),
        )

        self.norm1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.norm2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_mask: bool = False,
        key_value_states: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        attn_output = self.attention(
            self.norm1(hidden_states),
            attention_mask=attention_mask,
            causal_mask=causal_mask,
            key_value_states=key_value_states,
        )
        hidden_states = hidden_states + self.dropout(attn_output)

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
        context_output = self.cross_attention(
            self.norm(hidden_states),
            key_value_states=context_states,
            attention_mask=attention_mask,
        )
        return hidden_states + self.dropout(context_output)


def bound_lattice_lengths(raw_lengths):
    """Convert raw network outputs to realistic lattice length parameters (a, b, c)

    Args:
        raw_lengths: Raw values from the network

    Returns:
        Bounded lattice length values between 2 and 50 Å
    """
    return 2.0 + 48.0 * torch.sigmoid(raw_lengths)


def bound_lattice_angles(raw_angles):
    """Convert raw network outputs to realistic lattice angle parameters (alpha, beta, gamma)

    Args:
        raw_angles: Raw values from the network

    Returns:
        Bounded lattice angle values between 30 and 150 degrees
    """
    return 30.0 + 120.0 * torch.sigmoid(raw_angles)


def bound_fractional_coords(raw_coords):
    """Convert raw network outputs to fractional coordinates (x, y, z)

    Args:
        raw_coords: Raw values from the network

    Returns:
        Bounded fractional coordinates between 0 and 1
    """
    return torch.sigmoid(raw_coords)


class HierarchicalCrystalTransformer(nn.Module):
    """Hierarchical transformer for coarse-to-fine crystal structure generation"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        self.token_embeddings = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=2,  # Assuming padding idx is 2
        )

        self.position_embeddings = nn.Embedding(
            config.max_position_embeddings, config.hidden_size
        )

        self.token_type_embeddings = nn.Embedding(
            config.type_vocab_size, config.hidden_size
        )

        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

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

        if config.use_cross_attention:
            self.sg_from_comp_attention = CrossAttentionLayer(config)
            self.lattice_from_sg_attention = CrossAttentionLayer(config)
            self.atom_from_lattice_attention = CrossAttentionLayer(config)

        self.integration_layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(config.integration_layers)]
        )

        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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

        # Create discrete coordinate head only in discrete prediction mode
        if config.prediction_mode == "discrete":
            self.coordinate_head = nn.Linear(
                config.hidden_size, 10**config.coordinate_embedding_dim
            )  # For fractional coordinates
        else:
            # In continuous mode, don't create the large coordinate head to save memory
            self.coordinate_head = None

        self.lattice_length_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 3),  # 3 lengths: a, b, c
        )

        # Lattice angle parameters (alpha, beta, gamma) typically 30-150 degrees
        self.lattice_angle_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 3),  # 3 angles: alpha, beta, gamma
        )

        # Fractional coordinates (x, y, z) between 0 and 1
        self.fractional_coord_head = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.SiLU(),
            nn.Linear(config.hidden_size // 2, 3),  # 3 coordinates: x, y, z
        )

        self.active_modules = ["composition", "space_group", "lattice", "atoms"]

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

    def set_ground_truth_values(self, lattice_params=None, fractional_coords=None):
        """Set ground truth values for continuous prediction regression loss

        Args:
            lattice_params: Dictionary with 'lengths' and 'angles' tensors for lattice parameters
            fractional_coords: Dictionary with 'fractional_coords' tensor of shape [num_atoms, 3]
        """
        if lattice_params is not None:
            self._lattice_ground_truth = lattice_params

        if fractional_coords is not None:
            self._coordinate_ground_truth = fractional_coords

    def get_position_ids(self, input_ids):
        """Generate position IDs based on input sequence"""
        seq_length = input_ids.size(1)
        position_ids = torch.arange(
            seq_length, dtype=torch.long, device=input_ids.device
        )
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        max_pos = self.config.max_position_embeddings - 1
        position_ids = torch.clamp(position_ids, max=max_pos)
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

        position_ids = self.get_position_ids(input_ids)
        safe_input_ids = torch.clamp(input_ids, max=self.config.vocab_size - 1)
        safe_segment_ids = torch.clamp(segment_ids, max=self.config.type_vocab_size - 1)

        token_embeds = self.token_embeddings(safe_input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.token_type_embeddings(safe_segment_ids)

        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        if attention_mask is None:
            attention_mask = torch.ones(
                (batch_size, seq_length), device=input_ids.device
            )

        comp_mask = safe_segment_ids == self.config.SEGMENT_COMPOSITION
        sg_mask = safe_segment_ids == self.config.SEGMENT_SPACE_GROUP
        lattice_mask = safe_segment_ids == self.config.SEGMENT_LATTICE
        atom_mask = (
            (safe_segment_ids == self.config.SEGMENT_ELEMENT)
            | (safe_segment_ids == self.config.SEGMENT_WYCKOFF)
            | (safe_segment_ids == self.config.SEGMENT_COORDINATE)
        )

        hidden_states = embeddings

        if "composition" in self.active_modules:
            for layer in self.composition_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        if "space_group" in self.active_modules:
            if self.config.use_cross_attention:
                hidden_states = self.sg_from_comp_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                )

            for layer in self.space_group_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        if "lattice" in self.active_modules:
            if self.config.use_cross_attention:
                context_mask = (comp_mask | sg_mask).unsqueeze(1).unsqueeze(2)
                hidden_states = self.lattice_from_sg_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask,
                )

            for layer in self.lattice_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        if "atoms" in self.active_modules:
            if self.config.use_cross_attention:
                context_mask = (
                    (comp_mask | sg_mask | lattice_mask).unsqueeze(1).unsqueeze(2)
                )
                hidden_states = self.atom_from_lattice_attention(
                    hidden_states=hidden_states,
                    context_states=hidden_states,
                    attention_mask=context_mask,
                )

            atom_attention_mask = attention_mask.clone()
            if self.training:
                atom_emphasis = 1.0 + 0.2 * atom_mask.float()
                atom_attention_mask = atom_attention_mask * atom_emphasis.unsqueeze(1)

            for layer in self.atom_encoder:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=atom_attention_mask
                    if self.training
                    else attention_mask,
                    causal_mask=use_causal_mask,
                )

        for layer in self.integration_layers:
            hidden_states = layer(
                hidden_states,
                attention_mask=attention_mask,
                causal_mask=use_causal_mask,
            )

        logits = self.lm_head(hidden_states)

        outputs = {"logits": logits, "hidden_states": hidden_states}

        if labels is not None:
            try:
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous()
                shift_segments = safe_segment_ids[:, 1:].contiguous()

                valid_labels = (shift_labels >= 0) & (
                    shift_labels < self.config.vocab_size
                ) | (shift_labels == -100)
                if not torch.all(valid_labels):
                    shift_labels = torch.where(
                        valid_labels,
                        shift_labels,
                        torch.tensor(-100, device=shift_labels.device),
                    )

                try:
                    loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
                    token_losses = loss_fct(
                        shift_logits.view(-1, self.config.vocab_size),
                        shift_labels.view(-1),
                    )

                    token_losses = token_losses.view(batch_size, -1)

                    weighted_losses = token_losses.clone()

                    comp_indices = (
                        shift_segments == self.config.SEGMENT_COMPOSITION
                    ).float()
                    if self.config.composition_loss_weight != 1.0 and torch.any(
                        comp_indices
                    ):
                        weighted_losses = weighted_losses * (
                            1.0
                            + (self.config.composition_loss_weight - 1.0) * comp_indices
                        )

                    sg_indices = (
                        shift_segments == self.config.SEGMENT_SPACE_GROUP
                    ).float()
                    if self.config.space_group_loss_weight != 1.0 and torch.any(
                        sg_indices
                    ):
                        weighted_losses = weighted_losses * (
                            1.0
                            + (self.config.space_group_loss_weight - 1.0) * sg_indices
                        )

                    lattice_indices = (
                        shift_segments == self.config.SEGMENT_LATTICE
                    ).float()
                    if self.config.lattice_loss_weight != 1.0 and torch.any(
                        lattice_indices
                    ):
                        weighted_losses = weighted_losses * (
                            1.0
                            + (self.config.lattice_loss_weight - 1.0) * lattice_indices
                        )

                    atom_indices = (
                        (shift_segments == self.config.SEGMENT_ELEMENT)
                        | (shift_segments == self.config.SEGMENT_WYCKOFF)
                        | (shift_segments == self.config.SEGMENT_COORDINATE)
                    ).float()
                    if self.config.atom_loss_weight != 1.0 and torch.any(atom_indices):
                        weighted_losses = weighted_losses * (
                            1.0 + (self.config.atom_loss_weight - 1.0) * atom_indices
                        )

                    if weighted_losses.numel() > 0:
                        loss = weighted_losses.sum() / (
                            weighted_losses.size(0) * weighted_losses.size(1)
                        )
                        outputs["loss"] = loss
                    else:
                        outputs["loss"] = torch.tensor(
                            0.0, device=hidden_states.device, requires_grad=True
                        )
                except Exception as e:
                    print(f"Warning: Error in token loss calculation: {e}")
                    outputs["loss"] = torch.tensor(
                        0.0, device=hidden_states.device, requires_grad=True
                    )
            except Exception as e:
                print(f"Warning: Error in preparing token loss calculation: {e}")
                outputs["loss"] = torch.tensor(
                    0.0, device=hidden_states.device, requires_grad=True
                )

            if "space_group" in self.active_modules:
                sg_mask = safe_segment_ids == self.config.SEGMENT_SPACE_GROUP
                if torch.any(sg_mask):
                    sg_hidden = hidden_states[sg_mask]
                    if sg_hidden.size(0) > 0:
                        sg_logits = self.space_group_head(sg_hidden)
                        outputs["space_group_logits"] = sg_logits

                        if labels is not None:
                            sg_labels_mask = (
                                safe_segment_ids == self.config.SEGMENT_SPACE_GROUP
                            )

                            if torch.any(sg_labels_mask):
                                space_group_labels = labels[sg_labels_mask]

                                valid_labels = (space_group_labels >= 0) & (
                                    space_group_labels
                                    < self.space_group_head.out_features
                                ) | (space_group_labels == -100)
                                if not torch.all(valid_labels):
                                    space_group_labels = torch.where(
                                        valid_labels,
                                        space_group_labels,
                                        torch.tensor(
                                            -100, device=space_group_labels.device
                                        ),
                                    )

                                if sg_logits.size(0) == space_group_labels.size(0):
                                    try:
                                        space_group_loss = F.cross_entropy(
                                            sg_logits,
                                            space_group_labels,
                                            ignore_index=-100,
                                        )
                                        outputs["space_group_loss"] = space_group_loss
                                    except Exception as e:
                                        print(
                                            f"Warning: Failed to calculate space group loss: {e}"
                                        )
                                else:
                                    print(
                                        f"Warning: Space group logits and labels size mismatch: {sg_logits.size(0)} vs {space_group_labels.size(0)}"
                                    )

            if "lattice" in self.active_modules:
                lattice_mask = safe_segment_ids == self.config.SEGMENT_LATTICE
                if torch.any(lattice_mask):
                    lattice_hidden = hidden_states[lattice_mask]
                    if lattice_hidden.size(0) > 0:
                        # Discrete approach
                        lattice_logits = self.lattice_param_head(lattice_hidden)
                        outputs["lattice_logits"] = lattice_logits

                        # Continuous approach
                        raw_lattice_lengths = self.lattice_length_head(lattice_hidden)
                        raw_lattice_angles = self.lattice_angle_head(lattice_hidden)

                        lattice_lengths = bound_lattice_lengths(raw_lattice_lengths)
                        lattice_angles = bound_lattice_angles(raw_lattice_angles)

                        outputs["lattice_lengths"] = lattice_lengths
                        outputs["lattice_angles"] = lattice_angles

                        if labels is not None:
                            lat_labels_mask = (
                                safe_segment_ids == self.config.SEGMENT_LATTICE
                            )

                            if torch.any(lat_labels_mask):
                                lattice_labels = labels[lat_labels_mask]

                                valid_labels = (lattice_labels >= 0) & (
                                    lattice_labels
                                    < self.lattice_param_head.out_features
                                ) | (lattice_labels == -100)
                                if not torch.all(valid_labels):
                                    lattice_labels = torch.where(
                                        valid_labels,
                                        lattice_labels,
                                        torch.tensor(
                                            -100, device=lattice_labels.device
                                        ),
                                    )

                                if lattice_logits.size(0) == lattice_labels.size(0):
                                    try:
                                        lattice_loss = F.cross_entropy(
                                            lattice_logits,
                                            lattice_labels,
                                            ignore_index=-100,
                                        )
                                        outputs["lattice_loss"] = (
                                            lattice_loss  # Store discrete loss
                                        )
                                    except Exception as e:
                                        print(
                                            f"Warning: Failed to calculate lattice loss: {e}"
                                        )

                    if (
                        hasattr(self, "_lattice_ground_truth")
                        and self._lattice_ground_truth is not None
                    ):
                        gt_lengths = self._lattice_ground_truth["lengths"]
                        gt_angles = self._lattice_ground_truth["angles"]

                        length_loss = F.mse_loss(lattice_lengths, gt_lengths)
                        angle_loss = F.mse_loss(lattice_angles, gt_angles)

                        lattice_regression_loss = length_loss + angle_loss
                        outputs["lattice_regression_loss"] = lattice_regression_loss

            if "atoms" in self.active_modules:
                element_mask = safe_segment_ids == self.config.SEGMENT_ELEMENT
                wyckoff_mask = safe_segment_ids == self.config.SEGMENT_WYCKOFF
                coordinate_mask = safe_segment_ids == self.config.SEGMENT_COORDINATE

                if torch.any(element_mask):
                    element_hidden = hidden_states[element_mask]
                    if element_hidden.size(0) > 0:
                        element_logits = self.element_head(element_hidden)
                        outputs["element_logits"] = element_logits

                        if labels is not None:
                            element_labels = labels[element_mask]

                            valid_labels = (element_labels >= 0) & (
                                element_labels < self.element_head.out_features
                            ) | (element_labels == -100)
                            if not torch.all(valid_labels):
                                element_labels = torch.where(
                                    valid_labels,
                                    element_labels,
                                    torch.tensor(-100, device=element_labels.device),
                                )

                            if element_logits.size(0) == element_labels.size(0):
                                try:
                                    element_loss = F.cross_entropy(
                                        element_logits,
                                        element_labels,
                                        ignore_index=-100,
                                    )
                                    outputs["element_loss"] = element_loss
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to calculate element loss: {e}"
                                    )
                            else:
                                print(
                                    f"Warning: Element logits and labels size mismatch: {element_logits.size(0)} vs {element_labels.size(0)}"
                                )

                if torch.any(wyckoff_mask):
                    wyckoff_hidden = hidden_states[wyckoff_mask]
                    if wyckoff_hidden.size(0) > 0:
                        wyckoff_logits = self.wyckoff_head(wyckoff_hidden)
                        outputs["wyckoff_logits"] = wyckoff_logits

                        if labels is not None:
                            wyckoff_labels = labels[wyckoff_mask]

                            valid_labels = (wyckoff_labels >= 0) & (
                                wyckoff_labels < self.wyckoff_head.out_features
                            ) | (wyckoff_labels == -100)
                            if not torch.all(valid_labels):
                                wyckoff_labels = torch.where(
                                    valid_labels,
                                    wyckoff_labels,
                                    torch.tensor(-100, device=wyckoff_labels.device),
                                )

                            if wyckoff_logits.size(0) == wyckoff_labels.size(0):
                                try:
                                    wyckoff_loss = F.cross_entropy(
                                        wyckoff_logits,
                                        wyckoff_labels,
                                        ignore_index=-100,
                                    )
                                    outputs["wyckoff_loss"] = wyckoff_loss
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to calculate wyckoff loss: {e}"
                                    )
                            else:
                                print(
                                    f"Warning: Wyckoff logits and labels size mismatch: {wyckoff_logits.size(0)} vs {wyckoff_labels.size(0)}"
                                )

                if torch.any(coordinate_mask):
                    coordinate_hidden = hidden_states[coordinate_mask]
                    if coordinate_hidden.size(0) > 0:
                        outputs["coordinate_hidden"] = coordinate_hidden
                        outputs["avg_coordinate_embedding"] = coordinate_hidden.mean(
                            dim=0
                        )

                        # Group coordinates by atoms (each atom has 3 coordinates: x, y, z)
                        # This is a simplification - in a real implementation, we need
                        # to properly track which coordinates belong to which atoms :: TODO
                        batch_size = coordinate_hidden.size(0)

                        # Handle coordinate predictions based on prediction mode
                        if self.config.prediction_mode == "discrete" and hasattr(self, "coordinate_head") and self.coordinate_head is not None:
                            try:
                                coordinate_logits = self.coordinate_head(
                                    coordinate_hidden
                                )
                                outputs["coordinate_logits"] = coordinate_logits

                                if labels is not None:
                                    coordinate_labels = labels[coordinate_mask]

                                    valid_labels = (coordinate_labels >= 0) & (
                                        coordinate_labels
                                        < self.coordinate_head.out_features
                                    ) | (coordinate_labels == -100)
                                    if not torch.all(valid_labels):
                                        coordinate_labels = torch.where(
                                            valid_labels,
                                            coordinate_labels,
                                            torch.tensor(
                                                -100, device=coordinate_labels.device
                                            ),
                                        )

                                    if coordinate_logits.size(
                                        0
                                    ) == coordinate_labels.size(0):
                                        try:
                                            coordinate_loss = F.cross_entropy(
                                                coordinate_logits,
                                                coordinate_labels,
                                                ignore_index=-100,
                                            )
                                            outputs["coordinate_loss"] = coordinate_loss
                                        except Exception as e:
                                            print(
                                                f"Warning: Failed to calculate coordinate loss: {e}"
                                            )
                                    else:
                                        print(
                                            f"Warning: Coordinate logits and labels size mismatch: {coordinate_logits.size(0)} vs {coordinate_labels.size(0)}"
                                        )
                            except Exception as e:
                                print(
                                    f"Warning: Failed to compute coordinate logits: {e}"
                                )

                    try:
                        num_atoms = coordinate_hidden.size(0) // 3
                        if num_atoms > 0 and coordinate_hidden.size(0) >= 3:
                            raw_coords = self.fractional_coord_head(
                                coordinate_hidden[: num_atoms * 3 : 3]
                            )

                            fractional_coords = bound_fractional_coords(raw_coords)
                            outputs["fractional_coords"] = fractional_coords

                            if (
                                hasattr(self, "_coordinate_ground_truth")
                                and self._coordinate_ground_truth is not None
                            ):
                                try:
                                    gt_coords = self._coordinate_ground_truth[
                                        "fractional_coords"
                                    ]

                                    if fractional_coords.size() == gt_coords.size():
                                        coord_regression_loss = F.mse_loss(
                                            fractional_coords, gt_coords
                                        )
                                        outputs["coord_regression_loss"] = (
                                            coord_regression_loss
                                        )
                                    else:
                                        print(
                                            f"Warning: Coordinate shape mismatch: {fractional_coords.size()} vs {gt_coords.size()}"
                                        )

                                except Exception as e:
                                    print(
                                        f"Warning: Failed to calculate coordinate regression loss: {e}"
                                    )
                    except Exception as e:
                        print(
                            f"Warning: Error in continuous coordinate prediction: {e}"
                        )

                atom_mask = (
                    (safe_segment_ids == self.config.SEGMENT_ELEMENT)
                    | (safe_segment_ids == self.config.SEGMENT_WYCKOFF)
                    | (safe_segment_ids == self.config.SEGMENT_COORDINATE)
                )
                if torch.any(atom_mask):
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
        verbose: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using autoregressive generation.

        Args:
            input_ids: Input token ids
            segment_ids: Segment ids for the input tokens
            constraints: Dictionary of constraints for generation
            max_length: Maximum sequence length to generate
            temperature: Sampling temperature
            top_k: If specified, only sample from the top k most likely tokens
            top_p: If specified, sample from tokens with cumulative probability >= top_p
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: Token signifying end of sequence
            pad_token_id: Token used for padding
            verbose: Whether to print verbose output during generation
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.clone()
        generated_segments = segment_ids.clone()

        if verbose:
            print(
                f"Starting generation with temperature={temperature}, top_k={top_k}, top_p={top_p}"
            )
            print(f"Using prediction mode: {self.config.prediction_mode}")
            print(f"Initial tokens shape: {input_ids.shape}")
            idx_to_token = (
                constraints.get("token_id_maps", {}).get("idx_to_token", {})
                if constraints
                else {}
            )

        continuous_predictions = (
            {"lattice_lengths": [], "lattice_angles": [], "fractional_coords": []}
            if self.config.prediction_mode == "continuous"
            else None
        )

        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Initialize constraint handler (if provided)
        # constraint_handler = (
        #     CrystalConstraintHandler(constraints) if constraints else None
        # )
        constraint_handler = None
        while True:
            outputs = self.forward(
                input_ids=generated_ids,
                segment_ids=generated_segments,
                attention_mask=attention_mask,
                use_causal_mask=True,
            )

            if self.config.prediction_mode == "continuous":
                if "lattice_lengths" in outputs and "lattice_angles" in outputs:
                    continuous_predictions["lattice_lengths"].append(
                        outputs["lattice_lengths"]
                    )
                    continuous_predictions["lattice_angles"].append(
                        outputs["lattice_angles"]
                    )

                if "fractional_coords" in outputs:
                    continuous_predictions["fractional_coords"].append(
                        outputs["fractional_coords"]
                    )

            next_token_logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty

            if constraint_handler is not None:
                current_pos = generated_ids.shape[1] - 1
                token_type = generated_segments[:, -1].cpu().numpy()

                for i in range(batch_size):
                    if unfinished_sequences[i]:
                        mask = constraint_handler.get_mask(
                            generated_ids[i], token_type[i], current_pos
                        )
                        if mask is not None:
                            next_token_logits[i, ~mask] = float("-inf")

            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float("-inf")

            next_token_logits = torch.nan_to_num(
                next_token_logits, nan=-1e9, posinf=1e9, neginf=-1e9
            )

            probs = F.softmax(next_token_logits, dim=-1)

            invalid_probs = torch.isnan(probs) | torch.isinf(probs) | (probs < 0)
            if invalid_probs.any():
                probs = probs.clone()
                probs[invalid_probs] = 0.0
                row_sums = probs.sum(dim=-1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                probs = probs / row_sums

            probs = probs + 1e-10
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                ~unfinished_sequences
            )

            if verbose and (
                generated_ids.shape[1] % 10 == 0 or generated_ids.shape[1] < 10
            ):
                # Log every 10 tokens or the first few tokens
                for i in range(
                    min(batch_size, 2)
                ):  # Only show first 2 sequences to avoid clutter
                    token_id = next_tokens[i].item()
                    token_name = idx_to_token.get(str(token_id), f"<{token_id}>")
                    current_seg = segment_ids[i, -1].item()
                    segment_names = [
                        "SPECIAL",
                        "COMPOSITION",
                        "SPACE_GROUP",
                        "LATTICE",
                        "ELEMENT",
                        "WYCKOFF",
                        "COORDINATE",
                    ]
                    seg_name = (
                        segment_names[current_seg]
                        if current_seg < len(segment_names)
                        else f"SEG_{current_seg}"
                    )
                    print(
                        f"Seq {i}, Pos {generated_ids.shape[1]}: Generated token {token_id} ({token_name}) - Current segment: {seg_name}"
                    )

            next_segments = self._predict_next_segment_id(
                generated_ids, generated_segments, next_tokens
            )

            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )
            generated_segments = torch.cat([generated_segments, next_segments], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (
                    next_tokens != eos_token_id
                )

            if unfinished_sequences.sum() == 0 or generated_ids.shape[1] >= max_length:
                break

        result = {"sequences": generated_ids, "segment_ids": generated_segments}

        # Track if we got continuous predictions
        has_continuous_lattice = False
        has_continuous_coords = False

        if self.config.prediction_mode == "continuous" and continuous_predictions:
            if (
                continuous_predictions["lattice_lengths"]
                and len(continuous_predictions["lattice_lengths"]) > 0
            ):
                # Combine all predictions - for generation we want the last one for each sequence
                try:
                    result["continuous_lattice_lengths"] = torch.cat(
                        continuous_predictions["lattice_lengths"], dim=0
                    )
                    result["continuous_lattice_angles"] = torch.cat(
                        continuous_predictions["lattice_angles"], dim=0
                    )
                    has_continuous_lattice = True

                    if verbose:
                        print(
                            f"Successfully captured continuous lattice predictions with shape: {result['continuous_lattice_lengths'].shape}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous lattice predictions: {e}"
                    )

            if (
                continuous_predictions["fractional_coords"]
                and len(continuous_predictions["fractional_coords"]) > 0
            ):
                try:
                    result["continuous_fractional_coords"] = torch.cat(
                        continuous_predictions["fractional_coords"], dim=0
                    )
                    has_continuous_coords = True

                    if verbose:
                        print(
                            f"Successfully captured continuous coordinate predictions with shape: {result['continuous_fractional_coords'].shape}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous coordinate predictions: {e}"
                    )

        # Add flags to indicate whether continuous predictions were successfully obtained
        result["has_continuous_lattice"] = has_continuous_lattice
        result["has_continuous_coords"] = has_continuous_coords

        if verbose:
            # Print segment type distribution
            segment_counts = {}
            for i in range(generated_segments.size(1)):
                seg_id = generated_segments[0, i].item()  # Look at first sequence
                segment_counts[seg_id] = segment_counts.get(seg_id, 0) + 1

            segment_names = [
                "SPECIAL",
                "COMPOSITION",
                "SPACE_GROUP",
                "LATTICE",
                "ELEMENT",
                "WYCKOFF",
                "COORDINATE",
            ]
            print("\nGeneration summary:")
            print(f"Final sequence length: {generated_ids.shape[1]}")
            print("Segment distribution:")
            for seg_id, count in segment_counts.items():
                seg_name = (
                    segment_names[seg_id]
                    if seg_id < len(segment_names)
                    else f"SEG_{seg_id}"
                )
                print(f"  {seg_name}: {count} tokens")
            print(f"Used continuous lattice predictions: {has_continuous_lattice}")
            print(f"Used continuous coordinate predictions: {has_continuous_coords}")

        return result

    def _predict_next_segment_id(self, sequence_ids, segment_ids, next_tokens):
        """Predict the segment ID for the next token based on context"""
        batch_size = sequence_ids.shape[0]

        current_segment = segment_ids[:, -1]

        next_segments = current_segment.clone()

        for i in range(batch_size):
            token_id = next_tokens[i].item()
            curr_seg = current_segment[i].item()

            seq = sequence_ids[i]
            segs = segment_ids[i]

            if curr_seg == self.config.SEGMENT_COMPOSITION:
                if token_id == 3:  # COMP_SEP_TOKEN
                    next_segments[i] = self.config.SEGMENT_SPECIAL

            elif curr_seg == self.config.SEGMENT_SPECIAL:
                comp_sep_positions = (seq == 3).nonzero().squeeze(-1)
                if (
                    len(comp_sep_positions) > 0
                    and segs[-1] == self.config.SEGMENT_SPECIAL
                ):
                    next_segments[i] = self.config.SEGMENT_SPACE_GROUP

            elif curr_seg == self.config.SEGMENT_SPACE_GROUP:
                next_segments[i] = self.config.SEGMENT_LATTICE

            elif curr_seg == self.config.SEGMENT_LATTICE:
                lattice_positions = (segs == self.config.SEGMENT_LATTICE).sum().item()
                if (
                    lattice_positions >= 6
                ):  # 6 lattice parameters (a, b, c, alpha, beta, gamma)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

            elif curr_seg == self.config.SEGMENT_ELEMENT:
                next_segments[i] = self.config.SEGMENT_WYCKOFF

            elif curr_seg == self.config.SEGMENT_WYCKOFF:
                next_segments[i] = self.config.SEGMENT_COORDINATE

            elif curr_seg == self.config.SEGMENT_COORDINATE:
                coord_count = 0
                for j in range(len(segs) - 1, -1, -1):
                    if segs[j] == self.config.SEGMENT_COORDINATE:
                        coord_count += 1
                    else:
                        break

                if coord_count % 3 == 0:  # After 3 coordinates (x, y, z)
                    next_segments[i] = self.config.SEGMENT_ELEMENT

        return next_segments.unsqueeze(-1)
