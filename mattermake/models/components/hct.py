from dataclasses import dataclass
from typing import Optional, Dict, Any, Union, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from mattermake.models.components.hct_constraint_handler import CrystalConstraintHandler
from mattermake.utils.hct_wyckoff_mapping import (
    SpaceGroupWyckoffMapping,
)


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

    # Mixture Density Network parameters
    lattice_mixture_components: int = 5  # Number of components for lattice parameter MoG
    coord_mixture_components: int = 5  # Number of components for coordinate MoVM

    # Whether to apply Wyckoff position constraints
    apply_wyckoff_constraints: bool = False
    
    # Whether to use combined Wyckoff-multiplicity tokens
    use_combined_wyckoff_tokens: bool = True
    
    # Number of combined Wyckoff-multiplicity tokens
    # Default calculation: ~230 space groups * 26 letters = 5980 possible combinations
    num_wyckoff_mult_tokens: int = 6000

    SEGMENT_SPECIAL: int = 0
    SEGMENT_COMPOSITION: int = 1
    SEGMENT_SPACE_GROUP: int = 2
    SEGMENT_LATTICE: int = 3
    SEGMENT_ELEMENT: int = 4
    SEGMENT_WYCKOFF: int = 5
    SEGMENT_COORDINATE: int = 6


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with causal masking option and KV-caching support"""

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
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        batch_size, seq_len, _ = hidden_states.shape

        # Initialize new cache or use existing one
        new_cache = None
        cached_k, cached_v = None, None

        # Check if we have a valid KV cache to use
        if use_kv_cache and kv_cache is not None:
            if "k" in kv_cache and "v" in kv_cache:
                cached_k = kv_cache["k"]
                cached_v = kv_cache["v"]

        # Process query vectors for all tokens
        q = self.q_proj(hidden_states)

        # With KV caching, only process key and value for the new token
        if use_kv_cache and cached_k is not None and cached_v is not None:
            # Only compute k, v for the last token
            if key_value_states is not None:
                new_k = self.k_proj(key_value_states[:, -1:, :])
                new_v = self.v_proj(key_value_states[:, -1:, :])
            else:
                new_k = self.k_proj(hidden_states[:, -1:, :])
                new_v = self.v_proj(hidden_states[:, -1:, :])

            # Reshape new keys and values
            new_k = new_k.view(batch_size, 1, self.num_heads, self.head_size).transpose(
                1, 2
            )
            new_v = new_v.view(batch_size, 1, self.num_heads, self.head_size).transpose(
                1, 2
            )

            # Concatenate with cached keys and values
            k = torch.cat([cached_k, new_k], dim=2)
            v = torch.cat([cached_v, new_v], dim=2)
            kv_seq_len = k.size(2)

            # Create new cache
            new_cache = {"k": k, "v": v}
        else:
            # No caching or first token, compute k, v for all tokens
            if key_value_states is not None:
                k = self.k_proj(key_value_states)
                v = self.v_proj(key_value_states)
                kv_seq_len = key_value_states.shape[1]
            else:
                k = self.k_proj(hidden_states)
                v = self.v_proj(hidden_states)
                kv_seq_len = seq_len

            # Reshape k, v
            k = k.view(
                batch_size, kv_seq_len, self.num_heads, self.head_size
            ).transpose(1, 2)
            v = v.view(
                batch_size, kv_seq_len, self.num_heads, self.head_size
            ).transpose(1, 2)

            # Create new cache
            if use_kv_cache:
                new_cache = {"k": k, "v": v}

        # Reshape query
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

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

        # Return the output along with the updated cache if KV caching is enabled
        if use_kv_cache and new_cache is not None:
            return output, new_cache
        return output


class TransformerLayer(nn.Module):
    """Single transformer layer with self-attention and feed-forward networks with KV-caching support"""

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
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        # Apply attention with optional KV-caching
        normed_hidden_states = self.norm1(hidden_states)

        if use_kv_cache:
            attn_output, new_cache = self.attention(
                normed_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                key_value_states=key_value_states,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
        else:
            attn_output = self.attention(
                normed_hidden_states,
                attention_mask=attention_mask,
                causal_mask=causal_mask,
                key_value_states=key_value_states,
            )

        hidden_states = hidden_states + self.dropout(attn_output)

        # Apply feed-forward network
        ff_output = self.ff_net(self.norm2(hidden_states))
        hidden_states = hidden_states + ff_output

        # Return with cache if KV-caching is enabled
        if use_kv_cache:
            return hidden_states, new_cache
        return hidden_states


class CrossAttentionLayer(nn.Module):
    """Cross-attention layer for connecting different levels in the hierarchy with KV-caching support"""

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
        kv_cache: Optional[Dict[str, torch.Tensor]] = None,
        use_kv_cache: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        normed_hidden_states = self.norm(hidden_states)

        if use_kv_cache:
            context_output, new_cache = self.cross_attention(
                normed_hidden_states,
                key_value_states=context_states,
                attention_mask=attention_mask,
                kv_cache=kv_cache,
                use_kv_cache=True,
            )
            output = hidden_states + self.dropout(context_output)
            return output, new_cache
        else:
            context_output = self.cross_attention(
                normed_hidden_states,
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
        self._sg_wyckoff_mapping = None

        if (
            hasattr(config, "apply_wyckoff_constraints")
            and config.apply_wyckoff_constraints
        ):
            self.sg_wyckoff_mapping = SpaceGroupWyckoffMapping()

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

        # We don't use discrete lattice parameters, so no need for this head
        self.lattice_param_head = None
        self.wyckoff_head = nn.Linear(
            config.hidden_size, 26
        )  # 26 Wyckoff letters (a-z)
        
        # Add combined Wyckoff-multiplicity head if enabled
        if hasattr(config, "use_combined_wyckoff_tokens") and config.use_combined_wyckoff_tokens:
            self.wyckoff_mult_head = nn.Linear(
                config.hidden_size, config.num_wyckoff_mult_tokens
            )  # Combined Wyckoff letter + multiplicity tokens
        else:
            self.wyckoff_mult_head = None
            
        self.element_head = nn.Linear(
            config.hidden_size, 95
        )  # ~95 elements commonly found in crystals

        # No discrete coordinate head - always use continuous prediction
        self.coordinate_head = None

        # Always use mixture density networks for continuous prediction
        from .mixture_density import MixtureOfGaussiansHead, MixtureOfWrappedNormalsHead
        
        # Mixture of Gaussians for lattice parameters
        self.lattice_mog_head = MixtureOfGaussiansHead(
            hidden_size=config.hidden_size,
            output_dim=6,  # 3 lengths + 3 angles
            n_mixtures=config.lattice_mixture_components
        )
        
        # Mixture of wrapped normals for fractional coordinates
        self.fractional_coord_movm_head = MixtureOfWrappedNormalsHead(
            hidden_size=config.hidden_size,
            output_dim=3,  # x, y, z coordinates
            n_mixtures=config.coord_mixture_components
        )
        
        # No standard regression heads - only using mixture density networks
        self.lattice_length_head = None
        self.lattice_angle_head = None
        self.fractional_coord_head = None

        self.active_modules = ["composition", "space_group", "lattice", "atoms"]

        self.apply(self._init_weights)

        @property
        def sg_wyckoff_mapping(self):
            """Lazy loading for SpaceGroupWyckoffMapping"""
            if (
                not hasattr(self.config, "apply_wyckoff_constraints")
                or not self.config.apply_wyckoff_constraints
            ):
                return None

            if self._sg_wyckoff_mapping is None:
                self._sg_wyckoff_mapping = SpaceGroupWyckoffMapping()
            return self._sg_wyckoff_mapping

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
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
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
            for i, layer in enumerate(self.composition_encoder):
                layer_cache = (
                    None if kv_caches is None else kv_caches.get(f"composition_{i}")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        causal_mask=use_causal_mask,
                        kv_cache=layer_cache,
                        use_kv_cache=True,
                    )
                    kv_caches[f"composition_{i}"] = new_cache
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        causal_mask=use_causal_mask,
                    )

        if "space_group" in self.active_modules:
            if self.config.use_cross_attention:
                cross_attn_cache = (
                    None if kv_caches is None else kv_caches.get("sg_from_comp")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = self.sg_from_comp_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    kv_caches["sg_from_comp"] = new_cache
                else:
                    hidden_states = self.sg_from_comp_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=comp_mask.unsqueeze(1).unsqueeze(2),
                    )

            for i, layer in enumerate(self.space_group_encoder):
                layer_cache = (
                    None if kv_caches is None else kv_caches.get(f"space_group_{i}")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        causal_mask=use_causal_mask,
                        kv_cache=layer_cache,
                        use_kv_cache=True,
                    )
                    kv_caches[f"space_group_{i}"] = new_cache
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        causal_mask=use_causal_mask,
                    )

        if "lattice" in self.active_modules:
            if self.config.use_cross_attention:
                context_mask = (comp_mask | sg_mask).unsqueeze(1).unsqueeze(2)
                cross_attn_cache = (
                    None if kv_caches is None else kv_caches.get("lattice_from_sg")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = self.lattice_from_sg_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    kv_caches["lattice_from_sg"] = new_cache
                else:
                    hidden_states = self.lattice_from_sg_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                    )

            for i, layer in enumerate(self.lattice_encoder):
                layer_cache = (
                    None if kv_caches is None else kv_caches.get(f"lattice_{i}")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = layer(
                        hidden_states,
                        attention_mask=attention_mask,
                        causal_mask=use_causal_mask,
                        kv_cache=layer_cache,
                        use_kv_cache=True,
                    )
                    kv_caches[f"lattice_{i}"] = new_cache
                else:
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
                cross_attn_cache = (
                    None if kv_caches is None else kv_caches.get("atom_from_lattice")
                )

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = self.atom_from_lattice_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                        kv_cache=cross_attn_cache,
                        use_kv_cache=True,
                    )
                    kv_caches["atom_from_lattice"] = new_cache
                else:
                    hidden_states = self.atom_from_lattice_attention(
                        hidden_states=hidden_states,
                        context_states=hidden_states,
                        attention_mask=context_mask,
                    )

            atom_attention_mask = attention_mask.clone()
            if self.training:
                atom_emphasis = 1.0 + 0.2 * atom_mask.float()
                atom_attention_mask = atom_attention_mask * atom_emphasis.unsqueeze(1)

            for i, layer in enumerate(self.atom_encoder):
                layer_cache = None if kv_caches is None else kv_caches.get(f"atom_{i}")

                if use_kv_cache and kv_caches is not None:
                    hidden_states, new_cache = layer(
                        hidden_states,
                        attention_mask=atom_attention_mask
                        if self.training
                        else attention_mask,
                        causal_mask=use_causal_mask,
                        kv_cache=layer_cache,
                        use_kv_cache=True,
                    )
                    kv_caches[f"atom_{i}"] = new_cache
                else:
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=atom_attention_mask
                        if self.training
                        else attention_mask,
                        causal_mask=use_causal_mask,
                    )

        for i, layer in enumerate(self.integration_layers):
            layer_cache = (
                None if kv_caches is None else kv_caches.get(f"integration_{i}")
            )

            if use_kv_cache and kv_caches is not None:
                hidden_states, new_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                    kv_cache=layer_cache,
                    use_kv_cache=True,
                )
                kv_caches[f"integration_{i}"] = new_cache
            else:
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

            selected_space_groups = {}
            if "space_group" in self.active_modules:
                sg_mask = safe_segment_ids == self.config.SEGMENT_SPACE_GROUP
                if torch.any(sg_mask):
                    sg_hidden = hidden_states[sg_mask]
                    if sg_hidden.size(0) > 0:
                        sg_logits = self.space_group_head(sg_hidden)
                        outputs["space_group_logits"] = sg_logits

                        if not self.training:
                            # For inference, get the most likely space group
                            batch_indices = torch.where(sg_mask)[0]
                            for i, batch_idx in enumerate(batch_indices):
                                if batch_idx not in selected_space_groups:
                                    # +1 because space groups are 1-230, and our output is 0-229
                                    space_group = sg_logits[i].argmax().item() + 1
                                    # Ensure space group is in valid range
                                    if 1 <= space_group <= 230:
                                        selected_space_groups[batch_idx.item()] = (
                                            space_group
                                        )

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

            if "lattice" in self.active_modules:
                lattice_mask = safe_segment_ids == self.config.SEGMENT_LATTICE
                if torch.any(lattice_mask):
                    lattice_hidden = hidden_states[lattice_mask]
                    if lattice_hidden.size(0) > 0:
                        # Always use Mixture of Gaussians (MoG) for lattice predictions
                        lattice_mog_params = self.lattice_mog_head(lattice_hidden)
                        outputs["lattice_mog_params"] = lattice_mog_params
                        
                        # Extract predicted means for compatibility
                        # These will be used as the "best guess" values when needed
                        weights_probs = F.softmax(lattice_mog_params["weights_logits"], dim=-1)
                        
                        # Calculate weighted means for each parameter
                        # Shape: [batch_size, 6]
                        weighted_means = torch.sum(
                            weights_probs.unsqueeze(-2) * lattice_mog_params["means"],
                            dim=-1
                        )
                        
                        # Split into lengths and angles for compatibility
                        lattice_lengths = bound_lattice_lengths(weighted_means[:, :3])
                        lattice_angles = bound_lattice_angles(weighted_means[:, 3:])

                        outputs["lattice_lengths"] = lattice_lengths
                        outputs["lattice_angles"] = lattice_angles

                        # No discrete loss calculation for lattice parameters

                        # --- LATTICE REGRESSION LOSS ---
                        # Check if we are training AND have ground truth
                        if (
                            self.training
                            and hasattr(self, "_lattice_ground_truth")
                            and self._lattice_ground_truth is not None
                        ):
                            gt_lengths = self._lattice_ground_truth["lengths"]
                            gt_angles = self._lattice_ground_truth["angles"]
                            
                            # Combine lengths and angles for full lattice parameters
                            # Shape: [batch_size, 6]
                            gt_lattice_params = torch.cat([gt_lengths, gt_angles], dim=-1)
                            
                            # Always use negative log-likelihood loss with mixture of Gaussians
                            try:
                                # Get distribution from parameters
                                mog_dist = self.lattice_mog_head.get_distribution(outputs["lattice_mog_params"])
                                
                                # Calculate negative log-likelihood
                                log_prob = mog_dist.log_prob(gt_lattice_params)
                                lattice_regression_loss = -log_prob.mean()
                                outputs["lattice_regression_loss"] = lattice_regression_loss
                            except Exception as e:
                                print(f"Warning: Failed to calculate lattice MoG loss: {e}")

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
                        # Get batch indices for each Wyckoff position
                        batch_indices = torch.where(wyckoff_mask)[0]

                        # Prepare outputs for wyckoff logits
                        all_wyckoff_logits = []

                        for i, batch_idx in enumerate(batch_indices):
                            batch_idx = batch_idx.item()

                            # Determine whether to use combined or regular Wyckoff head
                            use_combined = (
                                hasattr(self.config, "use_combined_wyckoff_tokens")
                                and self.config.use_combined_wyckoff_tokens
                                and self.wyckoff_mult_head is not None
                            )
                            
                            if use_combined:
                                # Use combined Wyckoff-multiplicity head
                                wyckoff_logits_i = self.wyckoff_mult_head(
                                    wyckoff_hidden[i : i + 1]
                                )
                            else:
                                # Original fixed Wyckoff head
                                wyckoff_logits_i = self.wyckoff_head(
                                    wyckoff_hidden[i : i + 1]
                                )

                            # Apply space group constraints if applicable
                            if (
                                hasattr(self.config, "apply_wyckoff_constraints")
                                and self.config.apply_wyckoff_constraints
                                and batch_idx in selected_space_groups
                                and self._sg_wyckoff_mapping is not None
                            ):
                                space_group = selected_space_groups[batch_idx]

                                # Create appropriate mask based on whether we're using combined tokens
                                if use_combined:
                                    # Create a basic mask for the current space group
                                    # Get allowed Wyckoff positions
                                    allowed_wyckoff = self._sg_wyckoff_mapping.get_allowed_wyckoff_positions(space_group)
                                    
                                    # Create mask for combined tokens (only allows tokens for current space group)
                                    valid_mask = torch.zeros(
                                        self.config.num_wyckoff_mult_tokens,
                                        device=wyckoff_hidden.device,
                                        dtype=torch.bool
                                    )
                                    
                                    # Set valid positions based on the tokens for this space group
                                    offset = 3000 + (space_group * 100)  # Combined token range starts at 3000
                                    for letter_idx, letter in enumerate(allowed_wyckoff):
                                        idx = ord(letter) - ord('a')
                                        if 0 <= idx < 26:
                                            token_id = offset + idx
                                            if token_id < valid_mask.size(0):
                                                valid_mask[token_id] = True
                                else:
                                    # Original approach for individual Wyckoff tokens
                                    valid_mask = torch.tensor(
                                        self._sg_wyckoff_mapping.create_wyckoff_mask(
                                            space_group
                                        ),
                                        device=wyckoff_hidden.device,
                                        dtype=torch.bool,
                                    )

                                # Apply the mask by setting invalid positions to -inf
                                if valid_mask is not None and valid_mask.shape[0] <= wyckoff_logits_i.shape[1]:
                                    wyckoff_logits_i[:, ~valid_mask] = float("-inf")

                            all_wyckoff_logits.append(wyckoff_logits_i)

                        if all_wyckoff_logits:
                            wyckoff_logits = torch.cat(all_wyckoff_logits, dim=0)
                            outputs["wyckoff_logits"] = wyckoff_logits

                        if labels is not None:
                            wyckoff_labels = labels[wyckoff_mask]
                            
                            # Determine which head was used (combined or regular)
                            is_using_combined = (
                                hasattr(self.config, "use_combined_wyckoff_tokens")
                                and self.config.use_combined_wyckoff_tokens
                                and self.wyckoff_mult_head is not None
                                and "wyckoff_mult_head" in str(wyckoff_logits.shape)
                            )
                            
                            # Get the appropriate output feature size based on which head was used
                            if is_using_combined and hasattr(self, "wyckoff_mult_head") and self.wyckoff_mult_head is not None:
                                out_features = self.wyckoff_mult_head.out_features
                            else:
                                out_features = self.wyckoff_head.out_features

                            # Validate labels against the correct output size
                            valid_labels = (wyckoff_labels >= 0) & (
                                wyckoff_labels < out_features
                            ) | (wyckoff_labels == -100)
                            
                            if not torch.all(valid_labels):
                                # Handle case where labels might be for the wrong head type
                                if is_using_combined and torch.any(wyckoff_labels >= 3000):
                                    # Labels appear to be for combined tokens, keep them
                                    pass
                                elif not is_using_combined and torch.any(wyckoff_labels < 3000):
                                    # Labels appear to be for regular tokens, keep them
                                    pass
                                else:
                                    # Labels need to be replaced with ignore_index
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
                        batch_size = coordinate_hidden.size(
                            0
                        )  # This is incorrect, this is num_coord_tokens, not batch_size

                        # Coordinate predictions are always continuous

                    # --- COORDINATE REGRESSION LOSS ---
                    try:
                        num_atoms = coordinate_hidden.size(0) // 3
                        # Check num_atoms > 0 and that coordinate_hidden has enough elements
                        if num_atoms > 0 and coordinate_hidden.size(0) >= num_atoms * 3:
                            # Select coordinates corresponding to the start of each atom (x coord)
                            # Assuming coordinates are ordered like [x1,y1,z1, x2,y2,z2, ...] -> WRONG ASSUMPTION?
                            # Let's stick to the original logic but guard it

                            # Use hidden states for the 'x' coordinate token of each atom
                            # The original code `coordinate_hidden[: num_atoms * 3 : 3]` attempts this.
                            # Check if this indexing is correct based on how segments are laid out.
                            # A safer bet might be to average or pool, but let's try guarding first.
                            if (
                                coordinate_hidden.size(0) >= 3
                            ):  # Ensure there's at least one potential atom
                                # Make sure the slice index is valid
                                slice_end = num_atoms * 3
                                if slice_end <= coordinate_hidden.size(0):
                                    # Always use MoVM for coordinate predictions
                                    coord_movm_params = self.fractional_coord_movm_head(
                                        coordinate_hidden[:slice_end:3]  # Get hidden state for each atom
                                    )
                                    outputs["coord_movm_params"] = coord_movm_params
                                    
                                    # Calculate weighted means as "best guess" values for compatibility
                                    weights_probs = F.softmax(coord_movm_params["weights_logits"], dim=-1)
                                    
                                    # Calculate weighted means for each coordinate
                                    # Shape: [num_atoms, 3]
                                    weighted_means = torch.sum(
                                        weights_probs.unsqueeze(-2) * coord_movm_params["means"],
                                        dim=-1
                                    )
                                    
                                    # Bound coordinates to [0, 1)
                                    from .mixture_density import bound_mixture_fractional_coords
                                    fractional_coords = bound_mixture_fractional_coords(weighted_means)
                                        
                                    outputs["fractional_coords"] = fractional_coords

                                    # Check if we are training AND have ground truth
                                    if (
                                        self.training
                                        and hasattr(self, "_coordinate_ground_truth")
                                        and self._coordinate_ground_truth is not None
                                    ):
                                        try:
                                            gt_coords = self._coordinate_ground_truth[
                                                "fractional_coords"
                                            ]
                                            
                                            # Always use negative log-likelihood loss with mixture of wrapped normals
                                            try:
                                                # Get distribution from parameters
                                                movm_dist = self.fractional_coord_movm_head.get_distribution(outputs["coord_movm_params"])
                                                
                                                # Calculate negative log-likelihood
                                                log_prob = movm_dist.log_prob(gt_coords)
                                                coord_regression_loss = -log_prob.mean()
                                                outputs["coord_regression_loss"] = coord_regression_loss
                                            except Exception as e:
                                                print(f"Warning: Failed to calculate coordinate MoVM loss: {e}")

                                        except Exception as e:
                                            print(
                                                f"Warning: Failed to calculate coordinate regression loss: {e}"
                                            )
                                else:
                                    print(
                                        f"Warning: Invalid slicing for coord head. Hidden size: {coordinate_hidden.size(0)}, Slice end: {slice_end}"
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

        # For generation, ensure lattice and coordinate predictions are included
        # even if the right segments weren't encountered during this forward pass
        if not self.training:
            # Check if continuous predictions are missing and we need to add them
            if (
                "lattice" in self.active_modules
                and "lattice_lengths" not in outputs
                and "lattice_angles" not in outputs
            ):
                # Find if there are any lattice tokens in the input
                lattice_mask = safe_segment_ids == self.config.SEGMENT_LATTICE

                # If no lattice tokens, use the last hidden state to generate predictions
                if not torch.any(lattice_mask):
                    # Use the last token's hidden state for prediction
                    last_hidden = hidden_states[:, -1]

                    # Generate lattice predictions using Mixture of Gaussians
                    # Generate parameters for mixture of Gaussians
                    lattice_mog_params = self.lattice_mog_head(last_hidden)
                    outputs["lattice_mog_params"] = lattice_mog_params
                    
                    # Get distribution and sample from it
                    mog_dist = self.lattice_mog_head.get_distribution(lattice_mog_params)
                    lattice_samples = mog_dist.sample()
                    
                    # Split samples into lengths and angles for compatibility
                    lattice_lengths = bound_lattice_lengths(lattice_samples[:, :3])
                    lattice_angles = bound_lattice_angles(lattice_samples[:, 3:])

                    outputs["lattice_lengths"] = lattice_lengths
                    outputs["lattice_angles"] = lattice_angles

            # Similarly for coordinate predictions
            if "atoms" in self.active_modules and "fractional_coords" not in outputs:
                coordinate_mask = safe_segment_ids == self.config.SEGMENT_COORDINATE

                if not torch.any(coordinate_mask):
                    # Use the last token's hidden state
                    last_hidden = hidden_states[:, -1].unsqueeze(0)

                    # Generate coordinate predictions using Mixture of Wrapped Normals
                    # Generate parameters for mixture of wrapped normals
                    coord_movm_params = self.fractional_coord_movm_head(last_hidden)
                    outputs["coord_movm_params"] = coord_movm_params
                    
                    # Get distribution and sample from it
                    movm_dist = self.fractional_coord_movm_head.get_distribution(coord_movm_params)
                    coord_samples = movm_dist.sample()
                    
                    # Ensure coordinates are in [0, 1) range
                    from .mixture_density import bound_mixture_fractional_coords
                    fractional_coords = bound_mixture_fractional_coords(coord_samples)
                        
                    outputs["fractional_coords"] = fractional_coords

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
        use_kv_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Generate sequences using autoregressive generation with space group and Wyckoff constraints.
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
            print("Using mixture density networks for predictions")
            print(f"Initial tokens shape: {input_ids.shape}")
            idx_to_token = (
                constraints.get("token_id_maps", {}).get("idx_to_token", {})
                if constraints
                else {}
            )

        # Always use continuous predictions
        continuous_predictions = {"lattice_lengths": [], "lattice_angles": [], "fractional_coords": []}

        if verbose:
            print(f"Active modules: {self.active_modules}")

        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Initialize constraint handler
        constraint_handler = CrystalConstraintHandler(constraints)

        # Initialize KV caches for all transformer layers if KV caching is enabled
        kv_caches = None
        if use_kv_cache:
            kv_caches = {
                # Composition encoder layers
                **{
                    f"composition_{i}": None
                    for i in range(len(self.composition_encoder))
                },
                # Space group encoder layers
                **{
                    f"space_group_{i}": None
                    for i in range(len(self.space_group_encoder))
                },
                # Lattice encoder layers
                **{f"lattice_{i}": None for i in range(len(self.lattice_encoder))},
                # Atom encoder layers
                **{f"atom_{i}": None for i in range(len(self.atom_encoder))},
                # Integration layers
                **{
                    f"integration_{i}": None
                    for i in range(len(self.integration_layers))
                },
                # Cross-attention layers
                "sg_from_comp": None,
                "lattice_from_sg": None,
                "atom_from_lattice": None,
            }
            if verbose:
                print("Initialized KV caches for generation with KV-caching enabled")

        while True:
            # Make sure all modules are active during generation
            if hasattr(self, "active_modules"):
                # Store original active modules to restore later
                original_active_modules = (
                    self.active_modules.copy() if self.active_modules else []
                )
                # Ensure all necessary modules are active for continuous prediction
                self.active_modules = ["composition", "space_group", "lattice", "atoms"]

                if verbose and len(original_active_modules) != len(self.active_modules):
                    print(
                        f"Setting active_modules for generation: {self.active_modules}"
                    )

            outputs = self.forward(
                input_ids=generated_ids,
                segment_ids=generated_segments,
                attention_mask=attention_mask,
                use_causal_mask=True,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
            )

            # Restore original active modules
            if hasattr(self, "active_modules"):
                self.active_modules = original_active_modules
                if verbose:
                    print(f"Restored active_modules: {self.active_modules}")

            # Always collect continuous predictions
            if "lattice_lengths" in outputs and "lattice_angles" in outputs:
                if verbose:
                    print(
                        f"Found lattice predictions: lengths={outputs['lattice_lengths'].shape}, angles={outputs['lattice_angles'].shape}"
                    )
                continuous_predictions["lattice_lengths"].append(
                    outputs["lattice_lengths"]
                )
                continuous_predictions["lattice_angles"].append(
                    outputs["lattice_angles"]
                )
            elif verbose:
                print(
                    f"Missing lattice predictions in output keys: {list(outputs.keys())}"
                )

            if "fractional_coords" in outputs:
                if verbose:
                    print(
                        f"Found coordinate predictions: coords={outputs['fractional_coords'].shape}"
                    )
                continuous_predictions["fractional_coords"].append(
                    outputs["fractional_coords"]
                )
            elif verbose:
                print(
                    f"Missing coordinate predictions in output keys: {list(outputs.keys())}"
                )

            next_token_logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Apply constraints based on current segment type
            current_segments = generated_segments[:, -1].cpu().numpy()

            for i in range(batch_size):
                if unfinished_sequences[i]:
                    current_segment = current_segments[i]

                    # Apply Wyckoff position constraints
                    if (
                        current_segment == self.config.SEGMENT_WYCKOFF
                        and hasattr(self.config, "apply_wyckoff_constraints")
                        and self.config.apply_wyckoff_constraints
                    ):
                        wyckoff_mask = constraint_handler.get_wyckoff_mask(i)
                        if (
                            wyckoff_mask is not None
                            and wyckoff_mask.shape[0] <= next_token_logits.shape[1]
                        ):
                            next_token_logits[i, ~wyckoff_mask] = float("-inf")

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

            # Update constraint handler with generated tokens
            for i in range(batch_size):
                if unfinished_sequences[i]:
                    token_id = next_tokens[i].item()
                    current_segment = current_segments[i]

                    # Track space group tokens
                    if current_segment == self.config.SEGMENT_SPACE_GROUP:
                        constraint_handler.update_space_group(i, token_id)

                    # Track Wyckoff position tokens
                    elif current_segment == self.config.SEGMENT_WYCKOFF:
                        constraint_handler.update_wyckoff_position(i, token_id)

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

        # Process continuous predictions
        if continuous_predictions:
            if verbose:
                print("Processing continuous predictions")
                print(
                    f"Lattice lengths collected: {len(continuous_predictions['lattice_lengths'])}"
                )
                print(
                    f"Lattice angles collected: {len(continuous_predictions['lattice_angles'])}"
                )
                print(
                    f"Fractional coords collected: {len(continuous_predictions['fractional_coords'])}"
                )

            if (
                continuous_predictions["lattice_lengths"]
                and len(continuous_predictions["lattice_lengths"]) > 0
            ):
                # Combine all predictions - for generation we want the last one for each sequence
                try:
                    # For debugging, examine the shapes before concatenation
                    if verbose:
                        for i, tensor in enumerate(
                            continuous_predictions["lattice_lengths"]
                        ):
                            print(f"Lattice length tensor {i} shape: {tensor.shape}")

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
                        print(
                            f"Lattice lengths range: {result['continuous_lattice_lengths'].min().item():.3f} to {result['continuous_lattice_lengths'].max().item():.3f}"
                        )
                        print(
                            f"Lattice angles range: {result['continuous_lattice_angles'].min().item():.3f} to {result['continuous_lattice_angles'].max().item():.3f}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous lattice predictions: {e}"
                    )
                    print(
                        f"Tensor shapes: {[t.shape for t in continuous_predictions['lattice_lengths']]}"
                    )
                    print(
                        f"Tensor devices: {[t.device for t in continuous_predictions['lattice_lengths']]}"
                    )

            if (
                continuous_predictions["fractional_coords"]
                and len(continuous_predictions["fractional_coords"]) > 0
            ):
                try:
                    # For debugging, examine the shapes before concatenation
                    if verbose:
                        for i, tensor in enumerate(
                            continuous_predictions["fractional_coords"]
                        ):
                            print(f"Coordinate tensor {i} shape: {tensor.shape}")

                    result["continuous_fractional_coords"] = torch.cat(
                        continuous_predictions["fractional_coords"], dim=0
                    )
                    has_continuous_coords = True

                    if verbose:
                        print(
                            f"Successfully captured continuous coordinate predictions with shape: {result['continuous_fractional_coords'].shape}"
                        )
                        print(
                            f"Coordinate values range: {result['continuous_fractional_coords'].min().item():.3f} to {result['continuous_fractional_coords'].max().item():.3f}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous coordinate predictions: {e}"
                    )
                    print(
                        f"Tensor shapes: {[t.shape for t in continuous_predictions['fractional_coords']]}"
                    )
                    print(
                        f"Tensor devices: {[t.device for t in continuous_predictions['fractional_coords']]}"
                    )

        # Add flags to indicate whether continuous predictions were successfully obtained
        result["has_continuous_lattice"] = has_continuous_lattice
        result["has_continuous_coords"] = has_continuous_coords

        if verbose:
            print(
                f"Final results - has_continuous_lattice: {has_continuous_lattice}, has_continuous_coords: {has_continuous_coords}"
            )

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
