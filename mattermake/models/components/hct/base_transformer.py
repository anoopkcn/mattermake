import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple

from .attention import MultiHeadAttention


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


class BaseTransformer(nn.Module):
    """Base transformer model with common functionality"""

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Common embeddings
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

        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # Output projection
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Initialize weights with more stability for MoG/MoVM heads"""
        if isinstance(module, nn.Linear):
            # More conservative initialization for specific layers prone to instability
            if (
                "mixture_proj" in module._get_name()
                or "mog_head" in module._get_name()
                or "movm_head" in module._get_name()
            ):
                # More conservative initialization for mixture density network projections
                module.weight.data.normal_(
                    mean=0.0, std=min(0.01, self.config.initializer_range / 2)
                )
                if module.bias is not None:
                    # Start with slightly positive bias for standard deviations
                    if (
                        module.out_features % 3 == 0
                    ):  # Likely a mixture component output
                        # Every third output might be a scale parameter (needs to be positive)
                        third = module.out_features // 3
                        # Initialize std/scale biases to small positive values (log space for exp activation)
                        module.bias.data[2 * third :].fill_(0.0)
                    else:
                        module.bias.data.zero_()
            elif "space_group_head" in module._get_name():
                # Even more conservative initialization for space group head
                module.weight.data.normal_(mean=0.0, std=0.01)
                if module.bias is not None:
                    module.bias.data.zero_()
            else:
                # Standard initialization for other layers
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

    def get_input_embeddings(self, input_ids, segment_ids):
        """Get combined embeddings for the inputs"""
        # Apply safety clamps
        safe_input_ids = torch.clamp(input_ids, max=self.config.vocab_size - 1)
        safe_segment_ids = torch.clamp(segment_ids, max=self.config.type_vocab_size - 1)

        # Get position IDs
        position_ids = self.get_position_ids(input_ids)

        # Get all embeddings
        token_embeds = self.token_embeddings(safe_input_ids)
        position_embeds = self.position_embeddings(position_ids)
        segment_embeds = self.token_type_embeddings(safe_segment_ids)

        # Combine embeddings
        embeddings = token_embeds + position_embeds + segment_embeds
        embeddings = self.layer_norm(embeddings)
        embeddings = self.dropout(embeddings)

        return embeddings
