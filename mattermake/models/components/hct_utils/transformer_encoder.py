import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple

from mattermake.models.components.hct_utils.attention import MultiHeadAttention


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


class TransformerEncoder(nn.Module):
    """Transformer encoder with multiple layers and support for KV-caching"""

    def __init__(self, config, num_layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [TransformerLayer(config) for _ in range(num_layers)]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_causal_mask: bool = True,
        kv_caches: Optional[Dict[str, Dict[str, torch.Tensor]]] = None,
        use_kv_cache: bool = False,
        cache_prefix: str = "",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, Dict[str, torch.Tensor]]]]:
        """Process hidden states through multiple transformer layers

        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask of shape [batch_size, seq_length]
            use_causal_mask: Whether to use causal masking for attention
            kv_caches: Optional key-value caches for attention
            use_kv_cache: Whether to use key-value caching
            cache_prefix: Prefix for cache keys when using kv_caches

        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        for i, layer in enumerate(self.layers):
            layer_cache = (
                None if kv_caches is None else kv_caches.get(f"{cache_prefix}_{i}")
            )

            if use_kv_cache and kv_caches is not None:
                hidden_states, new_cache = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                    kv_cache=layer_cache,
                    use_kv_cache=True,
                )
                kv_caches[f"{cache_prefix}_{i}"] = new_cache
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    causal_mask=use_causal_mask,
                )

        return hidden_states
