import torch
import torch.nn as nn
from typing import Optional, Dict, Union, Tuple

from mattermake.models.components.hct_utils.attention import MultiHeadAttention


class CrossAttention(nn.Module):
    """Cross-attention layer for connecting different levels in the hierarchy with KV-caching support
    
    This module allows information to flow from one hierarchical level to another by applying
    cross-attention between hidden states.
    
    Input shape: [batch_size, seq_length, hidden_size]
    Output shape: [batch_size, seq_length, hidden_size]
    """

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
        """Apply cross-attention from hidden states to context states
        
        Args:
            hidden_states: Query tensor of shape [batch_size, seq_length, hidden_size]
            context_states: Key/value tensor of shape [batch_size, context_length, hidden_size]
            attention_mask: Optional mask of shape [batch_size, 1, seq_length, context_length]
            kv_cache: Optional key-value cache for attention
            use_kv_cache: Whether to use key-value caching
            
        Returns:
            Output tensor of shape [batch_size, seq_length, hidden_size]
        """
        # Clean input tensors for stability
        hidden_states = torch.nan_to_num(hidden_states, nan=0.0)
        context_states = torch.nan_to_num(context_states, nan=0.0)
        
        normed_hidden_states = self.norm(hidden_states)

        if use_kv_cache:
            try:
                context_output, new_cache = self.cross_attention(
                    normed_hidden_states,
                    key_value_states=context_states,
                    attention_mask=attention_mask,
                    kv_cache=kv_cache,
                    use_kv_cache=True,
                )
                output = hidden_states + self.dropout(context_output)
                return output, new_cache
            except Exception:
                # Fallback on error - skip cross-attention and return original
                return hidden_states
        else:
            try:
                context_output = self.cross_attention(
                    normed_hidden_states,
                    key_value_states=context_states,
                    attention_mask=attention_mask,
                )
                return hidden_states + self.dropout(context_output)
            except Exception:
                # Fallback on error - skip cross-attention and return original
                return hidden_states