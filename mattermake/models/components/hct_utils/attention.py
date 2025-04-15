import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Union, Tuple


class MultiHeadAttention(nn.Module):
    """Multi-head attention module with causal masking option and KV-caching support
    
    Input shapes:
      - hidden_states: [batch_size, seq_length, hidden_size]
      - attention_mask: [batch_size, seq_length] or [batch_size, 1, seq_length, kv_seq_length]
      - key_value_states: [batch_size, kv_seq_length, hidden_size] or None
      
    Output shape: [batch_size, seq_length, hidden_size]
    """

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
        """Forward pass with KV-caching support"""
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
                kv_states = key_value_states[:, -1:, :]
            else:
                kv_states = hidden_states[:, -1:, :]
                
            new_k = self.k_proj(kv_states)
            new_v = self.v_proj(kv_states)

            # Reshape new keys and values
            new_k = new_k.view(batch_size, 1, self.num_heads, self.head_size).transpose(1, 2)
            new_v = new_v.view(batch_size, 1, self.num_heads, self.head_size).transpose(1, 2)

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
            k = k.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(1, 2)
            v = v.view(batch_size, kv_seq_len, self.num_heads, self.head_size).transpose(1, 2)

            # Create new cache
            if use_kv_cache:
                new_cache = {"k": k, "v": v}

        # Reshape query
        q = q.view(batch_size, seq_len, self.num_heads, self.head_size).transpose(1, 2)

        # Calculate attention scores
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
            # Handle different attention mask shapes
            if attention_mask.dim() == 2:  # [batch_size, seq_length]
                # Convert to [batch_size, 1, seq_length, 1]
                extended_mask = attention_mask.unsqueeze(1).unsqueeze(-1)
                # Broadcast to [batch_size, num_heads, seq_length, kv_seq_length]
                extended_mask = (1.0 - extended_mask) * -10000.0
                attn_scores = attn_scores + extended_mask
            elif attention_mask.dim() == 4:  # [batch_size, 1, seq_length, kv_seq_length]
                attn_scores = attn_scores + attention_mask

        # Apply softmax to get attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout_layer(attn_weights)

        # Apply attention to values
        context = torch.matmul(attn_weights, v)

        # Reshape context tensor
        context = context.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)

        # Project to output
        output = self.out_proj(context)

        # Return with cache if requested
        if use_kv_cache and new_cache is not None:
            return output, new_cache
        return output