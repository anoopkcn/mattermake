import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """Standard positional encoding from original Transformer paper."""

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


class RotaryPositionalEmbedding(nn.Module):
    """
    Rotary Positional Embeddings (RoPE) implementation.
    Based on the paper: https://arxiv.org/abs/2104.09864

    RoPE encodes absolute positions with a rotation matrix that naturally
    incorporates explicit relative position dependency.
    """

    def __init__(
        self, d_model: int, dropout: float = 0.1, max_len: int = 5000, base: int = 10000
    ):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        self.base = base

        # Precompute the frequency tensor for faster access during forward pass
        self._precompute_freqs(max_len, d_model)

    def _precompute_freqs(self, seq_len: int, dim: int):
        """Precompute the frequency tensor for faster access during forward pass"""
        # First, create a sequence of indices for positions
        pos = torch.arange(0, seq_len, dtype=torch.float)

        # Create the decreasing frequency sequence (power decay)
        # For each dimension pair, we have a frequency
        freqs = torch.arange(0, dim, 2, dtype=torch.float)
        freqs = self.base ** (-freqs / dim)

        # Outer product of positions and frequencies
        freqs = torch.outer(pos, freqs)

        # Compute sine and cosine frequencies
        # [seq_len, d_model/2]
        freqs_cos = torch.cos(freqs)  # cosine of frequencies
        freqs_sin = torch.sin(freqs)  # sine of frequencies

        # Register cos/sin as buffers so they're moved to the right device automatically
        self.register_buffer("freqs_cos", freqs_cos)
        self.register_buffer("freqs_sin", freqs_sin)

    def _apply_rotary_pos_emb(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply the rotary positional embeddings to the input tensor"""
        seq_len = x.size(1)

        # Make sure we don't exceed the precomputed positions
        if seq_len + offset > self.max_len:
            raise ValueError(
                f"Sequence length ({seq_len + offset}) exceeds maximum length ({self.max_len})"
            )

        # Get the right part of the precomputed frequency tensor
        cos = self.freqs_cos[offset : offset + seq_len].unsqueeze(
            0
        )  # [1, seq, d_model/2]
        sin = self.freqs_sin[offset : offset + seq_len].unsqueeze(
            0
        )  # [1, seq, d_model/2]

        # For each embedding dimension, we need to rotate pairs of dimensions
        # Extract the even and odd dimensions - reshape to make this easier
        x_reshaped = x.view(x.shape[0], x.shape[1], -1, 2)
        x_even = x_reshaped[..., 0]  # [batch, seq, d_model/2]
        x_odd = x_reshaped[..., 1]  # [batch, seq, d_model/2]

        # Apply rotation using complex multiplication-like operations
        x_even_rot = x_even * cos - x_odd * sin
        x_odd_rot = x_even * sin + x_odd * cos

        # Recombine the rotated values into the original shape
        x_rot = torch.stack([x_even_rot, x_odd_rot], dim=-1)
        x_rot = x_rot.view(x.shape)

        return x_rot

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """Apply rotary position embeddings to input tensor.

        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] or [seq_len, batch_size, embedding_dim]
            offset: Optional position offset for continued sequences

        Returns:
            Tensor with rotary embeddings applied, in same format as input
        """
        # RoPE expects [batch, seq, dim] but original PE might get [seq, batch, dim]
        # Handle both potential formats
        is_seq_first = False
        if (
            x.size(0) <= x.size(1) and x.size(0) <= 32
        ):  # Heuristic to detect seq_len first
            # Likely [seq_len, batch_size, dim] format, swap dims
            x = x.transpose(0, 1)
            is_seq_first = True

        # Apply rotary embeddings
        x_with_rope = self._apply_rotary_pos_emb(x, offset)
        x_with_rope = self.dropout(x_with_rope)

        # Return in the original format
        if is_seq_first:
            x_with_rope = x_with_rope.transpose(0, 1)

        return x_with_rope
