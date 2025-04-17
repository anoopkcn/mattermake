import torch
import torch.nn as nn
from typing import Dict, List, Optional


class ModularCrossAttention(nn.Module):
    """Performs separate cross-attention to each encoder, then fuses results"""

    def __init__(
        self, encoder_names: List[str], d_model: int, nhead: int, dropout: float = 0.1
    ):
        super().__init__()
        self.d_model = d_model  # Store d_model for dynamic fusion

        # Store the possible encoder names (superset of what might be used)
        self.encoder_names = encoder_names

        # Create separate MultiheadAttention modules for each possible encoder
        self.cross_attns = nn.ModuleDict(
            {
                name: nn.MultiheadAttention(
                    d_model, nhead, dropout=dropout, batch_first=True
                )
                for name in encoder_names
            }
        )

        # Dynamic fusion approach - we'll handle the projection in forward
        # Instead of a fixed linear layer, use a weight matrix that can be indexed
        # Create a projection weight for each encoder
        self.fusion_weights = nn.Parameter(
            torch.randn(len(encoder_names), d_model, d_model)
        )
        self.fusion_bias = nn.Parameter(torch.zeros(d_model))
        # Initialize using xavier uniform
        nn.init.xavier_uniform_(self.fusion_weights)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        encoder_outputs: Dict[str, torch.Tensor],
        key_padding_masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Forward pass for ModularCrossAttention.

        Args:
            query: Decoder query tensor (batch_size, target_len, d_model)
            encoder_outputs: Dict mapping encoder names to their output tensors
                               (batch_size, src_len, d_model)
            key_padding_masks: Optional Dict mapping encoder names to their key padding masks
                                  (batch_size, src_len), True where padded.

        Returns:
            fused_context: Fused context tensor (batch_size, target_len, d_model)
        """
        attended_outputs = []
        for name in self.encoder_names:
            if name in encoder_outputs:
                # Prepare key_padding_mask if provided
                kpm = key_padding_masks.get(name) if key_padding_masks else None
                # Perform cross-attention
                # Note: If encoder output has seq_len=1, MHA might behave unexpectedly
                # with key_padding_mask. Ensure masks align with actual sequence lengths.
                output, _ = self.cross_attns[name](
                    query=query,
                    key=encoder_outputs[name],
                    value=encoder_outputs[name],
                    key_padding_mask=kpm,
                )
                attended_outputs.append(output)
            else:
                # Handle case where an expected encoder output might be missing
                # Option 1: Raise error
                # raise ValueError(f"Missing encoder output for '{name}'")
                # Option 2: Append zeros (less ideal, might mask issues)
                # attended_outputs.append(torch.zeros_like(query))
                # Option 3: Skip (implicitly handled by not appending)
                pass

        # Ensure at least one attention was performed
        if not attended_outputs:
            # Return zero tensor or handle appropriately
            return torch.zeros_like(query)

        # New dynamic fusion approach
        if not attended_outputs:
            # Return zero tensor if no attention was performed
            return torch.zeros_like(query)

        # Initialize output tensor
        batch_size, target_len, _ = query.shape
        fused = torch.zeros(batch_size, target_len, self.d_model, device=query.device)

        # Keep track of which encoders were actually used
        used_encoder_indices = []

        # Process each attended output separately
        for i, name in enumerate(self.encoder_names):
            if name in encoder_outputs and i < len(attended_outputs):
                # Get the attended output for this encoder
                attended = attended_outputs[len(used_encoder_indices)]

                # Apply this encoder's fusion weights
                # (batch, target_len, d_model) @ (d_model, d_model) -> (batch, target_len, d_model)
                projected = torch.matmul(attended, self.fusion_weights[i])

                # Add to the fused output
                fused = fused + projected
                used_encoder_indices.append(i)

        # Apply bias if any encoders were used
        if used_encoder_indices:
            # Average the projections and add bias
            fused = fused / len(used_encoder_indices) + self.fusion_bias

        # Apply residual connection, dropout, and normalization
        output = self.norm(query + self.dropout(fused))

        return output
