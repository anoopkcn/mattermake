import torch
import torch.nn as nn
from typing import Dict, Optional


class DecoderBase(nn.Module):
    """Base class for all modular decoders"""

    def __init__(self, d_model: int):
        super().__init__()
        self.d_model = d_model

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process hidden states and return predictions

        Args:
            hidden_states: Tensor with shape (batch_size, seq_len, d_model)
            encoder_contexts: Dictionary of encoder outputs
            mask: Optional attention mask

        Returns:
            Dictionary of predictions
        """
        raise NotImplementedError


class AtomTypeDecoder(DecoderBase):
    """Predicts atom types"""

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict atom types

        Args:
            hidden_states: Tensor with shape (batch_size, seq_len, d_model)
            encoder_contexts: Dictionary of encoder outputs
            mask: Optional attention mask

        Returns:
            Dictionary with type_logits
        """
        # Apply head to get logits
        type_logits = self.head(hidden_states)

        # Safety check for NaNs
        if torch.isnan(type_logits).any():
            type_logits = torch.nan_to_num(type_logits)

        return {"type_logits": type_logits}


class AtomCoordinateDecoder(DecoderBase):
    """Predicts fractional coordinates using Gaussian distribution"""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__(d_model)
        self.head = nn.Linear(d_model, 6)  # mean and log_var for x, y, z
        self.eps = eps

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict coordinates distribution parameters

        Args:
            hidden_states: Tensor with shape (batch_size, seq_len, d_model)
            encoder_contexts: Dictionary of encoder outputs
            mask: Optional attention mask

        Returns:
            Dictionary with coord_mean and coord_log_var
        """
        # Apply head to get raw parameters
        coord_params = self.head(hidden_states)

        # Process and stabilize parameters
        coord_params = torch.nan_to_num(coord_params, nan=0.0, posinf=1e6, neginf=-1e6)
        mean_raw, log_var_raw = torch.chunk(coord_params, 2, dim=-1)

        # Apply appropriate activations
        coord_mean = torch.sigmoid(mean_raw)
        coord_log_var = torch.clamp(log_var_raw, min=-10.0, max=0.0)

        # Final safeguard
        coord_mean = torch.nan_to_num(coord_mean, nan=0.5)
        coord_log_var = torch.nan_to_num(coord_log_var, nan=-3.0)

        return {"coord_mean": coord_mean, "coord_log_var": coord_log_var}


class DecoderRegistry(nn.Module):
    """Manages multiple decoders and combines their outputs"""

    def __init__(self, decoders: Dict[str, DecoderBase]):
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process hidden states through all decoders

        Args:
            hidden_states: Tensor with shape (batch_size, seq_len, d_model)
            encoder_contexts: Dictionary of encoder outputs
            mask: Optional attention mask

        Returns:
            Combined dictionary of all decoder outputs
        """
        # Collect outputs from all decoders
        outputs = {}
        for name, decoder in self.decoders.items():
            decoder_outputs = decoder(hidden_states, encoder_contexts, mask)
            outputs.update(decoder_outputs)

        return outputs
