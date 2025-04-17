import torch
import torch.nn as nn
from typing import Dict, Optional, List


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


class SpaceGroupDecoder(DecoderBase):
    """Explicitly dedicated Space Group decoder"""

    def __init__(self, d_model: int, sg_vocab_size: int = 230):
        super().__init__(d_model)
        self.sg_head = nn.Linear(d_model, sg_vocab_size)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict space group logits

        Args:
            hidden_states: Usually from composition encoder (batch_size, 1, d_model)
            encoder_contexts: All encoder outputs
            mask: Optional mask

        Returns:
            Dictionary with sg_logits
        """
        # Use composition context directly or input hidden states
        input_states = hidden_states

        # Handle sequence length > 1 case (take first token)
        if input_states.size(1) > 1:
            input_states = input_states[:, 0:1, :]

        # Apply head
        sg_logits = self.sg_head(input_states.squeeze(1))

        # Safety check for NaNs
        if torch.isnan(sg_logits).any():
            sg_logits = torch.nan_to_num(sg_logits)

        return {"sg_logits": sg_logits}


class LatticeDecoder(DecoderBase):
    """Explicitly dedicated Lattice decoder"""

    def __init__(self, d_model: int):
        super().__init__(d_model)
        self.lattice_head = nn.Linear(d_model, 6 * 2)  # mean + log_var per param
        # Initialize with small weights for stability
        nn.init.xavier_normal_(self.lattice_head.weight, gain=0.5)
        nn.init.zeros_(self.lattice_head.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Predict lattice parameters

        Args:
            hidden_states: Fused context for lattice (batch_size, 1, d_model)
            encoder_contexts: All encoder outputs
            mask: Optional mask

        Returns:
            Dictionary with lattice_mean and lattice_log_var
        """
        # Handle sequence length > 1 case (take first token)
        if hidden_states.size(1) > 1:
            hidden_states = hidden_states[:, 0:1, :]

        # Process and predict
        lattice_params = self.lattice_head(hidden_states.squeeze(1))

        # Process and stabilize parameters
        lattice_params = torch.nan_to_num(
            lattice_params, nan=0.0, posinf=1e6, neginf=-1e6
        )

        # Split into mean and log_var
        lattice_mean, lattice_log_var = torch.chunk(lattice_params, 2, dim=-1)

        # Safety clamp
        lattice_log_var = torch.clamp(lattice_log_var, -20, 2)

        return {"lattice_mean": lattice_mean, "lattice_log_var": lattice_log_var}


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


class OrderedDecoderRegistry(nn.Module):
    """Manages multiple decoders and applies them in a specific order"""

    def __init__(
        self, decoders: Dict[str, DecoderBase], order: Optional[List[str]] = None
    ):
        super().__init__()
        self.decoders = nn.ModuleDict(decoders)
        # If no order specified, use alphabetical
        self.order = order if order is not None else sorted(decoders.keys())

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_contexts: Dict[str, torch.Tensor],
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Process hidden states through all decoders in order

        Args:
            hidden_states: Tensor with shape (batch_size, seq_len, d_model)
            encoder_contexts: Dictionary of encoder outputs
            mask: Optional attention mask

        Returns:
            Combined dictionary of all decoder outputs
        """
        # Collect outputs from all decoders, following the specified order
        outputs = {}
        for name in self.order:
            if name in self.decoders:
                decoder_outputs = self.decoders[name](
                    hidden_states, encoder_contexts, mask
                )
                outputs.update(decoder_outputs)

        return outputs
