import torch
import torch.nn as nn


class EncoderBase(nn.Module):
    """Base class for all modular encoders"""

    def __init__(self, d_output: int):
        super().__init__()
        self.d_output = d_output

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns context vector/sequence for conditioning"""
        raise NotImplementedError


class CompositionEncoder(EncoderBase):
    """Encodes element composition"""

    def __init__(
        self,
        element_vocab_size: int,
        d_model: int,
        nhead: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float = 0.1,
    ):
        super().__init__(d_output=d_model)
        self.processor = nn.Linear(element_vocab_size, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(self, composition: torch.Tensor) -> torch.Tensor:
        """
        Args:
            composition: (batch_size, element_vocab_size)
        Returns:
            context: (batch_size, 1, d_model)
        """
        comp_input = self.processor(composition).unsqueeze(1)
        return self.encoder(comp_input)  # (batch, 1, d_model)


class SpaceGroupEncoder(EncoderBase):
    """Encodes space group using one-hot encoding instead of direct embedding"""

    def __init__(self, sg_vocab_size: int, sg_embed_dim: int, d_model: int):
        super().__init__(d_output=d_model)
        # Store parameters
        self.sg_vocab_size = sg_vocab_size
        self.sg_embed_dim = sg_embed_dim

        # Use a linear projection directly on one-hot encoded input
        # Functionally equivalent to embedding but more explicit about one-hot nature
        self.projector1 = nn.Linear(sg_vocab_size, sg_embed_dim)
        self.projector2 = nn.Linear(sg_embed_dim, d_model)

        # Optional activation between projections
        self.activation = nn.ReLU()

    def forward(self, spacegroup: torch.Tensor) -> torch.Tensor:
        """
        Args:
            spacegroup: (batch_size, 1) with values 1-230
        Returns:
            context: (batch_size, 1, d_model)
        """
        # Ensure input is long type and handle squeezing
        if spacegroup.ndim > 1 and spacegroup.size(1) == 1:
            spacegroup = spacegroup.squeeze(1)

        device = spacegroup.device
        batch_size = spacegroup.size(0)

        # Map 1-230 to 0-229 for one-hot indexing
        sg_idx = spacegroup.long() - 1  # Map 1-230 -> 0-229

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.sg_vocab_size, device=device)
        one_hot.scatter_(1, sg_idx.unsqueeze(1), 1.0)

        # Apply projections with activation in between
        hidden = self.activation(self.projector1(one_hot))
        output = self.projector2(hidden)

        return output.unsqueeze(1)  # (batch, 1, d_model)


class LatticeEncoder(EncoderBase):
    """Encodes lattice parameters to a latent representation"""

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 64,
        equivariant: bool = False,  # Start with simpler non-equivariant version
    ):
        super().__init__(d_output=d_model)

        # Store parameters
        self.d_model = d_model
        self.latent_dim = latent_dim

        # 6 lattice parameters (a, b, c, alpha, beta, gamma)
        input_dim = 6

        # Option 1: Simple MLP (non-equivariant)
        if not equivariant:
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, d_model),
            )
        # Option 2: Placeholder for future equivariant version
        else:
            # This would be replaced with a proper equivariant network
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, latent_dim),
                nn.ReLU(),
                nn.Linear(latent_dim, d_model),
            )

    def forward(self, lattice: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lattice: (batch_size, 6) lattice parameters
        Returns:
            context: (batch_size, 1, d_model)
        """
        # Ensure input is properly shaped
        if lattice.dim() > 2:
            lattice = lattice.squeeze(1)  # Remove any singleton dimensions

        # Apply encoder
        encoded = self.encoder(lattice)

        # Add sequence dimension if needed
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)  # (batch_size, 1, d_model)

        return encoded


# Add more encoders as needed (WyckoffEncoder, ConstraintEncoder, etc.)
