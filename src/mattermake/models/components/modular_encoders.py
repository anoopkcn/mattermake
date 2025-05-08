import torch
import torch.nn as nn
from typing import Optional


class EncoderBase(nn.Module):
    """Base class for all modular encoders"""

    def __init__(self, d_output: int):
        super().__init__()
        self.d_output = d_output
        # Optional projector for conditioning context - initialized in subclasses if needed
        self.condition_projector = None

    def forward(
        self, x: torch.Tensor, condition_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Returns context vector/sequence for conditioning"""
        raise NotImplementedError

    def _apply_conditioning(
        self, x: torch.Tensor, condition_context: Optional[torch.Tensor]
    ) -> torch.Tensor:
        """Helper to apply conditioning if projector and context exist."""
        if condition_context is not None and self.condition_projector is not None:
            # Project the condition context
            projected_condition = self.condition_projector(condition_context)

            # Handle shapes: condition_context might be (B, 1, D) or (B, D)
            # x might be (B, 1, D), (B, L, D), or (B, D)
            if x.dim() == 3 and projected_condition.dim() == 2:
                # Unsqueeze condition: (B, D) -> (B, 1, D)
                projected_condition = projected_condition.unsqueeze(1)
            elif (
                x.dim() == 2
                and projected_condition.dim() == 3
                and projected_condition.size(1) == 1
            ):
                # Squeeze condition: (B, 1, D) -> (B, D)
                projected_condition = projected_condition.squeeze(1)

            # Expand if necessary (e.g., global context for sequence encoder)
            if x.dim() == 3 and projected_condition.dim() == 3:
                if x.size(1) > 1 and projected_condition.size(1) == 1:
                    # Expand (B, 1, D) -> (B, L, D) to match x
                    projected_condition = projected_condition.expand(-1, x.size(1), -1)

            # Ensure shapes match for addition
            if x.shape == projected_condition.shape:
                x = x + projected_condition
            else:
                # Fallback or raise error if shapes still mismatch after adjustments
                print(
                    f"Warning: Shape mismatch in conditioning. x: {x.shape}, condition: {projected_condition.shape}. Skipping conditioning."
                )

        return x


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
        # No condition projector needed for the first encoder usually

    def forward(
        self,
        composition: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            composition: (batch_size, element_vocab_size)
            condition_context: Optional conditioning tensor (ignored here)
        Returns:
            context: (batch_size, 1, d_model)
        """
        comp_input = self.processor(composition).unsqueeze(1)
        # Conditioning could potentially be added here if needed in the future
        # comp_input = self._apply_conditioning(comp_input, condition_context)
        return self.encoder(comp_input)  # (batch, 1, d_model)


class SpaceGroupEncoder(EncoderBase):
    """Encodes space group using one-hot encoding instead of direct embedding"""

    def __init__(
        self,
        sg_vocab_size: int,
        sg_embed_dim: int,
        d_model: int,
        has_conditioning: bool = False,
    ):
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

        # Optional conditioning
        if has_conditioning:
            self.condition_projector = nn.Linear(d_model, sg_embed_dim)

    def forward(
        self, spacegroup: torch.Tensor, condition_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            spacegroup: (batch_size, 1) with values 1-230
            condition_context: Optional conditioning tensor
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

        # Apply conditioning after first projection
        hidden = self._apply_conditioning(hidden, condition_context)

        output = self.projector2(hidden)

        return output.unsqueeze(1)  # (batch, 1, d_model)


class LatticeEncoder(EncoderBase):
    """Encodes the 3x3 lattice matrix to a latent representation, optionally conditioned."""

    def __init__(
        self,
        d_model: int,
        latent_dim: int = 64,
        equivariant: bool = False,  # Start with simpler non-equivariant version
        has_conditioning: bool = False,  # Flag to indicate if conditioning is expected
    ):
        super().__init__(d_output=d_model)

        # Store parameters
        self.d_model = d_model
        self.latent_dim = latent_dim

        # 9 lattice matrix elements (flattened 3x3)
        input_dim = 9

        # Initialize condition projector if needed
        if has_conditioning:
            # Project conditioning (e.g., SG/Comp context) to latent_dim to add within MLP
            self.condition_projector = nn.Linear(d_model, latent_dim)

        # Option 1: Simple MLP (non-equivariant)
        if not equivariant:
            self.layer1 = nn.Linear(input_dim, latent_dim)  # Input is now 9
            self.act1 = nn.ReLU()
            self.layer2 = nn.Linear(latent_dim, latent_dim)
            self.act2 = nn.ReLU()
            self.layer3 = nn.Linear(latent_dim, d_model)
        # Option 2: Placeholder for future equivariant version
        else:
            # Replace with equivariant layers suitable for 3x3 matrix input
            self.layer1 = nn.Linear(input_dim, latent_dim)  # Input is now 9
            self.act1 = nn.ReLU()
            self.layer2 = nn.Linear(latent_dim, latent_dim)
            self.act2 = nn.ReLU()
            self.layer3 = nn.Linear(latent_dim, d_model)

    def _convert_params_to_matrix(self, lattice_params: torch.Tensor) -> torch.Tensor:
        """Convert 6 lattice parameters (a, b, c, alpha, beta, gamma) to a flattened 3x3 matrix.

        Args:
            lattice_params: Tensor of shape (batch_size, 6) with [a, b, c, alpha, beta, gamma]
                            where angles are in degrees
        Returns:
            Flattened lattice matrix of shape (batch_size, 9)
        """
        # Batch size and device are inferred from the parameters

        # Extract parameters
        a, b, c = lattice_params[:, 0], lattice_params[:, 1], lattice_params[:, 2]
        alpha, beta, gamma = (
            lattice_params[:, 3],
            lattice_params[:, 4],
            lattice_params[:, 5],
        )

        # Convert angles from degrees to radians
        alpha_rad = alpha * (torch.pi / 180.0)
        beta_rad = beta * (torch.pi / 180.0)
        gamma_rad = gamma * (torch.pi / 180.0)

        # Calculate matrix elements (simplifying to standard representation)
        # First row: [a, 0, 0]
        m11 = a
        m12 = torch.zeros_like(a)
        m13 = torch.zeros_like(a)

        # Second row: [b*cos(gamma), b*sin(gamma), 0]
        m21 = b * torch.cos(gamma_rad)
        m22 = b * torch.sin(gamma_rad)
        m23 = torch.zeros_like(a)

        # Third row: more complex due to alpha, beta constraints
        m31 = c * torch.cos(beta_rad)
        # Calculate m32 using constraints
        cos_alpha_star = (
            torch.cos(gamma_rad) * torch.cos(beta_rad) - torch.cos(alpha_rad)
        ) / (torch.sin(gamma_rad) * torch.sin(beta_rad))
        cos_alpha_star = torch.clamp(
            cos_alpha_star, -1.0, 1.0
        )  # Ensure value is within valid range
        m32 = c * torch.sin(beta_rad) * cos_alpha_star

        # Calculate m33 using constraints
        m33_sq = c**2 * torch.sin(beta_rad) ** 2 * (1 - cos_alpha_star**2)
        m33 = torch.sqrt(torch.clamp(m33_sq, min=1e-8))  # Avoid negative values in sqrt

        # Create flattened matrix
        matrix_flat = torch.stack([m11, m12, m13, m21, m22, m23, m31, m32, m33], dim=-1)
        return matrix_flat

    def forward(
        self,
        lattice_input: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            lattice_input: Either:
                - (batch_size, 3, 3) matrix
                - (batch_size, 9) flattened matrix
                - (batch_size, 6) lattice parameters [a, b, c, alpha, beta, gamma]
            condition_context: Optional (batch_size, 1, d_model) tensor (e.g., from SG/Comp encoder)
        Returns:
            context: (batch_size, 1, d_model)
        """
        # Ensure input is flattened (batch_size, 9)
        if lattice_input.dim() == 3 and lattice_input.shape[1:] == (3, 3):
            lattice_flat = lattice_input.reshape(
                lattice_input.size(0), -1
            )  # Flatten to (batch_size, 9)
        elif lattice_input.dim() == 2 and lattice_input.shape[1] == 9:
            lattice_flat = lattice_input
        elif lattice_input.dim() == 2 and lattice_input.shape[1] == 6:
            # Handle old format: Convert 6 lattice parameters to 9 matrix elements
            lattice_flat = self._convert_params_to_matrix(lattice_input)
        else:
            raise ValueError(
                f"Unexpected lattice_matrix shape: {lattice_input.shape}. Expected (B, 3, 3), (B, 9), or (B, 6)."
            )

        # MLP Forward Pass - first layer
        x = self.layer1(lattice_flat)

        # Apply conditioning *after* the first layer (inject into hidden state)
        # Note: _apply_conditioning projects and adds if context & projector exist
        x = self._apply_conditioning(x, condition_context)

        # Continue with MLP layers
        x = self.act1(x)
        x = self.layer2(x)
        x = self.act2(x)
        encoded = self.layer3(x)

        # Add sequence dimension if needed
        if encoded.dim() == 2:
            encoded = encoded.unsqueeze(1)  # (batch_size, 1, d_model)

        return encoded


class AtomTypeEncoder(EncoderBase):
    """Encodes atom type sequences, optionally conditioned."""

    def __init__(
        self,
        element_vocab_size: int,
        type_embed_dim: int,
        d_model: int,
        nhead: int = 8,
        num_layers: int = 2,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        pad_idx: int = 0,
        start_idx: int = -1,
        end_idx: int = -2,
        has_conditioning: bool = False,  # Flag
    ):
        super().__init__(d_output=d_model)
        # Create embedding layer
        self.effective_vocab_size = element_vocab_size + 3  # +3 for PAD, START, END
        self.type_embedding = nn.Embedding(
            self.effective_vocab_size, type_embed_dim, padding_idx=pad_idx
        )

        # Projection to d_model
        self.projection = nn.Linear(type_embed_dim, d_model)

        # Initialize condition projector if needed
        if has_conditioning:
            # Projects fused global context (d_model) to d_model
            self.condition_projector = nn.Linear(d_model, d_model)

        # Optional transformer layers for more sophisticated encoding
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        # Store index mapping info
        self.pad_idx = pad_idx
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.type_start_embed_idx = element_vocab_size + 1
        self.type_end_embed_idx = element_vocab_size + 2

    def _map_indices_for_embedding(self, indices: torch.Tensor) -> torch.Tensor:
        """Maps PAD, START, END indices to non-negative indices for embedding lookup."""
        indices = indices.long()
        mapped_indices = indices.clone()
        mapped_indices[indices == self.start_idx] = self.type_start_embed_idx
        mapped_indices[indices == self.end_idx] = self.type_end_embed_idx
        return mapped_indices

    def forward(
        self,
        atom_types: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            atom_types: (batch_size, seq_len) atom type indices
            mask: (batch_size, seq_len) boolean mask (True where padded)
            condition_context: Optional (batch_size, 1, d_model) fused global context
        Returns:
            context: (batch_size, seq_len, d_model)
        """
        # Map indices and get embeddings
        mapped_indices = self._map_indices_for_embedding(atom_types)
        mapped_indices = torch.clamp(mapped_indices, 0, self.effective_vocab_size - 1)

        # Get embeddings and project
        type_embeds = self.type_embedding(mapped_indices)
        proj_embeds = self.projection(type_embeds)

        # Apply conditioning *before* the transformer encoder
        # _apply_conditioning handles projection, expansion, and addition
        proj_embeds = self._apply_conditioning(proj_embeds, condition_context)

        # Apply transformer encoder if needed (with proper masking)
        key_padding_mask = ~mask if mask is not None else None
        output = self.encoder(proj_embeds, src_key_padding_mask=key_padding_mask)

        return output


class AtomCoordinateEncoder(EncoderBase):
    """Encodes atom coordinate sequences, optionally conditioned."""

    def __init__(
        self,
        d_model: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        nhead: int = 8,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        has_conditioning: bool = False,  # Flag
    ):
        super().__init__(d_output=d_model)
        # Process raw coordinates through MLP
        self.coords_mlp = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, d_model)
        )

        # Initialize condition projector if needed
        if has_conditioning:
            # Projects fused global context (d_model) to d_model
            self.condition_projector = nn.Linear(d_model, d_model)

        # Optional transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)

    def forward(
        self,
        coords: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            coords: (batch_size, seq_len, 3) fractional coordinates
            mask: (batch_size, seq_len) boolean mask (True where padded)
            condition_context: Optional (batch_size, 1, d_model) fused global context
        Returns:
            context: (batch_size, seq_len, d_model)
        """
        # Process coordinates
        processed_coords = self.coords_mlp(coords)

        # Apply conditioning *before* the transformer encoder
        # _apply_conditioning handles projection, expansion, and addition
        processed_coords = self._apply_conditioning(processed_coords, condition_context)

        # Apply transformer encoder if needed (with proper masking)
        key_padding_mask = ~mask if mask is not None else None
        output = self.encoder(processed_coords, src_key_padding_mask=key_padding_mask)

        return output


# Add more encoders as needed (WyckoffEncoder, ConstraintEncoder, etc.)
