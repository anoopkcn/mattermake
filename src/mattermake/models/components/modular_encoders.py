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
        x: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, element_vocab_size) composition tensor
            condition_context: Optional conditioning tensor (ignored here)
        Returns:
            context: (batch_size, 1, d_model)
        """
        comp_input = self.processor(x).unsqueeze(1)
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
        self, x: torch.Tensor, condition_context: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1) spacegroup tensor with values 1-230
            condition_context: Optional conditioning tensor
        Returns:
            context: (batch_size, 1, d_model)
        """
        # Ensure input is long type and handle squeezing
        if x.ndim > 1 and x.size(1) == 1:
            x = x.squeeze(1)

        device = x.device
        batch_size = x.size(0)

        # Map 1-230 to 0-229 for one-hot indexing
        sg_idx = x.long() - 1  # Map 1-230 -> 0-229

        # Create one-hot encoding
        one_hot = torch.zeros(batch_size, self.sg_vocab_size, device=device)
        one_hot.scatter_(1, sg_idx.unsqueeze(1), 1.0)

        # Apply projections with activation in between
        hidden = self.activation(self.projector1(one_hot))

        # Apply conditioning after first projection
        hidden = self._apply_conditioning(hidden, condition_context)

        output = self.projector2(hidden)

        return output.unsqueeze(1)  # (batch, 1, d_model)


class WyckoffEncoder(EncoderBase):
    """Encoder for Wyckoff positions."""
    
    def __init__(
        self,
        d_output: int,
        vocab_size: Optional[int] = None,
        embed_dim: int = 64,
        has_conditioning: bool = False,
        **kwargs
    ):
        super().__init__(d_output)
        
        # Import here to avoid circular imports
        from mattermake.data.components.wyckoff_interface import get_effective_wyckoff_vocab_size
        
        if vocab_size is None:
            vocab_size = get_effective_wyckoff_vocab_size()
        
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
    
        # Validate vocab size
        if vocab_size <= 1:
            raise ValueError(f"WyckoffEncoder vocab_size must be > 1, got {vocab_size}")
        if vocab_size > 50000:  # Sanity check for unreasonably large vocab
            print(f"Warning: WyckoffEncoder vocab_size is very large: {vocab_size}")
    
        # Embedding layer for Wyckoff positions
        self.wyckoff_embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        
        # Project to d_output
        self.output_projection = nn.Linear(embed_dim, d_output)
        
        # Conditioning projector (if needed)
        if has_conditioning:
            self.condition_projector = nn.Linear(d_output, d_output)
    
    def forward(
        self, 
        x: torch.Tensor, 
        condition_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L) tensor of Wyckoff indices
            condition_context: Optional conditioning context
            mask: Optional (B, L) boolean mask (True where padded)
        
        Returns:
            (B, L, d_output) tensor of Wyckoff embeddings
        """
        # Ensure indices are long type and within bounds
        x = x.long()
        
        # Map special tokens to valid vocabulary indices
        # PAD = 0 (stays 0)
        # START = -1 -> vocab_size - 2
        # END = -2 -> vocab_size - 1
        x_mapped = x.clone()
        x_mapped = torch.where(x == -1, self.vocab_size - 2, x_mapped)  # START token
        x_mapped = torch.where(x == -2, self.vocab_size - 1, x_mapped)  # END token
        
        # Check for any remaining out-of-bounds indices after mapping
        max_val = x_mapped.max().item() if x_mapped.numel() > 0 else 0
        min_val = x_mapped.min().item() if x_mapped.numel() > 0 else 0
        
        if max_val >= self.vocab_size or min_val < 0:
            print(f"Warning: Wyckoff indices still out of bounds after mapping. Min: {min_val}, Max: {max_val}, Vocab size: {self.vocab_size}")
            print(f"Clamping remaining indices to valid range [0, {self.vocab_size - 1}]")
        
        # Clamp any remaining out-of-bounds indices
        x_clamped = torch.clamp(x_mapped, 0, self.vocab_size - 1)
        
        # Get embeddings
        embeddings = self.wyckoff_embedding(x_clamped)  # (B, L, embed_dim)
        
        # Project to output dimension
        output = self.output_projection(embeddings)  # (B, L, d_output)
        
        # Apply conditioning if available
        output = self._apply_conditioning(output, condition_context)
        
        return output


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
        x: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: Lattice input in one of these formats:
                - (batch_size, 3, 3) matrix
                - (batch_size, 9) flattened matrix
                - (batch_size, 6) lattice parameters [a, b, c, alpha, beta, gamma]
            condition_context: Optional (batch_size, 1, d_model) tensor (e.g., from SG/Comp encoder)
        Returns:
            context: (batch_size, 1, d_model)
        """
        # Ensure input is flattened (batch_size, 9)
        if x.dim() == 3 and x.shape[1:] == (3, 3):
            lattice_flat = x.reshape(
                x.size(0), -1
            )  # Flatten to (batch_size, 9)
        elif x.dim() == 2 and x.shape[1] == 9:
            lattice_flat = x
        elif x.dim() == 2 and x.shape[1] == 6:
            # Handle old format: Convert 6 lattice parameters to 9 matrix elements
            lattice_flat = self._convert_params_to_matrix(x)
        else:
            raise ValueError(
                f"Unexpected lattice_matrix shape: {x.shape}. Expected (B, 3, 3), (B, 9), or (B, 6)."
            )

        # MLP Forward Pass - first layer
        hidden = self.layer1(lattice_flat)

        # Apply conditioning *after* the first layer (inject into hidden state)
        # Note: _apply_conditioning projects and adds if context & projector exist
        hidden = self._apply_conditioning(hidden, condition_context)

        # Continue with MLP layers
        hidden = self.act1(hidden)
        hidden = self.layer2(hidden)
        hidden = self.act2(hidden)
        encoded = self.layer3(hidden)

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
        x: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len) atom type indices
            condition_context: Optional (batch_size, 1, d_model) fused global context
            mask: (batch_size, seq_len) boolean mask (True where padded)
        Returns:
            context: (batch_size, seq_len, d_model)
        """
        # Map indices and get embeddings
        mapped_indices = self._map_indices_for_embedding(x)
        mapped_indices = torch.clamp(mapped_indices, 0, self.effective_vocab_size - 1)

        # Get embeddings and project
        type_embeds = self.type_embedding(mapped_indices)
        proj_embeds = self.projection(type_embeds)

        # Apply conditioning *before* the transformer encoder
        # _apply_conditioning handles projection, expansion, and addition
        proj_embeds = self._apply_conditioning(proj_embeds, condition_context)

        # Apply transformer encoder if needed (with proper masking)
        # Handle mask dtype conversion to avoid bool/float mismatch
        if mask is not None:
            # Convert boolean mask to match tensor dtype
            key_padding_mask = (~mask).to(dtype=torch.bool, device=proj_embeds.device)
        else:
            key_padding_mask = None
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
        x: torch.Tensor,
        condition_context: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, 3) fractional coordinates
            condition_context: Optional (batch_size, 1, d_model) fused global context
            mask: (batch_size, seq_len) boolean mask (True where padded)
        Returns:
            context: (batch_size, seq_len, d_model)
        """
        # Process coordinates
        processed_coords = self.coords_mlp(x)

        # Apply conditioning *before* the transformer encoder
        # _apply_conditioning handles projection, expansion, and addition
        processed_coords = self._apply_conditioning(processed_coords, condition_context)

        # Apply transformer encoder if needed (with proper masking)
        # Handle mask dtype conversion to avoid bool/float mismatch
        if mask is not None:
            # Convert boolean mask to match tensor dtype
            key_padding_mask = (~mask).to(dtype=torch.bool, device=processed_coords.device)
        else:
            key_padding_mask = None
        output = self.encoder(processed_coords, src_key_padding_mask=key_padding_mask)

        return output


# Add more encoders as needed (WyckoffEncoder, ConstraintEncoder, etc.)
