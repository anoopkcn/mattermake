import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist


class SimplifiedMixtureOfGaussians(nn.Module):
    """Streamlined Mixture of Gaussians with consistent tensor shapes"""

    def __init__(self, hidden_size, output_dim, n_mixtures):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_mixtures = n_mixtures

        # Single projection network with standardized output shape
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, output_dim * (2 * n_mixtures + 1)),
        )

    def forward(self, hidden):
        """
        Forward pass with explicit shape handling

        Args:
            hidden: Tensor of shape [batch_size, hidden_size]

        Returns:
            Dictionary with standardized shapes:
                weights_logits: [batch_size, output_dim, n_mixtures]
                means: [batch_size, output_dim, n_mixtures]
                stds: [batch_size, output_dim, n_mixtures]
        """
        batch_size = hidden.shape[0]

        # Ensure hidden states are clean
        hidden = torch.nan_to_num(hidden, nan=0.0)
        hidden = F.normalize(hidden, p=2, dim=-1)

        # Project to all parameters at once
        projected = self.projection(hidden)

        # Reshape with explicit dimensions
        # Shape: [batch_size, output_dim, 2*n_mixtures+1]
        reshaped = projected.view(batch_size, self.output_dim, 2 * self.n_mixtures + 1)

        # Extract parameters with clear indexing
        logits = reshaped[:, :, : self.n_mixtures]
        means = reshaped[:, :, self.n_mixtures : 2 * self.n_mixtures]
        log_stds = reshaped[:, :, 2 * self.n_mixtures]

        # Apply constraints for stability
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        means = torch.clamp(means, min=-10.0, max=10.0)
        log_stds = torch.clamp(log_stds.unsqueeze(-1), min=-5.0, max=2.0)
        stds = torch.exp(log_stds).expand_as(means)

        return {
            "weights_logits": logits,  # [batch_size, output_dim, n_mixtures]
            "means": means,  # [batch_size, output_dim, n_mixtures]
            "stds": stds,  # [batch_size, output_dim, n_mixtures]
        }

    def compute_weighted_mean(self, params):
        """Compute weighted mean with explicit shape handling"""
        # Clean parameters
        clean_logits = torch.nan_to_num(params["weights_logits"], nan=0.0)
        clean_means = torch.nan_to_num(params["means"], nan=0.0)

        # Apply softmax to get mixture weights
        weights = F.softmax(clean_logits, dim=-1)

        # Explicit weighted mean calculation
        # weights: [batch_size, output_dim, n_mixtures]
        # means: [batch_size, output_dim, n_mixtures]
        weighted_mean = torch.sum(weights * clean_means, dim=-1)
        # Result: [batch_size, output_dim]

        return weighted_mean

    def get_distribution(self, params):
        """Create mixture distribution for log-likelihood calculation"""
        # Clean parameters for numerical stability
        clean_means = torch.nan_to_num(params["means"], nan=0.0)
        clean_stds = torch.nan_to_num(params["stds"], nan=1.0)
        clean_stds = torch.clamp(clean_stds, min=1e-3, max=10.0)

        # Create independent normal distribution
        return dist.Independent(dist.Normal(clean_means, clean_stds), 1)


class SimplifiedMixtureOfWrappedNormals(nn.Module):
    """Streamlined Mixture of Wrapped Normals for fractional coordinates"""

    def __init__(self, hidden_size, output_dim, n_mixtures):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_mixtures = n_mixtures

        # Single projection network with standardized output shape
        self.projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, output_dim * (2 * n_mixtures + 1)),
        )

    def forward(self, hidden):
        """
        Forward pass with explicit shape handling

        Args:
            hidden: Tensor of shape [batch_size, hidden_size]

        Returns:
            Dictionary with standardized shapes:
                weights_logits: [batch_size, output_dim, n_mixtures]
                means: [batch_size, output_dim, n_mixtures]
                scales: [batch_size, output_dim, n_mixtures]
        """
        batch_size = hidden.shape[0]

        # Ensure hidden states are clean
        hidden = torch.nan_to_num(hidden, nan=0.0)
        hidden = F.normalize(hidden, p=2, dim=-1)

        # Project to all parameters at once
        projected = self.projection(hidden)

        # Reshape with explicit dimensions
        # Shape: [batch_size, output_dim, 2*n_mixtures+1]
        reshaped = projected.view(batch_size, self.output_dim, 2 * self.n_mixtures + 1)

        # Extract parameters with clear indexing
        logits = reshaped[:, :, : self.n_mixtures]
        means = reshaped[:, :, self.n_mixtures : 2 * self.n_mixtures]
        log_scales = reshaped[:, :, 2 * self.n_mixtures]

        # Apply constraints for stability
        logits = torch.clamp(logits, min=-20.0, max=20.0)
        means = torch.clamp(
            means, min=-0.5, max=1.5
        )  # Allow slightly outside [0,1] for wrapping
        log_scales = torch.clamp(log_scales.unsqueeze(-1), min=-5.0, max=1.0)
        scales = torch.exp(log_scales).expand_as(means)

        return {
            "weights_logits": logits,  # [batch_size, output_dim, n_mixtures]
            "means": means,  # [batch_size, output_dim, n_mixtures]
            "scales": scales,  # [batch_size, output_dim, n_mixtures]
        }

    def compute_weighted_mean(self, params):
        """Compute weighted mean with explicit shape handling"""
        # Clean parameters
        clean_logits = torch.nan_to_num(params["weights_logits"], nan=0.0)
        clean_means = torch.nan_to_num(params["means"], nan=0.5)

        # Apply softmax to get mixture weights
        weights = F.softmax(clean_logits, dim=-1)

        # Explicit weighted mean calculation
        # weights: [batch_size, output_dim, n_mixtures]
        # means: [batch_size, output_dim, n_mixtures]
        weighted_mean = torch.sum(weights * clean_means, dim=-1)
        # Result: [batch_size, output_dim]

        # Ensure result is in [0, 1) range
        wrapped_mean = torch.fmod(torch.fmod(weighted_mean, 1.0) + 1.0, 1.0)

        return wrapped_mean

    def get_distribution(self, params):
        """Create wrapped normal distribution for log-likelihood calculation"""
        # Clean parameters for numerical stability
        clean_means = torch.nan_to_num(params["means"], nan=0.5)
        clean_scales = torch.nan_to_num(params["scales"], nan=0.1)
        clean_scales = torch.clamp(clean_scales, min=1e-3, max=0.2)

        # Create independent normal distribution as approximation
        return dist.Independent(dist.Normal(clean_means, clean_scales), 1)


def bound_lattice_lengths(raw_lengths):
    """Convert raw network outputs to realistic lattice length parameters (a, b, c)

    Args:
        raw_lengths: Raw values from the network [batch_size, 3]

    Returns:
        Bounded lattice length values between 2 and 50 Å [batch_size, 3]
    """
    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_lengths, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid and scale to desired range
    result = 2.0 + 48.0 * torch.sigmoid(clean_values)

    # Final safety check
    result = torch.clamp(result, min=2.0, max=50.0)
    return result


def bound_lattice_angles(raw_angles):
    """Convert raw network outputs to realistic lattice angle parameters (alpha, beta, gamma)

    Args:
        raw_angles: Raw values from the network [batch_size, 3]

    Returns:
        Bounded lattice angle values between 30 and 150 degrees [batch_size, 3]
    """
    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_angles, nan=0.0, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid and scale to desired range
    result = 30.0 + 120.0 * torch.sigmoid(clean_values)

    # Final safety check
    result = torch.clamp(result, min=30.0, max=150.0)
    return result


def bound_fractional_coords(raw_coords):
    """Convert raw network outputs to fractional coordinates (x, y, z)

    Args:
        raw_coords: Raw values from the network [batch_size, 3]

    Returns:
        Bounded fractional coordinates between 0 and 1 [batch_size, 3]
    """
    # Check for NaN/Inf first
    clean_values = torch.nan_to_num(raw_coords, nan=0.5, posinf=10.0, neginf=-10.0)

    # Clamp to reasonable range before sigmoid to avoid numerical instability
    clean_values = torch.clamp(clean_values, min=-10.0, max=10.0)

    # Apply sigmoid to get values in [0,1] range
    result = torch.sigmoid(clean_values)

    # Ensure coordinates are exactly in [0,1) range for periodic boundary conditions
    result = torch.fmod(result, 1.0)
    return result
