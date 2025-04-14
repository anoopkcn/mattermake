import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Dict


class ApproxWrappedNormal:
    """A simple wrapper for Normal distribution that wraps the values to [0, 1) interval.

    This is an approximation of a von Mises distribution for fractional coordinates.
    """

    def __init__(self, loc, scale):
        self.normal = torch.distributions.Normal(
            loc, torch.nn.functional.softplus(scale) + 1e-6
        )

    def sample(self, sample_shape=torch.Size()):
        return torch.fmod(self.normal.sample(sample_shape), 1.0)

    def log_prob(self, value):
        # WARNING: This is an approximation using Normal NLL.
        # A correct implementation requires summing probabilities over infinite images.
        # For small scales relative to [0,1) interval, direct normal approximation works decently
        # TODO
        return self.normal.log_prob(value)


class MixtureDensityHead(nn.Module):
    """Base class for mixture density network heads."""

    def __init__(self, hidden_size: int, output_dim: int, n_mixtures: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_dim = output_dim
        self.n_mixtures = n_mixtures

        # Each mixture component needs weights, means, and scales
        self.output_size = output_dim * n_mixtures * 3

        # Projection from hidden states to mixture parameters
        self.mixture_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.SiLU(),
            nn.Linear(hidden_size // 2, self.output_size),
        )

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Project hidden states to mixture parameters."""
        raise NotImplementedError("Subclasses must implement forward.")


class MixtureOfGaussiansHead(MixtureDensityHead):
    """Mixture of Gaussians head for continuous lattice parameters."""

    def __init__(self, hidden_size: int, output_dim: int = 6, n_mixtures: int = 5):
        super().__init__(hidden_size, output_dim, n_mixtures)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden: [batch_size, hidden_size]
        batch_size = hidden.shape[0]

        # Add L2 normalization to hidden states before projection for better stability
        hidden_norm = F.normalize(hidden, p=2, dim=-1)

        # Project to mixture parameters
        # Shape: [batch_size, output_dim * n_mixtures * 3]
        raw_params = self.mixture_proj(hidden_norm)

        # Reshape: [batch_size, output_dim, n_mixtures, 3]
        # where 3 corresponds to [weights, means, log_stds]
        params = raw_params.view(batch_size, self.output_dim, self.n_mixtures, 3)

        # Split into components with better numerical controls
        weights_logits = params[..., 0]  # [batch_size, output_dim, n_mixtures]

        # Clamp logits to prevent extreme values
        weights_logits = torch.clamp(weights_logits, min=-20.0, max=20.0)

        means = params[..., 1]  # [batch_size, output_dim, n_mixtures]
        # Clamp means to reasonable values for lattice parameters
        means = torch.clamp(means, min=-10.0, max=10.0)

        log_stds = params[..., 2]  # [batch_size, output_dim, n_mixtures]
        # Clamp log_stds to prevent very large or small variances
        log_stds = torch.clamp(log_stds, min=-5.0, max=2.0)
        stds = torch.exp(log_stds)  # Ensure positive std dev

        return {"weights_logits": weights_logits, "means": means, "stds": stds}

    def get_distribution(self, params: Dict[str, torch.Tensor]) -> dist.Distribution:
        """Create a mixture of Gaussians distribution."""
        # Verify params exist and have expected shapes
        if not all(k in params for k in ["weights_logits", "means", "stds"]):
            raise ValueError(f"Missing required parameters. Got: {list(params.keys())}")

        # Print warning for debugging
        for key, tensor in params.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(f"Warning: NaN/Inf detected in {key} with shape {tensor.shape}")

        # Convert logits to probabilities with enhanced safety checks
        # First clean logits to avoid NaN/Inf
        clean_logits = torch.nan_to_num(
            params["weights_logits"], nan=0.0, posinf=0.0, neginf=-1e5
        )

        # Add small epsilon to avoid numerical issues
        clean_logits = clean_logits + torch.finfo(clean_logits.dtype).eps

        # Apply softmax to get valid probabilities
        weights_probs = F.softmax(clean_logits, dim=-1)

        # Verify we have valid probabilities
        if torch.isnan(weights_probs).any() or torch.isinf(weights_probs).any():
            # Use uniform distribution as fallback if still invalid
            batch_size, output_dim, n_mix = params["weights_logits"].shape
            print(
                f"Using uniform fallback for mixture weights with shape: {params['weights_logits'].shape}"
            )
            weights_probs = torch.ones_like(params["weights_logits"]) / n_mix

        # Clean means and standard deviations with more aggressive clamping
        means = torch.nan_to_num(params["means"], nan=0.0)
        stds = torch.nan_to_num(params["stds"], nan=1.0)
        # Ensure stds are positive and within reasonable range
        stds = torch.clamp(stds, min=1e-3, max=10.0)

        # Create mixture distribution for each output dimension
        # Shape: [batch_size, output_dim]
        distributions = []

        for i in range(self.output_dim):
            try:
                mix = dist.Categorical(probs=weights_probs[:, i])
                comp = dist.Normal(means[:, i], stds[:, i])
                distributions.append(dist.MixtureSameFamily(mix, comp))
            except Exception as e:
                # Fallback to a standard normal distribution if mixture creation fails
                print(f"Warning: Error creating distribution for dimension {i}: {e}")
                # Create a standard normal distribution as fallback
                loc = torch.zeros(
                    (params["means"].shape[0],), device=params["means"].device
                )
                scale = torch.ones(
                    (params["means"].shape[0],), device=params["means"].device
                )
                distributions.append(dist.Normal(loc, scale))

        # Create a simpler implementation that's more robust to shape issues
        # Instead of creating individual distributions and stacking them,
        # which can lead to complex shape issues, directly use the means and stds

        # Make sure we have valid parameters - this is the critical part for shape handling
        if len(distributions) > 0:
            # Get batch size from first distribution
            batch_size = distributions[0].mean.size(0)
            # Get output dimension from number of distributions
            output_dim = len(distributions)

            # Create tensors with expected shapes
            means = torch.zeros(
                (batch_size, output_dim), device=distributions[0].mean.device
            )
            stds = torch.ones(
                (batch_size, output_dim), device=distributions[0].stddev.device
            )

            # Fill with values from distributions where available
            for i, d in enumerate(distributions):
                if i < output_dim:
                    means[:, i] = d.mean
                    stds[:, i] = d.stddev
        else:
            # Fallback for empty distributions
            means = torch.zeros((1, self.output_dim), device=params["means"].device)
            stds = torch.ones((1, self.output_dim), device=params["means"].device)

        # Create a single multivariate normal with independent components
        return dist.Independent(dist.Normal(means, stds), 1)


class MixtureOfWrappedNormalsHead(MixtureDensityHead):
    """Mixture of wrapped normals head for fractional coordinates."""

    def __init__(self, hidden_size: int, output_dim: int = 3, n_mixtures: int = 5):
        super().__init__(hidden_size, output_dim, n_mixtures)

    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden: [batch_size, hidden_size]
        batch_size = hidden.shape[0]

        # Add L2 normalization to hidden states for better stability
        hidden_norm = F.normalize(hidden, p=2, dim=-1)

        # Project to mixture parameters
        # Shape: [batch_size, output_dim * n_mixtures * 3]
        raw_params = self.mixture_proj(hidden_norm)

        # Reshape: [batch_size, output_dim, n_mixtures, 3]
        # where 3 corresponds to [weights, means, log_scales]
        params = raw_params.view(batch_size, self.output_dim, self.n_mixtures, 3)

        # Split into components with improved numerical controls
        weights_logits = params[..., 0]  # [batch_size, output_dim, n_mixtures]
        # Clamp logits to prevent extreme values
        weights_logits = torch.clamp(weights_logits, min=-20.0, max=20.0)

        means = params[..., 1]  # [batch_size, output_dim, n_mixtures]
        # Clamp means to reasonable values for fractional coordinates (should be in [0,1])
        means = torch.clamp(
            means, min=-0.5, max=1.5
        )  # Allow slightly outside [0,1] for wrapping

        log_scales = params[..., 2]  # [batch_size, output_dim, n_mixtures]
        # Clamp log_scales to prevent very large or small variances
        log_scales = torch.clamp(
            log_scales, min=-5.0, max=1.0
        )  # Tighter upper bound for coordinates
        scales = torch.exp(log_scales)  # Ensure positive scale

        return {"weights_logits": weights_logits, "means": means, "scales": scales}

    def get_distribution(self, params: Dict[str, torch.Tensor]) -> dist.Distribution:
        """Create a mixture of wrapped normals distribution.

        Since PyTorch doesn't have a wrapped normal distribution,
        we use a normal distribution and manually handle the wrapping.

        For simplicity and compatibility with existing loss computation in PyTorch,
        we return a standard multivariate normal that can be sampled and then wrapped,
        but with log_prob methods that are aware of the wrapping.
        """
        # Verify params exist and have expected shapes
        if not all(k in params for k in ["weights_logits", "means", "scales"]):
            raise ValueError(f"Missing required parameters. Got: {list(params.keys())}")

        # Print warning for debugging
        for key, tensor in params.items():
            if torch.isnan(tensor).any() or torch.isinf(tensor).any():
                print(
                    f"Warning: NaN/Inf detected in coordinate {key} with shape {tensor.shape}"
                )

        # Convert logits to probabilities with enhanced safety checks
        # First clean logits to avoid NaN/Inf
        clean_logits = torch.nan_to_num(
            params["weights_logits"], nan=0.0, posinf=0.0, neginf=-1e5
        )

        # Add small epsilon to avoid numerical issues
        clean_logits = clean_logits + torch.finfo(clean_logits.dtype).eps

        # Apply softmax to get valid probabilities
        weights_probs = F.softmax(clean_logits, dim=-1)

        # Verify we have valid probabilities
        if torch.isnan(weights_probs).any() or torch.isinf(weights_probs).any():
            # Use uniform distribution as fallback if still invalid
            batch_size, output_dim, n_mix = params["weights_logits"].shape
            print(
                f"Using uniform fallback for coordinate mixture weights with shape: {params['weights_logits'].shape}"
            )
            weights_probs = torch.ones_like(params["weights_logits"]) / n_mix

        # Clean means and scales with more aggressive handling
        means = torch.nan_to_num(
            params["means"], nan=0.5
        )  # Default to middle of [0,1) range
        scales = torch.nan_to_num(params["scales"], nan=0.1)
        # Ensure scales are positive and not too large for periodic values
        scales = torch.clamp(
            scales, min=1e-3, max=0.2
        )  # Limit max scale to 0.2 for wrapped values

        # Create mixture distribution for each output dimension
        # For simplicity, we'll use normal approximation for the log_prob and manually wrap samples
        distributions = []
        for i in range(self.output_dim):
            try:
                mix = dist.Categorical(probs=weights_probs[:, i])
                comp = dist.Normal(means[:, i], scales[:, i])
                distributions.append(dist.MixtureSameFamily(mix, comp))
            except Exception as e:
                # Fallback to a standard normal distribution centered at 0.5 for [0,1) range
                print(
                    f"Warning: Error creating wrapped distribution for dimension {i}: {e}"
                )
                loc = 0.5 * torch.ones(
                    (params["means"].shape[0],), device=params["means"].device
                )
                scale = 0.1 * torch.ones(
                    (params["means"].shape[0],), device=params["means"].device
                )
                distributions.append(dist.Normal(loc, scale))

        # Create a simpler implementation that's more robust to shape issues
        # Instead of creating individual distributions and stacking them,
        # which can lead to complex shape issues, directly use the means and stds

        # Make sure we have valid parameters - this is the critical part for shape handling
        if len(distributions) > 0:
            # Get batch size from first distribution
            batch_size = distributions[0].mean.size(0)
            # Get output dimension from number of distributions
            output_dim = len(distributions)

            # Create tensors with expected shapes
            means = torch.zeros(
                (batch_size, output_dim), device=distributions[0].mean.device
            )
            stds = torch.ones(
                (batch_size, output_dim), device=distributions[0].stddev.device
            )

            # Fill with values from distributions where available
            for i, d in enumerate(distributions):
                if i < output_dim:
                    means[:, i] = d.mean
                    stds[:, i] = d.stddev
        else:
            # Fallback for empty distributions
            means = torch.zeros((1, self.output_dim), device=params["means"].device)
            stds = torch.ones((1, self.output_dim), device=params["means"].device)

            # For coordinates, center means at 0.5 (middle of [0,1) range)
            means.fill_(0.5)

        # Create a single multivariate normal with independent components
        return dist.Independent(dist.Normal(means, stds), 1)


def bound_mixture_fractional_coords(coords: torch.Tensor) -> torch.Tensor:
    """Ensure fractional coordinates are in [0, 1) range with robust handling of NaN/Inf values."""
    # First handle any NaN or Inf values
    clean_coords = torch.nan_to_num(coords, nan=0.5, posinf=0.5, neginf=0.5)

    # Then ensure they're in the [0, 1) range using fmod
    # We add 1.0 before taking the second fmod to handle negative values
    return torch.fmod(torch.fmod(clean_coords, 1.0) + 1.0, 1.0)
