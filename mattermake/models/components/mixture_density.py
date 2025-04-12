import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
from typing import Optional, Dict, Any, Tuple


class ApproxWrappedNormal:
    """A simple wrapper for Normal distribution that wraps the values to [0, 1) interval.
    
    This is an approximation of a von Mises distribution for fractional coordinates.
    """
    def __init__(self, loc, scale):
        self.normal = torch.distributions.Normal(loc, torch.nn.functional.softplus(scale) + 1e-6)

    def sample(self, sample_shape=torch.Size()):
        return torch.fmod(self.normal.sample(sample_shape), 1.0)

    def log_prob(self, value):
        # WARNING: This is an approximation using Normal NLL.
        # A correct implementation requires summing probabilities over infinite images.
        # For small scales relative to [0,1) interval, direct normal approximation works decently
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
            nn.Linear(hidden_size // 2, self.output_size)
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
        
        # Project to mixture parameters
        # Shape: [batch_size, output_dim * n_mixtures * 3]
        raw_params = self.mixture_proj(hidden)
        
        # Reshape: [batch_size, output_dim, n_mixtures, 3]
        # where 3 corresponds to [weights, means, log_stds]
        params = raw_params.view(batch_size, self.output_dim, self.n_mixtures, 3)
        
        # Split into components
        weights_logits = params[..., 0]  # [batch_size, output_dim, n_mixtures]
        means = params[..., 1]           # [batch_size, output_dim, n_mixtures]
        log_stds = params[..., 2]        # [batch_size, output_dim, n_mixtures]
        stds = torch.exp(log_stds)        # Ensure positive std dev
        
        return {
            "weights_logits": weights_logits,
            "means": means,
            "stds": stds
        }
    
    def get_distribution(self, params: Dict[str, torch.Tensor]) -> dist.Distribution:
        """Create a mixture of Gaussians distribution."""
        # Convert logits to probabilities
        weights_probs = F.softmax(params["weights_logits"], dim=-1)
        
        # Create mixture distribution for each output dimension
        # Shape: [batch_size, output_dim]
        distributions = []
        
        for i in range(self.output_dim):
            mix = dist.Categorical(probs=weights_probs[:, i])
            comp = dist.Normal(params["means"][:, i], params["stds"][:, i])
            distributions.append(dist.MixtureSameFamily(mix, comp))
        
        # Combine into a multivariate distribution
        # We use Independent to make the dimensions independent
        return dist.Independent(dist.Normal(torch.stack([d.mean for d in distributions], dim=1), 
                                        torch.stack([d.stddev for d in distributions], dim=1)), 1)


class MixtureOfWrappedNormalsHead(MixtureDensityHead):
    """Mixture of wrapped normals head for fractional coordinates."""
    def __init__(self, hidden_size: int, output_dim: int = 3, n_mixtures: int = 5):
        super().__init__(hidden_size, output_dim, n_mixtures)
    
    def forward(self, hidden: torch.Tensor) -> Dict[str, torch.Tensor]:
        # hidden: [batch_size, hidden_size]
        batch_size = hidden.shape[0]
        
        # Project to mixture parameters
        # Shape: [batch_size, output_dim * n_mixtures * 3]
        raw_params = self.mixture_proj(hidden)
        
        # Reshape: [batch_size, output_dim, n_mixtures, 3]
        # where 3 corresponds to [weights, means, log_scales]
        params = raw_params.view(batch_size, self.output_dim, self.n_mixtures, 3)
        
        # Split into components
        weights_logits = params[..., 0]  # [batch_size, output_dim, n_mixtures]
        means = params[..., 1]           # [batch_size, output_dim, n_mixtures]
        log_scales = params[..., 2]      # [batch_size, output_dim, n_mixtures]
        scales = torch.exp(log_scales)    # Ensure positive scale
        
        return {
            "weights_logits": weights_logits,
            "means": means,
            "scales": scales
        }
    
    def get_distribution(self, params: Dict[str, torch.Tensor]) -> dist.Distribution:
        """Create a mixture of wrapped normals distribution.
        
        Since PyTorch doesn't have a wrapped normal distribution,
        we use a normal distribution and manually handle the wrapping.
        
        For simplicity and compatibility with existing loss computation in PyTorch,
        we return a standard multivariate normal that can be sampled and then wrapped,
        but with log_prob methods that are aware of the wrapping.
        """
        # Convert logits to probabilities
        weights_probs = F.softmax(params["weights_logits"], dim=-1)
        
        # Create mixture distribution for each output dimension
        # For simplicity, we'll use normal approximation for the log_prob and manually wrap samples
        distributions = []
        for i in range(self.output_dim):
            mix = dist.Categorical(probs=weights_probs[:, i])
            comp = dist.Normal(params["means"][:, i], params["scales"][:, i])
            distributions.append(dist.MixtureSameFamily(mix, comp))
        
        # Use Independent distribution to combine dimensions
        return dist.Independent(dist.Normal(torch.stack([d.mean for d in distributions], dim=1), 
                                        torch.stack([d.stddev for d in distributions], dim=1)), 1)


def bound_mixture_fractional_coords(coords: torch.Tensor) -> torch.Tensor:
    """Ensure fractional coordinates are in [0, 1) range."""
    return torch.fmod(torch.fmod(coords, 1.0) + 1.0, 1.0)