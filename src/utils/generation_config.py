from dataclasses import dataclass
from typing import Optional

@dataclass
class GenerationConfig:
    """Configuration for text generation with the GPT model."""

    max_new_tokens: int = 100
    """Maximum number of tokens to generate."""

    temperature: float = 0.3
    """Temperature for sampling. Lower values mean more focused/deterministic output."""

    top_k: Optional[int] = 30
    """If specified, only sample from the top k most likely tokens."""

    show_original: bool = True
    """Whether to show the original slices alongside generated ones."""

    num_samples: int = 10
    """Number of samples to generate."""
