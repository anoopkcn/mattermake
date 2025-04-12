import torch
import random
from typing import Dict, Any


class CrystalAugmentation:
    """Data augmentation for crystal structure sequences"""

    def __init__(
        self,
        token_drop_prob: float = 0.05,
        token_shuffle_prob: float = 0.1,
        mask_prob: float = 0.15,
        mask_token_id: int = 2,  # Assuming PAD_TOKEN is used for masking
        swap_elements_prob: float = 0.05,
    ):
        """
        Initialize augmentation parameters

        Args:
            token_drop_prob: Probability of dropping a token
            token_shuffle_prob: Probability of shuffling token sequences
            mask_prob: Probability of masking tokens (for MLM-style training)
            mask_token_id: Token ID to use for masking
            swap_elements_prob: Probability of swapping elements
        """
        self.token_drop_prob = token_drop_prob
        self.token_shuffle_prob = token_shuffle_prob
        self.mask_prob = mask_prob
        self.mask_token_id = mask_token_id
        self.swap_elements_prob = swap_elements_prob

    def __call__(self, item: Dict[str, Any]) -> Dict[str, Any]:
        """Apply augmentations to the data item"""
        # Get key components
        tokens = item["input_ids"]
        segment_ids = item["segment_ids"]

        # Apply token masking (MLM-style)
        if self.mask_prob > 0 and random.random() < 0.5:
            tokens, masked_positions = self._apply_token_masking(tokens, segment_ids)
            item["input_ids"] = tokens
            item["masked_positions"] = masked_positions

        # Apply element swapping (changing composition)
        if self.swap_elements_prob > 0 and random.random() < 0.3:
            tokens, segment_ids = self._apply_element_swapping(tokens, segment_ids)
            item["input_ids"] = tokens
            item["segment_ids"] = segment_ids

        return item

    def _apply_token_masking(self, tokens, segment_ids):
        """Apply masked language modeling by randomly masking tokens"""
        tokens = tokens.clone()
        masked_positions = torch.zeros_like(tokens, dtype=torch.bool)

        # Only mask non-special tokens
        mask_candidates = (segment_ids > 0) & (tokens > 2)  # Skip special tokens

        # Randomly select tokens to mask
        mask_indices = torch.nonzero(mask_candidates).squeeze(-1)
        num_to_mask = max(1, int(len(mask_indices) * self.mask_prob))

        if len(mask_indices) > 0 and num_to_mask > 0:
            to_mask = mask_indices[torch.randperm(len(mask_indices))[:num_to_mask]]
            tokens[to_mask] = self.mask_token_id
            masked_positions[to_mask] = True

        return tokens, masked_positions

    def _apply_element_swapping(self, tokens, segment_ids):
        """Swap elements in composition to create alternative compositions"""
        tokens = tokens.clone()
        segment_ids = segment_ids.clone()

        # Find composition tokens
        composition_mask = segment_ids == 1  # SEGMENT_COMPOSITION
        composition_indices = torch.nonzero(composition_mask).squeeze(-1)

        if len(composition_indices) >= 2:
            # Randomly select two indices to swap
            idx1, idx2 = torch.randperm(len(composition_indices))[:2]
            pos1 = composition_indices[idx1]
            pos2 = composition_indices[idx2]

            # Swap tokens
            tokens[pos1], tokens[pos2] = tokens[pos2], tokens[pos1]

        return tokens, segment_ids
