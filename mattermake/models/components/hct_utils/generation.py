import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any


class GenerationMixin:
    """Mixin for generation functionality in the Hierarchical Crystal Transformer"""

    def generate(
        self,
        input_ids: torch.Tensor,
        segment_ids: torch.Tensor,
        constraints: Dict[str, Any] = None,
        max_length: int = 512,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        repetition_penalty: float = 1.0,
        eos_token_id: Optional[int] = None,
        pad_token_id: int = 2,
        verbose: bool = False,
        use_kv_cache: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """Generate sequences using autoregressive generation with crystal constraints
        
        Args:
            input_ids: Starting token IDs of shape [batch_size, seq_length]
            segment_ids: Starting segment IDs of shape [batch_size, seq_length]
            constraints: Dictionary of constraints for generation
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: If set, sample from top k most likely tokens
            top_p: If set, sample from tokens with cumulative probability >= top_p
            repetition_penalty: Penalty for repeating tokens
            eos_token_id: Token ID that signals end of sequence
            pad_token_id: Token ID for padding
            verbose: Whether to print verbose output during generation
            use_kv_cache: Whether to use key-value caching
            
        Returns:
            Dictionary with generated sequences and related data
        """
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.clone()
        generated_segments = segment_ids.clone()

        # Track continuous predictions
        continuous_predictions = {
            "lattice_lengths": [],
            "lattice_angles": [],
            "fractional_coords": [],
        }

        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        # Initialize KV caches for all transformer layers if KV caching is enabled
        kv_caches = {} if use_kv_cache else None

        step = 0  # Add step counter
        while True:
            # Make sure all modules are active during generation
            if hasattr(self, "active_modules"):
                # Store original active modules to restore later
                original_active_modules = \
                    self.active_modules.copy() if self.active_modules else []
                # Enable all modules for generation
                self.active_modules = ["composition", "space_group", "lattice", "atoms"]

            # Forward pass
            outputs = self.forward(
                input_ids=generated_ids,
                segment_ids=generated_segments,
                attention_mask=attention_mask,
                use_causal_mask=True,
                kv_caches=kv_caches,
                use_kv_cache=use_kv_cache,
            )

            # Restore original active modules
            if hasattr(self, "active_modules"):
                self.active_modules = original_active_modules

            # Track continuous predictions
            if "lattice_lengths" in outputs and "lattice_angles" in outputs:
                continuous_predictions["lattice_lengths"].append(
                    outputs["lattice_lengths"]
                )
                continuous_predictions["lattice_angles"].append(
                    outputs["lattice_angles"]
                )

            if "fractional_coords" in outputs:
                continuous_predictions["fractional_coords"].append(
                    outputs["fractional_coords"]
                )

            # Get next token logits
            next_token_logits = outputs["logits"][:, -1, :]

            # Apply temperature
            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            # Apply repetition penalty
            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Apply Wyckoff constraints if applicable
            current_segments = generated_segments[:, -1].cpu().numpy()
            for i in range(batch_size):
                if unfinished_sequences[i] and current_segments[i] == self.segment_wyckoff:
                    if hasattr(self, "_apply_wyckoff_constraints"):
                        next_token_logits[i] = self._apply_wyckoff_constraints(
                            next_token_logits[i].unsqueeze(0)
                        ).squeeze(0)

            # Apply top-k filtering
            if top_k is not None:
                # Keep only top k tokens
                top_k_values, top_k_indices = torch.topk(
                    next_token_logits, k=min(top_k, next_token_logits.size(-1))
                )
                
                # Create new logits filled with negative infinity
                filtered_logits = torch.full_like(
                    next_token_logits, float("-inf")
                )
                
                # Fill in values for top-k tokens
                for i in range(batch_size):
                    filtered_logits[i, top_k_indices[i]] = top_k_values[i]
                
                next_token_logits = filtered_logits

            # Apply top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # Create a mask for tokens to remove
                sorted_mask = cumulative_probs > top_p
                
                # Shift mask to keep the first token above threshold
                sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
                sorted_mask[..., 0] = False

                # Fill in filtered logits
                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_mask[i]]
                    next_token_logits[i, indices_to_remove] = float("-inf")

            # Clean logits for sampling
            next_token_logits = torch.nan_to_num(
                next_token_logits, nan=-1e9, posinf=1e9, neginf=-1e9
            )

            # Convert logits to probabilities
            probs = F.softmax(next_token_logits, dim=-1)

            # Handle invalid probabilities
            invalid_probs = torch.isnan(probs) | torch.isinf(probs) | (probs < 0)
            if invalid_probs.any():
                probs = probs.clone()
                probs[invalid_probs] = 0.0
                # Renormalize
                row_sums = probs.sum(dim=-1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                probs = probs / row_sums

            # Add small epsilon for numerical stability
            probs = probs + 1e-10
            probs = probs / probs.sum(dim=-1, keepdim=True)

            # Sample next tokens
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            # Skip generation for finished sequences
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (~unfinished_sequences)

            # Update tracking for space groups and Wyckoff positions
            self._update_selected_space_groups(outputs, segment_ids == self.segment_space_group, outputs["hidden_states"]["final"])

            # Predict next segment IDs
            next_segments = self._predict_next_segment_id(
                generated_ids, generated_segments, next_tokens
            )

            # Append new tokens and segments
            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )
            generated_segments = torch.cat([generated_segments, next_segments], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )

            # --- DEBUG OUTPUT ---
            if verbose:
                idx_to_token = None
                if hasattr(self, 'tokenizer_config') and self.tokenizer_config and 'idx_to_token' in self.tokenizer_config:
                    idx_to_token = self.tokenizer_config['idx_to_token']
                print(f"[GENERATION][Step {step}] Batch size: {batch_size}")
                for i in range(batch_size):
                    token_id = int(generated_ids[i, -1].item())
                    segment_id = int(generated_segments[i, -1].item())
                    token_name = idx_to_token.get(token_id, str(token_id)) if idx_to_token else str(token_id)
                    print(f"  Sample {i}: Token ID: {token_id}, Segment ID: {segment_id}, Token: {token_name}")
            step += 1
            # --- END DEBUG OUTPUT ---

            # Check if generation is finished
            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (next_tokens != eos_token_id)

            # Stop if all sequences are finished or max length reached
            if unfinished_sequences.sum() == 0 or generated_ids.shape[1] >= max_length:
                break

        # Prepare results
        result = {
            "sequences": generated_ids, 
            "segment_ids": generated_segments
        }

        # Process continuous predictions
        if continuous_predictions["lattice_lengths"] and len(continuous_predictions["lattice_lengths"]) > 0:
            try:
                result["continuous_lattice_lengths"] = torch.cat(
                    continuous_predictions["lattice_lengths"], dim=0
                )
                result["continuous_lattice_angles"] = torch.cat(
                    continuous_predictions["lattice_angles"], dim=0
                )
                result["has_continuous_lattice"] = True
            except Exception:
                result["has_continuous_lattice"] = False
        else:
            result["has_continuous_lattice"] = False

        if continuous_predictions["fractional_coords"] and len(continuous_predictions["fractional_coords"]) > 0:
            try:
                result["continuous_fractional_coords"] = torch.cat(
                    continuous_predictions["fractional_coords"], dim=0
                )
                result["has_continuous_coords"] = True
            except Exception:
                result["has_continuous_coords"] = False
        else:
            result["has_continuous_coords"] = False

        return result