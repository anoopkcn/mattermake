import torch
import torch.nn.functional as F
from typing import Dict, Optional, Any
from .constraint_handler import CrystalConstraintHandler


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
        use_kv_cache: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Generate sequences using autoregressive generation with space group and Wyckoff constraints."""
        self.eval()
        batch_size = input_ids.shape[0]
        device = input_ids.device

        attention_mask = torch.ones_like(input_ids)

        generated_ids = input_ids.clone()
        generated_segments = segment_ids.clone()

        if verbose:
            print(
                f"Starting generation with temperature={temperature}, top_k={top_k}, top_p={top_p}"
            )
            print("Using mixture density networks for predictions")
            print(f"Initial tokens shape: {input_ids.shape}")
            idx_to_token = (
                constraints.get("token_id_maps", {}).get("idx_to_token", {})
                if constraints
                else {}
            )

        # Always use continuous predictions
        continuous_predictions = {
            "lattice_lengths": [],
            "lattice_angles": [],
            "fractional_coords": [],
        }

        if verbose:
            print(f"Active modules: {self.active_modules}")

        unfinished_sequences = torch.ones(batch_size, dtype=torch.bool, device=device)

        constraint_handler = CrystalConstraintHandler(constraints)

        # Initialize KV caches for all transformer layers if KV caching is enabled
        kv_caches = None
        if use_kv_cache:
            # We'll initialize all possible cache keys
            kv_caches = {}
            # This is a simplification - in practice we'd enumerate all possible layer caches
            if verbose:
                print("Initialized KV caches for generation with KV-caching enabled")

        while True:
            # Make sure all modules are active during generation
            if hasattr(self, "active_modules"):
                # Store original active modules to restore later
                original_active_modules = (
                    self.active_modules.copy() if self.active_modules else []
                )
                # Ensure all necessary modules are active for continuous prediction
                self.active_modules = ["composition", "space_group", "lattice", "atoms"]

                if verbose and len(original_active_modules) != len(self.active_modules):
                    print(
                        f"Setting active_modules for generation: {self.active_modules}"
                    )

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
                if verbose:
                    print(f"Restored active_modules: {self.active_modules}")

            # Always collect continuous predictions
            if "lattice_lengths" in outputs and "lattice_angles" in outputs:
                if verbose:
                    print(
                        f"Found lattice predictions: lengths={outputs['lattice_lengths'].shape}, angles={outputs['lattice_angles'].shape}"
                    )
                continuous_predictions["lattice_lengths"].append(
                    outputs["lattice_lengths"]
                )
                continuous_predictions["lattice_angles"].append(
                    outputs["lattice_angles"]
                )
            elif verbose:
                print(
                    f"Missing lattice predictions in output keys: {list(outputs.keys())}"
                )

            if "fractional_coords" in outputs:
                if verbose:
                    print(
                        f"Found coordinate predictions: coords={outputs['fractional_coords'].shape}"
                    )
                continuous_predictions["fractional_coords"].append(
                    outputs["fractional_coords"]
                )
            elif verbose:
                print(
                    f"Missing coordinate predictions in output keys: {list(outputs.keys())}"
                )

            next_token_logits = outputs["logits"][:, -1, :]

            if temperature != 1.0:
                next_token_logits = next_token_logits / temperature

            if repetition_penalty != 1.0:
                for i in range(batch_size):
                    for previous_token in generated_ids[i]:
                        if previous_token != pad_token_id:
                            next_token_logits[i, previous_token] /= repetition_penalty

            # Apply constraints based on current segment type
            current_segments = generated_segments[:, -1].cpu().numpy()

            for i in range(batch_size):
                if unfinished_sequences[i]:
                    current_segment = current_segments[i]

                    # Apply Wyckoff position constraints
                    if (
                        current_segment == self.config.SEGMENT_WYCKOFF
                        and hasattr(self.config, "apply_wyckoff_constraints")
                        and self.config.apply_wyckoff_constraints
                    ):
                        wyckoff_mask = constraint_handler.get_wyckoff_mask(i)
                        if (
                            wyckoff_mask is not None
                            and wyckoff_mask.shape[0] <= next_token_logits.shape[1]
                        ):
                            next_token_logits[i, ~wyckoff_mask] = float("-inf")

            if top_k is not None:
                indices_to_remove = (
                    next_token_logits
                    < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                )
                next_token_logits[indices_to_remove] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(
                    next_token_logits, descending=True
                )
                cumulative_probs = torch.cumsum(
                    F.softmax(sorted_logits, dim=-1), dim=-1
                )

                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[
                    ..., :-1
                ].clone()
                sorted_indices_to_remove[..., 0] = 0

                for i in range(batch_size):
                    indices_to_remove = sorted_indices[i][sorted_indices_to_remove[i]]
                    next_token_logits[i, indices_to_remove] = float("-inf")

            next_token_logits = torch.nan_to_num(
                next_token_logits, nan=-1e9, posinf=1e9, neginf=-1e9
            )

            probs = F.softmax(next_token_logits, dim=-1)

            invalid_probs = torch.isnan(probs) | torch.isinf(probs) | (probs < 0)
            if invalid_probs.any():
                probs = probs.clone()
                probs[invalid_probs] = 0.0
                row_sums = probs.sum(dim=-1, keepdim=True)
                row_sums[row_sums == 0] = 1.0
                probs = probs / row_sums

            probs = probs + 1e-10
            probs = probs / probs.sum(dim=-1, keepdim=True)  # Renormalize

            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)

            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (
                ~unfinished_sequences
            )

            # Update constraint handler with generated tokens
            for i in range(batch_size):
                if unfinished_sequences[i]:
                    token_id = next_tokens[i].item()
                    current_segment = current_segments[i]

                    # Track space group tokens
                    if current_segment == self.config.SEGMENT_SPACE_GROUP:
                        constraint_handler.update_space_group(i, token_id)

                    # Track Wyckoff position tokens
                    elif current_segment == self.config.SEGMENT_WYCKOFF:
                        constraint_handler.update_wyckoff_position(i, token_id)

            if verbose and (
                generated_ids.shape[1] % 10 == 0 or generated_ids.shape[1] < 10
            ):
                # Log every 10 tokens or the first few tokens
                for i in range(min(batch_size, 2)):
                    token_id = next_tokens[i].item()
                    token_name = idx_to_token.get(str(token_id), f"<{token_id}>")
                    current_seg = segment_ids[i, -1].item()
                    segment_names = [
                        "SPECIAL",
                        "COMPOSITION",
                        "SPACE_GROUP",
                        "LATTICE",
                        "ELEMENT",
                        "WYCKOFF",
                        "COORDINATE",
                    ]
                    seg_name = (
                        segment_names[current_seg]
                        if current_seg < len(segment_names)
                        else f"SEG_{current_seg}"
                    )
                    print(
                        f"Seq {i}, Pos {generated_ids.shape[1]}: Generated token {token_id} ({token_name}) - Current segment: {seg_name}"
                    )

            next_segments = self._predict_next_segment_id(
                generated_ids, generated_segments, next_tokens
            )

            generated_ids = torch.cat(
                [generated_ids, next_tokens.unsqueeze(-1)], dim=-1
            )
            generated_segments = torch.cat([generated_segments, next_segments], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((batch_size, 1))], dim=-1
            )

            if eos_token_id is not None:
                unfinished_sequences = unfinished_sequences & (
                    next_tokens != eos_token_id
                )

            if unfinished_sequences.sum() == 0 or generated_ids.shape[1] >= max_length:
                break

        result = {"sequences": generated_ids, "segment_ids": generated_segments}

        # Track if we got continuous predictions
        has_continuous_lattice = False
        has_continuous_coords = False

        # Process continuous predictions
        if continuous_predictions:
            if verbose:
                print("Processing continuous predictions")
                print(
                    f"Lattice lengths collected: {len(continuous_predictions['lattice_lengths'])}"
                )
                print(
                    f"Lattice angles collected: {len(continuous_predictions['lattice_angles'])}"
                )
                print(
                    f"Fractional coords collected: {len(continuous_predictions['fractional_coords'])}"
                )

            if (
                continuous_predictions["lattice_lengths"]
                and len(continuous_predictions["lattice_lengths"]) > 0
            ):
                # Combine all predictions - for generation we want the last one for each sequence
                try:
                    # For debugging, examine the shapes before concatenation
                    if verbose:
                        for i, tensor in enumerate(
                            continuous_predictions["lattice_lengths"]
                        ):
                            print(f"Lattice length tensor {i} shape: {tensor.shape}")

                    result["continuous_lattice_lengths"] = torch.cat(
                        continuous_predictions["lattice_lengths"], dim=0
                    )
                    result["continuous_lattice_angles"] = torch.cat(
                        continuous_predictions["lattice_angles"], dim=0
                    )
                    has_continuous_lattice = True

                    if verbose:
                        print(
                            f"Successfully captured continuous lattice predictions with shape: {result['continuous_lattice_lengths'].shape}"
                        )
                        print(
                            f"Lattice lengths range: {result['continuous_lattice_lengths'].min().item():.3f} to {result['continuous_lattice_lengths'].max().item():.3f}"
                        )
                        print(
                            f"Lattice angles range: {result['continuous_lattice_angles'].min().item():.3f} to {result['continuous_lattice_angles'].max().item():.3f}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous lattice predictions: {e}"
                    )
                    print(
                        f"Tensor shapes: {[t.shape for t in continuous_predictions['lattice_lengths']]}"
                    )
                    print(
                        f"Tensor devices: {[t.device for t in continuous_predictions['lattice_lengths']]}"
                    )

            if (
                continuous_predictions["fractional_coords"]
                and len(continuous_predictions["fractional_coords"]) > 0
            ):
                try:
                    # For debugging, examine the shapes before concatenation
                    if verbose:
                        for i, tensor in enumerate(
                            continuous_predictions["fractional_coords"]
                        ):
                            print(f"Coordinate tensor {i} shape: {tensor.shape}")

                    result["continuous_fractional_coords"] = torch.cat(
                        continuous_predictions["fractional_coords"], dim=0
                    )
                    has_continuous_coords = True

                    if verbose:
                        print(
                            f"Successfully captured continuous coordinate predictions with shape: {result['continuous_fractional_coords'].shape}"
                        )
                        print(
                            f"Coordinate values range: {result['continuous_fractional_coords'].min().item():.3f} to {result['continuous_fractional_coords'].max().item():.3f}"
                        )
                except Exception as e:
                    print(
                        f"Warning: Failed to process continuous coordinate predictions: {e}"
                    )
                    print(
                        f"Tensor shapes: {[t.shape for t in continuous_predictions['fractional_coords']]}"
                    )
                    print(
                        f"Tensor devices: {[t.device for t in continuous_predictions['fractional_coords']]}"
                    )

        # Add flags to indicate whether continuous predictions were successfully obtained
        result["has_continuous_lattice"] = has_continuous_lattice
        result["has_continuous_coords"] = has_continuous_coords

        if verbose:
            print(
                f"Final results - has_continuous_lattice: {has_continuous_lattice}, has_continuous_coords: {has_continuous_coords}"
            )

        if verbose:
            # Print segment type distribution
            segment_counts = {}
            for i in range(generated_segments.size(1)):
                seg_id = generated_segments[0, i].item()  # Look at first sequence
                segment_counts[seg_id] = segment_counts.get(seg_id, 0) + 1

            segment_names = [
                "SPECIAL",
                "COMPOSITION",
                "SPACE_GROUP",
                "LATTICE",
                "ELEMENT",
                "WYCKOFF",
                "COORDINATE",
            ]
            print("\nGeneration summary:")
            print(f"Final sequence length: {generated_ids.shape[1]}")
            print("Segment distribution:")
            for seg_id, count in segment_counts.items():
                seg_name = (
                    segment_names[seg_id]
                    if seg_id < len(segment_names)
                    else f"SEG_{seg_id}"
                )
                print(f"  {seg_name}: {count} tokens")
            print(f"Used continuous lattice predictions: {has_continuous_lattice}")
            print(f"Used continuous coordinate predictions: {has_continuous_coords}")

        return result
