import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class LossCalculationMixin:
    """Mixin for loss calculation in the Hierarchical Crystal Transformer"""

    def calculate_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
        batch_size: int,
    ) -> Dict[str, torch.Tensor]:
        """Calculate all losses for the model"""
        try:
            self._calculate_token_loss(outputs, labels, segment_ids, batch_size)
            self._calculate_space_group_loss(
                outputs, labels, segment_ids, hidden_states
            )
            self._calculate_element_loss(outputs, labels, segment_ids, hidden_states)
            self._calculate_wyckoff_loss(outputs, labels, segment_ids, hidden_states)
            self._calculate_continuous_losses(
                outputs, labels, segment_ids, hidden_states
            )
        except Exception as e:
            print(f"Warning: Error in loss calculation: {e}")
            if "loss" not in outputs:
                outputs["loss"] = torch.tensor(
                    0.0, device=hidden_states.device, requires_grad=True
                )

        return outputs

    def _calculate_token_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        batch_size: int,
    ) -> None:
        """Calculate token prediction loss with segment-based weighting"""
        try:
            # Prepare the labels and logits
            shift_logits = outputs["logits"][:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            shift_segments = segment_ids[:, 1:].contiguous()

            # Validate labels
            valid_labels = (shift_labels >= 0) & (
                shift_labels < self.config.vocab_size
            ) | (shift_labels == -100)
            if not torch.all(valid_labels):
                shift_labels = torch.where(
                    valid_labels,
                    shift_labels,
                    torch.tensor(-100, device=shift_labels.device),
                )

            # Calculate loss with ignore index
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
            token_losses = loss_fct(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

            # Reshape to batch size and sequence length
            token_losses = token_losses.view(batch_size, -1)

            # Apply segment-based weighting
            weighted_losses = token_losses.clone()

            # Apply composition weight
            comp_indices = (shift_segments == self.config.SEGMENT_COMPOSITION).float()
            if self.config.composition_loss_weight != 1.0 and torch.any(comp_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.composition_loss_weight - 1.0) * comp_indices
                )

            # Apply space group weight
            sg_indices = (shift_segments == self.config.SEGMENT_SPACE_GROUP).float()
            if self.config.space_group_loss_weight != 1.0 and torch.any(sg_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.space_group_loss_weight - 1.0) * sg_indices
                )

            # Apply lattice weight
            lattice_indices = (shift_segments == self.config.SEGMENT_LATTICE).float()
            if self.config.lattice_loss_weight != 1.0 and torch.any(lattice_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.lattice_loss_weight - 1.0) * lattice_indices
                )

            # Apply atom weight
            atom_indices = (
                (shift_segments == self.config.SEGMENT_ELEMENT)
                | (shift_segments == self.config.SEGMENT_WYCKOFF)
                | (shift_segments == self.config.SEGMENT_COORDINATE)
            ).float()
            if self.config.atom_loss_weight != 1.0 and torch.any(atom_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.atom_loss_weight - 1.0) * atom_indices
                )

            # Calculate mean loss
            if weighted_losses.numel() > 0:
                loss = weighted_losses.sum() / (
                    weighted_losses.size(0) * weighted_losses.size(1)
                )
                outputs["loss"] = loss
            else:
                outputs["loss"] = torch.tensor(
                    0.0, device=weighted_losses.device, requires_grad=True
                )
        except Exception as e:
            print(f"Warning: Error in token loss calculation: {e}")
            outputs["loss"] = torch.tensor(
                0.0, device=labels.device, requires_grad=True
            )

    def _calculate_space_group_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate space group prediction loss"""
        # Only calculate if in active modules
        if "space_group" not in getattr(self, "active_modules", ["space_group"]):
            return

        try:
            # Identify space group tokens
            sg_mask = segment_ids == self.config.SEGMENT_SPACE_GROUP
            if not torch.any(sg_mask):
                return

            # Get hidden states for space group tokens
            sg_hidden = hidden_states[sg_mask]
            if sg_hidden.size(0) == 0:
                return

            # Calculate space group logits
            sg_logits = self.space_group_head(sg_hidden)
            outputs["space_group_logits"] = sg_logits

            # Get space group labels
            sg_labels_mask = segment_ids == self.config.SEGMENT_SPACE_GROUP
            if not torch.any(sg_labels_mask):
                return

            space_group_labels = labels[sg_labels_mask]

            # Validate labels
            valid_labels = (space_group_labels >= 0) & (
                space_group_labels < self.space_group_head.out_features
            ) | (space_group_labels == -100)

            if not torch.all(valid_labels):
                space_group_labels = torch.where(
                    valid_labels,
                    space_group_labels,
                    torch.tensor(-100, device=space_group_labels.device),
                )

            # Check sizes match
            if sg_logits.size(0) != space_group_labels.size(0):
                print(
                    f"Warning: Space group logits and labels size mismatch: {sg_logits.size(0)} vs {space_group_labels.size(0)}"
                )
                return

            # Calculate loss with label smoothing
            try:
                # Check for NaN/Inf in logits before log_softmax
                if torch.isnan(sg_logits).any() or torch.isinf(sg_logits).any():
                    print("Warning: NaN/Inf detected in space group logits")
                    # Clean logits to avoid NaN/Inf propagation
                    sg_logits = torch.nan_to_num(
                        sg_logits, nan=0.0, posinf=0.0, neginf=-1e5
                    )

                # Add small epsilon for numerical stability
                sg_logits = sg_logits + torch.finfo(sg_logits.dtype).eps * torch.sign(
                    sg_logits
                )

                log_probs = F.log_softmax(sg_logits, dim=-1)

                label_smoothing = 0.1
                n_classes = log_probs.size(-1)  # Number of space groups (230)

                # Create one-hot encoding of labels
                mask = space_group_labels != -100
                label_indices = space_group_labels.masked_fill(~mask, 0)
                one_hot = torch.zeros_like(log_probs).scatter_(
                    1, label_indices.unsqueeze(1), 1.0
                )

                # Apply label smoothing: (1-α)×one_hot + α/K
                smoothed_targets = (
                    one_hot * (1.0 - label_smoothing) + label_smoothing / n_classes
                )

                loss_per_sample = -(smoothed_targets * log_probs).sum(dim=1)

                if mask.any():
                    space_group_loss = loss_per_sample[mask].mean()
                    if torch.isnan(space_group_loss) or torch.isinf(space_group_loss):
                        print(
                            "Warning: NaN/Inf detected in space group loss computation!"
                        )
                        space_group_loss = torch.tensor(0.0, device=log_probs.device)
                else:
                    space_group_loss = torch.tensor(0.0, device=log_probs.device)

                outputs["space_group_loss"] = space_group_loss
            except Exception as e:
                print(f"Warning: Failed to calculate space group loss: {e}")
        except Exception as e:
            print(f"Warning: Error in space group loss processing: {e}")

    def _calculate_element_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate element prediction loss"""
        # Only calculate if atoms are in active modules
        if "atoms" not in getattr(self, "active_modules", ["atoms"]):
            return

        try:
            # Identify element tokens
            element_mask = segment_ids == self.config.SEGMENT_ELEMENT
            if not torch.any(element_mask):
                return

            # Get hidden states for element tokens
            element_hidden = hidden_states[element_mask]
            if element_hidden.size(0) == 0:
                return

            # Calculate element logits
            element_logits = self.element_head(element_hidden)
            outputs["element_logits"] = element_logits

            # Get element labels
            element_labels = labels[element_mask]

            # Validate labels
            valid_labels = (element_labels >= 0) & (
                element_labels < self.element_head.out_features
            ) | (element_labels == -100)

            if not torch.all(valid_labels):
                element_labels = torch.where(
                    valid_labels,
                    element_labels,
                    torch.tensor(-100, device=element_labels.device),
                )

            # Check sizes match
            if element_logits.size(0) != element_labels.size(0):
                print(
                    f"Warning: Element logits and labels size mismatch: {element_logits.size(0)} vs {element_labels.size(0)}"
                )
                return

            # Calculate standard cross-entropy loss
            try:
                element_loss = F.cross_entropy(
                    element_logits, element_labels, ignore_index=-100
                )
                outputs["element_loss"] = element_loss
            except Exception as e:
                print(f"Warning: Failed to calculate element loss: {e}")
        except Exception as e:
            print(f"Warning: Error in element loss processing: {e}")

    def _calculate_wyckoff_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate Wyckoff position prediction loss"""
        # Only calculate if atoms are in active modules
        if "atoms" not in getattr(self, "active_modules", ["atoms"]):
            return

        try:
            # Identify Wyckoff tokens
            wyckoff_mask = segment_ids == self.config.SEGMENT_WYCKOFF
            if not torch.any(wyckoff_mask):
                return

            # Get hidden states for Wyckoff tokens
            wyckoff_hidden = hidden_states[wyckoff_mask]
            if wyckoff_hidden.size(0) == 0:
                return

            # Get batch indices for each Wyckoff position
            batch_indices = torch.where(wyckoff_mask)[0]

            # Prepare outputs for wyckoff logits
            all_wyckoff_logits = []

            # Process each Wyckoff position
            for i, batch_idx in enumerate(batch_indices):
                batch_idx = batch_idx.item()

                # Determine whether to use combined or regular Wyckoff head
                use_combined = (
                    hasattr(self.config, "use_combined_wyckoff_tokens")
                    and self.config.use_combined_wyckoff_tokens
                    and self.wyckoff_mult_head is not None
                )

                if use_combined:
                    # Use combined Wyckoff-multiplicity head
                    wyckoff_logits_i = self.wyckoff_mult_head(wyckoff_hidden[i : i + 1])
                else:
                    # Original fixed Wyckoff head
                    wyckoff_logits_i = self.wyckoff_head(wyckoff_hidden[i : i + 1])

                # Apply space group constraints if applicable
                if (
                    hasattr(self.config, "apply_wyckoff_constraints")
                    and self.config.apply_wyckoff_constraints
                    and hasattr(self, "_selected_space_groups")
                    and batch_idx in self._selected_space_groups
                    and self._sg_wyckoff_mapping is not None
                ):
                    space_group = self._selected_space_groups[batch_idx]

                    # Create appropriate mask based on whether we're using combined tokens
                    if use_combined:
                        # Create a basic mask for the current space group
                        # Get allowed Wyckoff positions
                        allowed_wyckoff = (
                            self._sg_wyckoff_mapping.get_allowed_wyckoff_positions(
                                space_group
                            )
                        )

                        # Create mask for combined tokens (only allows tokens for current space group)
                        valid_mask = torch.zeros(
                            self.config.num_wyckoff_mult_tokens,
                            device=wyckoff_hidden.device,
                            dtype=torch.bool,
                        )

                        # Set valid positions based on the tokens for this space group
                        offset = 3000 + (
                            space_group * 100
                        )  # Combined token range starts at 3000
                        for letter_idx, letter in enumerate(allowed_wyckoff):
                            idx = ord(letter) - ord("a")
                            if 0 <= idx < 26:
                                token_id = offset + idx
                                if token_id < valid_mask.size(0):
                                    valid_mask[token_id] = True
                    else:
                        # Original approach for individual Wyckoff tokens
                        valid_mask = torch.tensor(
                            self._sg_wyckoff_mapping.create_wyckoff_mask(space_group),
                            device=wyckoff_hidden.device,
                            dtype=torch.bool,
                        )

                    # Apply the mask by setting invalid positions to -inf
                    if (
                        valid_mask is not None
                        and valid_mask.shape[0] <= wyckoff_logits_i.shape[1]
                    ):
                        wyckoff_logits_i[:, ~valid_mask] = float("-inf")

                all_wyckoff_logits.append(wyckoff_logits_i)

            # Combine all Wyckoff logits
            if all_wyckoff_logits:
                wyckoff_logits = torch.cat(all_wyckoff_logits, dim=0)
                outputs["wyckoff_logits"] = wyckoff_logits

                # Get Wyckoff labels
                wyckoff_labels = labels[wyckoff_mask]

                # Determine which head was used (combined or regular)
                is_using_combined = (
                    hasattr(self.config, "use_combined_wyckoff_tokens")
                    and self.config.use_combined_wyckoff_tokens
                    and self.wyckoff_mult_head is not None
                    and "wyckoff_mult_head" in str(wyckoff_logits.shape)
                )

                # Get the appropriate output feature size based on which head was used
                if (
                    is_using_combined
                    and hasattr(self, "wyckoff_mult_head")
                    and self.wyckoff_mult_head is not None
                ):
                    out_features = self.wyckoff_mult_head.out_features
                else:
                    out_features = self.wyckoff_head.out_features

                # Validate labels against the correct output size
                valid_labels = (wyckoff_labels >= 0) & (
                    wyckoff_labels < out_features
                ) | (wyckoff_labels == -100)

                if not torch.all(valid_labels):
                    # Handle case where labels might be for the wrong head type
                    if is_using_combined and torch.any(wyckoff_labels >= 3000):
                        # Labels appear to be for combined tokens, keep them
                        pass
                    elif not is_using_combined and torch.any(wyckoff_labels < 3000):
                        # Labels appear to be for regular tokens, keep them
                        pass
                    else:
                        # Labels need to be replaced with ignore_index
                        wyckoff_labels = torch.where(
                            valid_labels,
                            wyckoff_labels,
                            torch.tensor(-100, device=wyckoff_labels.device),
                        )

                # Check sizes match
                if wyckoff_logits.size(0) == wyckoff_labels.size(0):
                    try:
                        wyckoff_loss = F.cross_entropy(
                            wyckoff_logits, wyckoff_labels, ignore_index=-100
                        )
                        outputs["wyckoff_loss"] = wyckoff_loss
                    except Exception as e:
                        print(f"Warning: Failed to calculate wyckoff loss: {e}")
                else:
                    print(
                        f"Warning: Wyckoff logits and labels size mismatch: {wyckoff_logits.size(0)} vs {wyckoff_labels.size(0)}"
                    )
        except Exception as e:
            print(f"Warning: Error in Wyckoff loss processing: {e}")

    def _calculate_continuous_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate losses for continuous values (lattice parameters, coordinates)"""
        # Lattice parameter regression loss
        self._calculate_lattice_regression_loss(outputs, segment_ids, hidden_states)

        # Coordinate regression loss
        self._calculate_coordinate_regression_loss(outputs, segment_ids, hidden_states)

    def _calculate_lattice_regression_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate lattice parameter regression loss using mixture density networks"""
        # Only calculate if lattice module is active and we're training
        if not self.training or "lattice" not in getattr(
            self, "active_modules", ["lattice"]
        ):
            return

        # Check if we have ground truth values
        if (
            not hasattr(self, "_lattice_ground_truth")
            or self._lattice_ground_truth is None
        ):
            return

        try:
            # Identify lattice tokens
            lattice_mask = segment_ids == self.config.SEGMENT_LATTICE
            if not torch.any(lattice_mask):
                return

            # Get hidden states for lattice tokens
            lattice_hidden = hidden_states[lattice_mask]
            if lattice_hidden.size(0) == 0:
                return

            # Check for NaN/Inf in lattice_hidden before passing to MoG head
            if torch.isnan(lattice_hidden).any() or torch.isinf(lattice_hidden).any():
                print("Warning: NaN/Inf detected in lattice_hidden")
                lattice_hidden = torch.nan_to_num(
                    lattice_hidden, nan=0.0, posinf=0.0, neginf=-1e5
                )

            # Try to get lattice parameters with error handling
            try:
                lattice_mog_params = self.lattice_mog_head(lattice_hidden)
                outputs["lattice_mog_params"] = lattice_mog_params

                # Get ground truth values
                gt_lengths = self._lattice_ground_truth["lengths"]
                gt_angles = self._lattice_ground_truth["angles"]

                # Shape: [batch_size, 6]
                gt_lattice_params = torch.cat([gt_lengths, gt_angles], dim=-1)

                try:
                    mog_dist = self.lattice_mog_head.get_distribution(
                        outputs["lattice_mog_params"]
                    )
                    log_prob = mog_dist.log_prob(gt_lattice_params)
                    lattice_regression_loss = -log_prob.mean()
                    outputs["lattice_regression_loss"] = lattice_regression_loss
                except Exception as e:
                    print(f"Warning: Failed to calculate lattice MoG loss: {e}")
            except Exception as e:
                print(f"Warning: Failed to calculate lattice parameters: {e}")
        except Exception as e:
            print(f"Warning: Error in lattice regression loss processing: {e}")

    def _calculate_coordinate_regression_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate coordinate regression loss using mixture density networks"""
        # Only calculate if atoms module is active and we're training
        if not self.training or "atoms" not in getattr(
            self, "active_modules", ["atoms"]
        ):
            return

        # Check if we have ground truth values
        if (
            not hasattr(self, "_coordinate_ground_truth")
            or self._coordinate_ground_truth is None
        ):
            return

        try:
            # Identify coordinate tokens
            coordinate_mask = segment_ids == self.config.SEGMENT_COORDINATE
            if not torch.any(coordinate_mask):
                return

            # Get hidden states for coordinate tokens
            coordinate_hidden = hidden_states[coordinate_mask]
            if coordinate_hidden.size(0) == 0:
                return

            # More robust approach to coordinate handling
            # First check for NaN/Inf in coordinate hidden states
            if (
                torch.isnan(coordinate_hidden).any()
                or torch.isinf(coordinate_hidden).any()
            ):
                print("Warning: NaN/Inf detected in coordinate hidden states")
                coordinate_hidden = torch.nan_to_num(
                    coordinate_hidden, nan=0.0, posinf=0.0, neginf=-1e5
                )

            # Calculate number of atoms based on coordinate tokens
            num_atoms = coordinate_hidden.size(0) // 3

            # Safety check: ensure we have complete atoms (multiples of 3 tokens)
            if coordinate_hidden.size(0) % 3 != 0:
                num_atoms = coordinate_hidden.size(0) // 3
                print(
                    f"Warning: Coordinate tokens ({coordinate_hidden.size(0)}) not a multiple of 3. Truncating to {num_atoms * 3} tokens."
                )

            # Check num_atoms > 0 and that coordinate_hidden has enough elements
            if num_atoms > 0 and coordinate_hidden.size(0) >= num_atoms * 3:
                # Only proceed if we have enough tokens for at least one atom
                if coordinate_hidden.size(0) >= 3:
                    try:
                        # Use strided indexing to get first token of each triplet
                        slice_end = num_atoms * 3
                        if slice_end <= coordinate_hidden.size(0):
                            # Safer slicing with explicit shape check
                            coord_input = coordinate_hidden[:slice_end:3]

                            if coord_input.size(0) > 0:
                                # Always use MoVM for coordinate predictions
                                try:
                                    coord_movm_params = self.fractional_coord_movm_head(
                                        coord_input
                                    )
                                    outputs["coord_movm_params"] = coord_movm_params

                                    # Get ground truth coordinates
                                    gt_coords = self._coordinate_ground_truth[
                                        "fractional_coords"
                                    ]

                                    # Always use negative log-likelihood loss with mixture of wrapped normals
                                    try:
                                        # Get distribution from parameters
                                        movm_dist = self.fractional_coord_movm_head.get_distribution(
                                            outputs["coord_movm_params"]
                                        )

                                        # Calculate negative log-likelihood
                                        log_prob = movm_dist.log_prob(gt_coords)
                                        coord_regression_loss = -log_prob.mean()
                                        outputs["coord_regression_loss"] = (
                                            coord_regression_loss
                                        )
                                    except Exception as e:
                                        print(
                                            f"Warning: Failed to calculate coordinate MoVM loss: {e}"
                                        )
                                except Exception as e:
                                    print(
                                        f"Warning: Failed to get coordinate distribution: {e}"
                                    )
                    except Exception as e:
                        print(f"Warning: Error in coordinate processing: {e}")
        except Exception as e:
            print(f"Warning: Error in coordinate regression loss processing: {e}")
