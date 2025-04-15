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
        """Calculate all losses for the model with explicit shape handling

        Args:
            outputs: Dictionary of model outputs
            labels: Target token IDs of shape [batch_size, seq_length]
            segment_ids: Segment IDs of shape [batch_size, seq_length]
            hidden_states: Hidden states of shape [batch_size, seq_length, hidden_size]
            batch_size: Batch size

        Returns:
            Updated outputs dictionary with loss values
        """
        try:
            # Calculate main token prediction loss
            self._calculate_token_loss(outputs, labels, segment_ids, batch_size)

            # Calculate component-specific losses
            if "space_group_logits" in outputs:
                self._calculate_space_group_loss(outputs, labels, segment_ids)

            if "element_logits" in outputs:
                self._calculate_element_loss(outputs, labels, segment_ids)

            if "wyckoff_logits" in outputs:
                self._calculate_wyckoff_loss(outputs, labels, segment_ids)

            # Calculate continuous prediction losses
            self._calculate_continuous_losses(
                outputs, labels, segment_ids, hidden_states
            )
        except Exception:
            # Ensure a loss value always exists
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
        # Prepare the labels and logits
        if "logits" not in outputs:
            outputs["loss"] = torch.tensor(
                0.0, device=labels.device, requires_grad=True
            )
            return

        shift_logits = outputs["logits"][:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()
        shift_segments = segment_ids[:, 1:].contiguous()

        # Validate labels
        valid_labels = (shift_labels >= 0) & (shift_labels < self.config.vocab_size) | (
            shift_labels == -100
        )
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

        # Apply segment-based weighting if configuration exists
        weighted_losses = token_losses.clone()

        # Apply composition weight if configured
        if (
            hasattr(self.config, "composition_loss_weight")
            and self.config.composition_loss_weight != 1.0
        ):
            comp_indices = (shift_segments == self.segment_composition).float()
            if torch.any(comp_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.composition_loss_weight - 1.0) * comp_indices
                )

        # Apply space group weight if configured
        if (
            hasattr(self.config, "space_group_loss_weight")
            and self.config.space_group_loss_weight != 1.0
        ):
            sg_indices = (shift_segments == self.segment_space_group).float()
            if torch.any(sg_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.space_group_loss_weight - 1.0) * sg_indices
                )

        # Apply lattice weight if configured
        if (
            hasattr(self.config, "lattice_loss_weight")
            and self.config.lattice_loss_weight != 1.0
        ):
            lattice_indices = (shift_segments == self.segment_lattice).float()
            if torch.any(lattice_indices):
                weighted_losses = weighted_losses * (
                    1.0 + (self.config.lattice_loss_weight - 1.0) * lattice_indices
                )

        # Apply atom weight if configured
        if (
            hasattr(self.config, "atom_loss_weight")
            and self.config.atom_loss_weight != 1.0
        ):
            atom_indices = (
                (shift_segments == self.segment_element)
                | (shift_segments == self.segment_wyckoff)
                | (shift_segments == self.segment_coordinate)
            ).float()
            if torch.any(atom_indices):
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

    def _calculate_space_group_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> None:
        """Calculate space group prediction loss"""
        # Only calculate if logits are available
        if "space_group_logits" not in outputs:
            return

        sg_logits = outputs["space_group_logits"]

        # Get space group labels
        sg_labels_mask = segment_ids == self.segment_space_group
        if not torch.any(sg_labels_mask):
            return

        space_group_labels = labels[sg_labels_mask]

        # Validate labels
        valid_labels = (space_group_labels >= 0) & (
            space_group_labels < sg_logits.size(-1)
        ) | (space_group_labels == -100)

        if not torch.all(valid_labels):
            space_group_labels = torch.where(
                valid_labels,
                space_group_labels,
                torch.tensor(-100, device=space_group_labels.device),
            )

        # Check sizes match
        if sg_logits.size(0) != space_group_labels.size(0):
            return

        # Calculate loss with label smoothing
        try:
            # Enhanced cleaning of logits to avoid NaN/Inf
            sg_logits = torch.nan_to_num(sg_logits, nan=0.0, posinf=0.0, neginf=-1e5)
    
            # Apply additional safeguards
            sg_logits = torch.clamp(sg_logits, min=-50.0, max=50.0)
    
            # Add small epsilon for numerical stability
            sg_logits = sg_logits + torch.finfo(sg_logits.dtype).eps
    
            # Apply standard cross-entropy loss with label smoothing
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1, reduction='none')
    
            # Calculate loss with element-wise reduction first
            element_losses = loss_fct(sg_logits, space_group_labels)
    
            # Clean any NaN/Inf values in the element-wise losses
            element_losses = torch.nan_to_num(element_losses, nan=0.0, posinf=0.0, neginf=0.0)
    
            # Take the mean of valid losses
            if element_losses.numel() > 0 and torch.sum(element_losses) > 0:
                space_group_loss = element_losses.mean()
        
                # Final NaN check before adding to outputs
                if not torch.isnan(space_group_loss) and not torch.isinf(space_group_loss):
                    outputs["space_group_loss"] = space_group_loss
        except Exception:
            # Skip on error
            pass

    def _calculate_element_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> None:
        """Calculate element prediction loss"""
        # Only calculate if logits are available
        if "element_logits" not in outputs:
            return

        element_logits = outputs["element_logits"]

        # Get element labels
        element_mask = segment_ids == self.segment_element
        if not torch.any(element_mask):
            return

        element_labels = labels[element_mask]

        # Validate labels
        valid_labels = (element_labels >= 0) & (
            element_labels < element_logits.size(-1)
        ) | (element_labels == -100)

        if not torch.all(valid_labels):
            element_labels = torch.where(
                valid_labels,
                element_labels,
                torch.tensor(-100, device=element_labels.device),
            )

        # Check sizes match
        if element_logits.size(0) != element_labels.size(0):
            return

        # Calculate standard cross-entropy loss
        try:
            element_loss = F.cross_entropy(
                element_logits, element_labels, ignore_index=-100
            )
            outputs["element_loss"] = element_loss
        except Exception:
            # Skip on error
            pass

    def _calculate_wyckoff_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
    ) -> None:
        """Calculate Wyckoff position prediction loss"""
        # Only calculate if logits are available
        if "wyckoff_logits" not in outputs:
            return

        wyckoff_logits = outputs["wyckoff_logits"]

        # Get Wyckoff labels
        wyckoff_mask = segment_ids == self.segment_wyckoff
        if not torch.any(wyckoff_mask):
            return

        wyckoff_labels = labels[wyckoff_mask]

        # Validate labels
        valid_labels = (wyckoff_labels >= 0) & (
            wyckoff_labels < wyckoff_logits.size(-1)
        ) | (wyckoff_labels == -100)

        if not torch.all(valid_labels):
            wyckoff_labels = torch.where(
                valid_labels,
                wyckoff_labels,
                torch.tensor(-100, device=wyckoff_labels.device),
            )

        # Check sizes match
        if wyckoff_logits.size(0) != wyckoff_labels.size(0):
            return

        # Calculate standard cross-entropy loss
        try:
            wyckoff_loss = F.cross_entropy(
                wyckoff_logits, wyckoff_labels, ignore_index=-100
            )
            outputs["wyckoff_loss"] = wyckoff_loss
        except Exception:
            # Skip on error
            pass

    def _calculate_continuous_losses(
        self,
        outputs: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        segment_ids: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> None:
        """Calculate losses for continuous values (lattice parameters, coordinates)"""
        # Lattice parameter regression loss
        self._calculate_lattice_regression_loss(outputs)

        # Coordinate regression loss
        self._calculate_coordinate_regression_loss(outputs)

    def _calculate_lattice_regression_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Calculate lattice parameter regression loss using mixture density networks"""
        # Only calculate if we're training and have lattice parameters
        if not self.training or "lattice_mog_params" not in outputs:
            return

        # Check if we have ground truth values
        if (
            not hasattr(self, "_lattice_ground_truth")
            or self._lattice_ground_truth is None
        ):
            return

        try:
            # Get ground truth values
            gt_lengths = self._lattice_ground_truth["lengths"]
            gt_angles = self._lattice_ground_truth["angles"]

            # Shape: [batch_size, 6]
            gt_lattice_params = torch.cat([gt_lengths, gt_angles], dim=-1)

            # Calculate negative log likelihood using the continuous head
            mog_dist = self.continuous_heads["lattice"].get_distribution(
                outputs["lattice_mog_params"]
            )
            log_prob = mog_dist.log_prob(gt_lattice_params)
            lattice_regression_loss = -log_prob.mean()
            outputs["lattice_regression_loss"] = lattice_regression_loss
        except Exception:
            # Skip on error
            pass

    def _calculate_coordinate_regression_loss(
        self,
        outputs: Dict[str, torch.Tensor],
    ) -> None:
        """Calculate coordinate regression loss using mixture density networks"""
        # Only calculate if we're training and have coordinate parameters
        if not self.training or "coord_movm_params" not in outputs:
            return

        # Check if we have ground truth values
        if (
            not hasattr(self, "_coordinate_ground_truth")
            or self._coordinate_ground_truth is None
        ):
            return

        try:
            # Get ground truth coordinates
            gt_coords = self._coordinate_ground_truth["fractional_coords"]

            # Calculate negative log likelihood using the continuous head
            movm_dist = self.continuous_heads["coordinate"].get_distribution(
                outputs["coord_movm_params"]
            )
            log_prob = movm_dist.log_prob(gt_coords)
            coord_regression_loss = -log_prob.mean()
            outputs["coordinate_regression_loss"] = coord_regression_loss
        except Exception:
            # Skip on error
            pass
