from typing import Dict, Optional, Any, Tuple
import torch
from mattermake.data.components.space_group_wyckoff_mapping import (
    SpaceGroupWyckoffMapping,
)


class CrystalConstraintHandler:
    """Handles constraints during crystal generation."""

    def __init__(self, constraints: Dict[str, Any] = None):
        """Initialize the constraint handler.

        Args:
            constraints: Dictionary of constraints for generation
        """
        self.constraints = constraints or {}
        self.sg_wyckoff_mapping = SpaceGroupWyckoffMapping()

        # Track space groups and Wyckoff positions during generation
        self.space_groups = {}
        self.wyckoff_positions = {}
        self.wyckoff_multiplicities = {}
        self.current_atom_indices = {}

        # Maps for token conversion
        if constraints and "token_id_maps" in constraints:
            self.token_maps = constraints["token_id_maps"]
        else:
            self.token_maps = {
                "space_group_to_token": {},
                "token_to_space_group": {},
                "wyckoff_to_token": {},
                "token_to_wyckoff": {},
            }

    def update_space_group(self, batch_idx: int, token_id: int) -> None:
        """Update the tracked space group for a batch.

        Args:
            batch_idx: Batch index
            token_id: Token ID representing the space group
        """
        # Convert token to space group number
        space_group = int(
            self.token_maps.get("token_to_space_group", {}).get(str(token_id), token_id)
        )

        # Ensure space group is in valid range (1-230)
        if 1 <= space_group <= 230:
            self.space_groups[batch_idx] = space_group

            # Reset Wyckoff positions for this batch
            if batch_idx in self.wyckoff_positions:
                self.wyckoff_positions[batch_idx] = []
                self.wyckoff_multiplicities[batch_idx] = []

            # Reset atom indices
            if batch_idx in self.current_atom_indices:
                self.current_atom_indices[batch_idx] = 0

    def update_wyckoff_position(self, batch_idx: int, token_id: int) -> None:
        """Update the tracked Wyckoff position for a batch.

        Args:
            batch_idx: Batch index
            token_id: Token ID representing the Wyckoff position
        """
        if batch_idx not in self.space_groups:
            return

        # Convert token to Wyckoff letter
        wyckoff_letter = self.token_maps.get("token_to_wyckoff", {}).get(str(token_id))
        if wyckoff_letter is None:
            # Convert token ID to letter (a=0, b=1, etc.)
            wyckoff_letter = chr(ord("a") + (token_id % 26))

        # Check if this Wyckoff position is valid for the space group
        space_group = self.space_groups[batch_idx]
        allowed_wyckoff = self.sg_wyckoff_mapping.get_allowed_wyckoff_positions(
            space_group
        )

        if wyckoff_letter not in allowed_wyckoff:
            # If invalid, use the first allowed Wyckoff position
            if allowed_wyckoff:
                wyckoff_letter = allowed_wyckoff[0]
            else:
                wyckoff_letter = "a"  # Fallback

        # Store the Wyckoff position
        if batch_idx not in self.wyckoff_positions:
            self.wyckoff_positions[batch_idx] = []
            self.wyckoff_multiplicities[batch_idx] = []

        self.wyckoff_positions[batch_idx].append(wyckoff_letter)

        # Get and store multiplicity
        multiplicity = self.sg_wyckoff_mapping.get_wyckoff_multiplicity(
            space_group, wyckoff_letter
        )
        self.wyckoff_multiplicities[batch_idx].append(multiplicity)

    def get_wyckoff_mask(self, batch_idx: int) -> Optional[torch.Tensor]:
        """Get a mask for valid Wyckoff positions for the current space group.

        Args:
            batch_idx: Batch index

        Returns:
            Boolean mask where True indicates a valid Wyckoff position, or None if no space group is set
        """
        if batch_idx not in self.space_groups:
            return None

        space_group = self.space_groups[batch_idx]
        mask = self.sg_wyckoff_mapping.create_wyckoff_mask(space_group)
        return torch.tensor(mask, dtype=torch.bool)

    def apply_coordinate_constraints(
        self, coords: torch.Tensor, batch_idx: int, atom_idx: int
    ) -> torch.Tensor:
        """Apply coordinate constraints based on Wyckoff position.

        Args:
            coords: Coordinates to constrain [batch_size, 3]
            batch_idx: Batch index
            atom_idx: Atom index within the batch

        Returns:
            Constrained coordinates
        """
        if (
            batch_idx not in self.space_groups
            or batch_idx not in self.wyckoff_positions
            or atom_idx >= len(self.wyckoff_positions[batch_idx])
        ):
            return coords

        space_group = self.space_groups[batch_idx]
        wyckoff_letter = self.wyckoff_positions[batch_idx][atom_idx]

        return self.sg_wyckoff_mapping.apply_coordinate_constraints(
            coords, space_group, wyckoff_letter
        )

    def generate_symmetry_equivalent_atoms(
        self, coords: torch.Tensor, elements: torch.Tensor, batch_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate symmetry-equivalent atoms for a batch.

        Args:
            coords: Atom coordinates [num_atoms, 3]
            elements: Atom elements [num_atoms]
            batch_idx: Batch index

        Returns:
            Tuple of (expanded_coords, expanded_elements)
        """
        if (
            batch_idx not in self.space_groups
            or batch_idx not in self.wyckoff_positions
            or len(self.wyckoff_positions[batch_idx]) == 0
        ):
            return coords, elements

        space_group = self.space_groups[batch_idx]
        wyckoff_positions = self.wyckoff_positions[batch_idx]

        expanded_coords = []
        expanded_elements = []

        for atom_idx, wyckoff_letter in enumerate(wyckoff_positions):
            if atom_idx >= len(coords):
                break

            multiplicity = self.sg_wyckoff_mapping.get_wyckoff_multiplicity(
                space_group, wyckoff_letter
            )

            if multiplicity > 1:
                # Generate all symmetry-equivalent atoms
                equiv_coords = (
                    self.sg_wyckoff_mapping.generate_symmetry_equivalent_positions(
                        coords[atom_idx : atom_idx + 1], space_group, wyckoff_letter
                    )
                )
                expanded_coords.append(equiv_coords)

                # Repeat the element for each equivalent position
                expanded_elements.append(
                    elements[atom_idx].repeat(equiv_coords.shape[0])
                )
            else:
                expanded_coords.append(coords[atom_idx : atom_idx + 1])
                expanded_elements.append(elements[atom_idx : atom_idx + 1])

        return torch.cat(expanded_coords, dim=0), torch.cat(expanded_elements, dim=0)
