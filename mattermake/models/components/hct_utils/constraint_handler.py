from typing import Dict, Optional, Any, Tuple
import torch
from mattermake.utils.hct_wyckoff_mapping import SpaceGroupWyckoffMapping


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
                "wyckoff_mult_to_token": {},
                "token_to_wyckoff_mult": {},
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
            token_id: Token ID representing the Wyckoff position or combined Wyckoff-multiplicity
        """
        if batch_idx not in self.space_groups:
            return

        space_group = self.space_groups[batch_idx]

        # Check if this is a combined Wyckoff-multiplicity token (token IDs starting at 3000)
        if token_id >= 3000:
            # Extract Wyckoff letter from token ID
            # Combined token format: 3000 + (sg * 100) + (ord(letter) - ord('a'))
            relative_token = token_id - 3000
            wyckoff_idx = relative_token % 100
            wyckoff_letter = chr(ord("a") + wyckoff_idx)

            # Get multiplicity directly from the token
            # The multiplicity is already encoded in the token mapping
            multiplicity = self.sg_wyckoff_mapping.get_wyckoff_multiplicity(
                space_group, wyckoff_letter
            )

            # Special case: if we can't determine multiplicity from mapping,
            # try to extract it from token name in token_maps
            if multiplicity <= 0:
                token_name = self.token_maps.get("token_to_wyckoff_mult", {}).get(
                    str(token_id)
                )
                if token_name and token_name.startswith("WYCK_"):
                    # Extract multiplicity from token name (format: WYCK_a4 for position 'a' with multiplicity 4)
                    try:
                        letter_and_mult = token_name[5:]  # Skip 'WYCK_'
                        if len(letter_and_mult) > 1:
                            wyckoff_letter = letter_and_mult[0]
                            multiplicity = int(letter_and_mult[1:])
                    except (ValueError, IndexError):
                        # If parsing fails, use default multiplicity
                        multiplicity = 1
        else:
            # Original handling for individual Wyckoff tokens
            wyckoff_letter = self.token_maps.get("token_to_wyckoff", {}).get(
                str(token_id)
            )
            if wyckoff_letter is None:
                # Convert token ID to letter (a=0, b=1, etc.)
                wyckoff_letter = chr(ord("a") + (token_id % 26))

            # Get multiplicity from the mapping
            multiplicity = self.sg_wyckoff_mapping.get_wyckoff_multiplicity(
                space_group, wyckoff_letter
            )

        # Check if this Wyckoff position is valid for the space group
        allowed_wyckoff = self.sg_wyckoff_mapping.get_allowed_wyckoff_positions(
            space_group
        )

        if wyckoff_letter not in allowed_wyckoff:
            # If invalid, use the first allowed Wyckoff position
            if allowed_wyckoff:
                wyckoff_letter = allowed_wyckoff[0]
                # Update multiplicity for the new Wyckoff letter
                multiplicity = self.sg_wyckoff_mapping.get_wyckoff_multiplicity(
                    space_group, wyckoff_letter
                )
            else:
                wyckoff_letter = "a"  # Fallback
                multiplicity = 1

        # Store the Wyckoff position
        if batch_idx not in self.wyckoff_positions:
            self.wyckoff_positions[batch_idx] = []
            self.wyckoff_multiplicities[batch_idx] = []

        self.wyckoff_positions[batch_idx].append(wyckoff_letter)
        self.wyckoff_multiplicities[batch_idx].append(multiplicity)

    def get_wyckoff_mask(
        self, batch_idx: int, use_combined_tokens: bool = True
    ) -> Optional[torch.Tensor]:
        """Get a mask for valid Wyckoff positions for the current space group.

        Args:
            batch_idx: Batch index
            use_combined_tokens: Whether to create a mask for combined Wyckoff-multiplicity tokens

        Returns:
            Boolean mask where True indicates a valid Wyckoff position, or None if no space group is set
        """
        if batch_idx not in self.space_groups:
            return None

        space_group = self.space_groups[batch_idx]

        if not use_combined_tokens:
            # Original behavior - create a mask for letter-only tokens
            mask = self.sg_wyckoff_mapping.create_wyckoff_mask(space_group)
            return torch.tensor(mask, dtype=torch.bool)
        else:
            # For combined tokens, we need to create a larger mask
            # First, get all allowed Wyckoff letters for this space group
            allowed_wyckoff = self.sg_wyckoff_mapping.get_allowed_wyckoff_positions(
                space_group
            )

            # Create a mapping from token_id to validity
            token_mask = {}

            # For each token in the vocabulary
            for token_key, token_id in self.token_maps.get(
                "wyckoff_mult_to_token", {}
            ).items():
                # Parse token key to get letter and multiplicity
                if isinstance(token_key, str) and len(token_key) >= 1:
                    letter = token_key[0]
                    # Check if this Wyckoff letter is allowed in this space group
                    is_valid = letter in allowed_wyckoff
                    token_mask[token_id] = is_valid

            # If we don't have any mapped tokens, create a basic mask for the token range
            if not token_mask and use_combined_tokens:
                # Define the range of combined tokens (3000 to ~33000)
                num_wyckoff_letters = 26
                for wyckoff_idx in range(num_wyckoff_letters):
                    letter = chr(ord("a") + wyckoff_idx)
                    is_valid = letter in allowed_wyckoff

                    # Calculate token IDs for this letter across all space groups
                    # We only care about tokens for the current space group
                    token_id = 3000 + (space_group * 100) + wyckoff_idx
                    token_mask[token_id] = is_valid

            # Convert the dictionary to a tensor mask
            # Size of mask should cover all possible token IDs
            max_token_id = (
                max(token_mask.keys()) if token_mask else 3000 + (230 * 100) + 26
            )
            mask = torch.zeros(max_token_id + 1, dtype=torch.bool)

            for token_id, is_valid in token_mask.items():
                if token_id < mask.size(0):
                    mask[token_id] = is_valid

            return mask

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
