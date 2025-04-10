from typing import List, Optional, Union
import numpy as np
import torch
import spglib
from ase import Atoms


class SpaceGroupWyckoffMapping:
    """Maps space groups to their allowed Wyckoff positions and handles constraints using spglib."""

    def __init__(self):
        """Initialize the mapping between space groups and their Wyckoff positions using spglib."""
        self._initialize_space_group_data()

    def _initialize_space_group_data(self):
        """Initialize space group data from spglib."""
        self.sg_to_wyckoff = {}
        self.sg_wyckoff_to_multiplicity = {}
        self.sg_wyckoff_to_constraints = {}
        self.sg_to_operations = {}

        # Iterate through all space groups
        for sg_number in range(1, 231):
            # Get dataset for this space group
            try:
                dataset = spglib.get_symmetry_dataset(
                    self._create_dummy_cell(sg_number)
                )

                # Get Wyckoff positions data
                wyckoff_positions = spglib.get_wyckoff_positions(sg_number)

                self.sg_to_wyckoff[sg_number] = []
                self.sg_wyckoff_to_multiplicity[sg_number] = {}
                self.sg_wyckoff_to_constraints[sg_number] = {}

                # Get rotation matrices and translations for this space group
                rot_matrices = dataset["rotations"]
                translations = dataset["translations"]
                self.sg_to_operations[sg_number] = (rot_matrices, translations)

                # Process Wyckoff positions
                for letter, positions in zip(
                    "abcdefghijklmnopqrstuvwxyz", wyckoff_positions
                ):
                    if len(positions) == 0:
                        continue

                    # Store mapping
                    self.sg_to_wyckoff[sg_number].append(letter)
                    self.sg_wyckoff_to_multiplicity[sg_number][letter] = len(positions)

                    # Determine coordinate constraints
                    # This is a simplification - in reality we'd need to analyze the positions
                    constraints = []
                    if len(positions) == 1:
                        # For multiplicity 1, we can directly extract fixed coordinates
                        constraint = []
                        for i, coord in enumerate(positions[0]):
                            # Check if the coordinate is fixed (approximately equal to a simple fraction)
                            for denom in [1, 2, 3, 4, 6]:
                                for num in range(denom + 1):
                                    if abs(coord - num / denom) < 1e-5:
                                        constraint.append(num / denom)
                                        break
                                else:
                                    continue
                                break
                            else:
                                constraint.append(None)  # Free coordinate
                        constraints.append(constraint)
                    else:
                        # For higher multiplicities, we need more complex analysis
                        # For simplicity in this implementation, we'll just mark all as free
                        constraints.append([None, None, None])

                    self.sg_wyckoff_to_constraints[sg_number][letter] = constraints

            except Exception as e:
                print(f"Warning: Failed to process space group {sg_number}: {e}")
                # Add placeholder data
                self.sg_to_wyckoff[sg_number] = ["a"]
                self.sg_wyckoff_to_multiplicity[sg_number] = {"a": 1}
                self.sg_wyckoff_to_constraints[sg_number] = {"a": [[None, None, None]]}

    def _create_dummy_cell(self, sg_number):
        """Create a dummy cell for the given space group number."""
        # Create a simple cubic cell with a single atom
        # This is just to get the space group operations
        a = 1.0
        cell = np.array([[a, 0, 0], [0, a, 0], [0, 0, a]])
        positions = np.array([[0, 0, 0]])
        atoms = Atoms("H", positions=positions, cell=cell, pbc=True)

        # Set the space group number
        atoms.info["spacegroup"] = sg_number

        return (cell, positions, np.array([1]))  # Cell, positions, types

    def get_allowed_wyckoff_positions(self, space_group: int) -> List[str]:
        """Get allowed Wyckoff positions for a given space group.

        Args:
            space_group: Space group number (1-230)

        Returns:
            List of allowed Wyckoff position letters
        """
        return self.sg_to_wyckoff.get(space_group, [])

    def get_wyckoff_multiplicity(self, space_group: int, wyckoff_letter: str) -> int:
        """Get the multiplicity of a specific Wyckoff position in a space group.

        Args:
            space_group: Space group number (1-230)
            wyckoff_letter: Wyckoff position letter

        Returns:
            Multiplicity of the Wyckoff position
        """
        return self.sg_wyckoff_to_multiplicity.get(space_group, {}).get(
            wyckoff_letter, 0
        )

    def get_coordinate_constraints(
        self, space_group: int, wyckoff_letter: str
    ) -> List[List[Optional[float]]]:
        """Get coordinate constraints for a Wyckoff position.

        Args:
            space_group: Space group number (1-230)
            wyckoff_letter: Wyckoff position letter

        Returns:
            List of coordinate constraints. Each constraint is a list of 3 elements (x,y,z),
            where None means the coordinate is free, and a float means it's fixed to that value.
        """
        return self.sg_wyckoff_to_constraints.get(space_group, {}).get(
            wyckoff_letter, []
        )

    def create_wyckoff_mask(
        self, space_group: int, num_wyckoff_positions: int = 26
    ) -> List[bool]:
        """Create a boolean mask for valid Wyckoff positions for a space group.

        Args:
            space_group: Space group number (1-230)
            num_wyckoff_positions: Total number of possible Wyckoff positions (default: 26 for a-z)

        Returns:
            Boolean mask where True indicates a valid Wyckoff position
        """
        allowed = self.get_allowed_wyckoff_positions(space_group)
        mask = [False] * num_wyckoff_positions

        for letter in allowed:
            # Convert letter to index (a=0, b=1, ...)
            index = ord(letter.lower()) - ord("a")
            if 0 <= index < num_wyckoff_positions:
                mask[index] = True

        return mask

    def apply_coordinate_constraints(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        space_group: int,
        wyckoff_letter: str,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Apply coordinate constraints for a given Wyckoff position.

        Args:
            coords: Coordinates to constrain [batch_size, 3]
            space_group: Space group number
            wyckoff_letter: Wyckoff position letter

        Returns:
            Constrained coordinates
        """
        constraints = self.get_coordinate_constraints(space_group, wyckoff_letter)
        if not constraints or not constraints[0]:
            return coords

        # Get the first constraint (for simplicity)
        constraint = constraints[0]

        # Apply constraints based on whether input is numpy or torch
        if isinstance(coords, np.ndarray):
            constrained_coords = coords.copy()
            for i, fixed_value in enumerate(constraint):
                if fixed_value is not None and i < coords.shape[1]:
                    constrained_coords[:, i] = fixed_value
        else:  # torch.Tensor
            constrained_coords = coords.clone()
            for i, fixed_value in enumerate(constraint):
                if fixed_value is not None and i < coords.shape[1]:
                    constrained_coords[:, i] = fixed_value

        return constrained_coords

    def generate_symmetry_equivalent_positions(
        self,
        coords: Union[np.ndarray, torch.Tensor],
        space_group: int,
        wyckoff_letter: str,
    ) -> Union[np.ndarray, torch.Tensor]:
        """Generate symmetry-equivalent positions for coordinates.

        Args:
            coords: Original coordinates [batch_size, 3]
            space_group: Space group number
            wyckoff_letter: Wyckoff position letter

        Returns:
            Full set of symmetry-equivalent positions
        """
        # Get the symmetry operations for this space group
        if space_group not in self.sg_to_operations:
            return coords

        rotations, translations = self.sg_to_operations[space_group]
        multiplicity = self.get_wyckoff_multiplicity(space_group, wyckoff_letter)

        if multiplicity <= 1:
            return coords

        # Convert PyTorch tensor to numpy if needed
        is_torch = isinstance(coords, torch.Tensor)
        if is_torch:
            device = coords.device
            dtype = coords.dtype
            coords_np = coords.cpu().numpy()
        else:
            coords_np = coords

        # Apply symmetry operations to generate equivalent positions
        equiv_positions = []

        for i in range(min(multiplicity, len(rotations))):
            rot = rotations[i]
            trans = translations[i]

            # Apply the symmetry operation
            new_pos = np.dot(coords_np, rot.T) + trans

            # Ensure coordinates are in the unit cell [0, 1)
            new_pos = new_pos % 1.0

            equiv_positions.append(new_pos)

        # Combine all positions
        all_positions = np.vstack(equiv_positions)

        # Convert back to PyTorch tensor if input was a tensor
        if is_torch:
            return torch.tensor(all_positions, device=device, dtype=dtype)
        else:
            return all_positions
