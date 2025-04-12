from typing import List, Optional
import numpy as np
import os
import csv
import ast
import re
import logging

logger = logging.getLogger(__name__)

try:
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning(
        "PyTorch not available. Only numpy arrays will be supported for coordinate operations."
    )


class SpaceGroupWyckoffMapping:
    """Maps space groups to their allowed Wyckoff positions using data from CSV files."""

    def __init__(self, csv_dir=None):
        """Initialize the mapping between space groups and their Wyckoff positions from CSV files.

        Args:
            csv_dir: Directory containing the CSV files. If None, looks in the same directory as this file.
        """
        if csv_dir is None:
            csv_dir = os.path.dirname(os.path.abspath(__file__))

        self.wyckoff_list_path = os.path.join(csv_dir, "wyckoff_list.csv")
        self.wyckoff_symbols_path = os.path.join(csv_dir, "wyckoff_symbols.csv")

        logger.info(f"Looking for Wyckoff data in {csv_dir}")

        # Initialize data structures
        self.sg_to_wyckoff = {}
        self.sg_wyckoff_to_multiplicity = {}
        self.sg_wyckoff_to_constraints = {}
        self.sg_to_operations = {}

        # Load data from CSV files
        self._load_csv_data()

    def _load_csv_data(self):
        """Load data from CSV files and populate internal data structures."""
        # Initialize with fallback data first in case CSV loading fails
        self._initialize_fallback_data()

        # Then try to load the more detailed data from CSV
        try:
            # Load Wyckoff symbols (with multiplicity)
            self._load_wyckoff_symbols()

            # Load Wyckoff positions (for constraints)
            self._load_wyckoff_positions()

            # Validate loaded data
            self._validate_data()

            logger.info(
                f"Loaded Wyckoff data for {len(self.sg_to_wyckoff)} space groups"
            )
        except Exception as e:
            logger.error(f"Error loading Wyckoff data from CSV: {e}")
            logger.info("Using fallback data for all space groups")

    def _load_wyckoff_symbols(self):
        """Load Wyckoff symbols from CSV file."""
        try:
            if not os.path.exists(self.wyckoff_symbols_path):
                logger.warning(
                    f"Wyckoff symbols file not found: {self.wyckoff_symbols_path}"
                )
                return

            with open(self.wyckoff_symbols_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    space_group = int(row["Space Group"])
                    # Parse the symbols list using ast.literal_eval
                    wyckoff_symbols = ast.literal_eval(row["Wyckoff Symbols"])

                    # Extract letters and multiplicities
                    wyckoff_letters = []
                    for symbol in wyckoff_symbols:
                        # Extract letter (last character) and multiplicity (everything before)
                        letter = symbol[-1]
                        multiplicity = int(re.match(r"(\d+)", symbol).group(1))

                        wyckoff_letters.append(letter)
                        self.sg_wyckoff_to_multiplicity.setdefault(space_group, {})[
                            letter
                        ] = multiplicity

                    self.sg_to_wyckoff[space_group] = wyckoff_letters

        except Exception as e:
            logger.error(f"Error loading Wyckoff symbols: {e}")
            # Continue with fallback data

    def _load_wyckoff_positions(self):
        """Load Wyckoff positions from CSV file for constraints."""
        try:
            if not os.path.exists(self.wyckoff_list_path):
                logger.warning(
                    f"Wyckoff positions file not found: {self.wyckoff_list_path}"
                )
                return

            with open(self.wyckoff_list_path, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    space_group = int(row["Space Group"])
                    # Parse the positions list using ast.literal_eval
                    wyckoff_positions_list = ast.literal_eval(row["Wyckoff Positions"])

                    # Set up constraints dictionary for this space group
                    self.sg_wyckoff_to_constraints.setdefault(space_group, {})

                    # Match positions with letters from sg_to_wyckoff
                    if space_group in self.sg_to_wyckoff:
                        wyckoff_letters = self.sg_to_wyckoff[space_group]

                        for i, letter in enumerate(reversed(wyckoff_letters)):
                            if i < len(wyckoff_positions_list):
                                positions = wyckoff_positions_list[i]
                                self.sg_wyckoff_to_constraints[space_group][letter] = (
                                    self._extract_constraints(positions)
                                )

        except Exception as e:
            logger.error(f"Error loading Wyckoff positions: {e}")
            # Continue with fallback constraints

    def _extract_constraints(self, positions):
        """Extract coordinate constraints from Wyckoff position strings.

        Args:
            positions: List of position strings like ['x, y, z', '1/2, y, -z+1/2']

        Returns:
            List of constraints, where each constraint is a list of 3 elements (x,y,z)
            where None means the coordinate is free, and a float means it's fixed
        """
        constraints = []

        # Handle both string and list inputs
        if isinstance(positions, str):
            positions = [positions]

        for position in positions:
            coords = position.split(",")
            if len(coords) != 3:
                continue

            constraint = []
            for coord in coords:
                coord = coord.strip()

                # Check if this is a fixed coordinate (doesn't contain a variable)
                if (
                    "x" not in coord.lower()
                    and "y" not in coord.lower()
                    and "z" not in coord.lower()
                ):
                    # Parse fractions like 1/2
                    if "/" in coord:
                        parts = coord.split("/")
                        if len(parts) == 2:
                            try:
                                num = float(parts[0])
                                denom = float(parts[1])
                                value = num / denom
                            except ValueError:
                                value = 0.0
                        else:
                            value = 0.0
                    else:
                        try:
                            value = float(coord) if coord.strip() else 0.0
                        except ValueError:
                            value = 0.0
                    constraint.append(value)
                else:
                    # Variable coordinate (not fixed)
                    constraint.append(None)

            constraints.append(constraint)

        return constraints

    def _validate_data(self):
        """Validate that we have data for all 230 space groups."""
        missing_sgs = []
        for sg in range(1, 231):
            if sg not in self.sg_to_wyckoff:
                missing_sgs.append(sg)

        if missing_sgs:
            logger.warning(
                f"Missing data for {len(missing_sgs)} space groups: {missing_sgs}"
            )
            # Add default data for missing space groups
            for sg in missing_sgs:
                self.sg_to_wyckoff[sg] = ["a"]
                self.sg_wyckoff_to_multiplicity.setdefault(sg, {})["a"] = 1
                self.sg_wyckoff_to_constraints.setdefault(sg, {})["a"] = [
                    [None, None, None]
                ]

    def _initialize_fallback_data(self):
        """Initialize fallback data when CSV files are not available."""
        logger.info("Initializing fallback Wyckoff data (minimal)")

        # Initialize with minimal data for space groups 1-230
        for sg in range(1, 231):
            # Always provide at least Wyckoff position 'a'
            self.sg_to_wyckoff[sg] = ["a"]
            self.sg_wyckoff_to_multiplicity.setdefault(sg, {})["a"] = 1
            self.sg_wyckoff_to_constraints.setdefault(sg, {})["a"] = [
                [None, None, None]
            ]

    def get_allowed_wyckoff_positions(self, space_group: int) -> List[str]:
        """Get allowed Wyckoff positions for a given space group.

        Args:
            space_group: Space group number (1-230)

        Returns:
            List of allowed Wyckoff position letters
        """
        return self.sg_to_wyckoff.get(space_group, ["a"])

    def get_wyckoff_multiplicity(self, space_group: int, wyckoff_letter: str) -> int:
        """Get the multiplicity of a specific Wyckoff position in a space group.

        Args:
            space_group: Space group number (1-230)
            wyckoff_letter: Wyckoff position letter

        Returns:
            Multiplicity of the Wyckoff position
        """
        return self.sg_wyckoff_to_multiplicity.get(space_group, {}).get(
            wyckoff_letter, 1
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
        # Return empty constraints if not found - means no specific constraints
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
        coords,  # Can be numpy array or torch tensor
        space_group: int,
        wyckoff_letter: str,
    ):
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

        # Check if input is torch tensor or numpy array
        is_torch = False
        if TORCH_AVAILABLE:
            import torch

            is_torch = isinstance(coords, torch.Tensor)

        # Apply constraints based on whether input is numpy or torch
        if not is_torch:
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
        coords,  # Can be numpy array or torch tensor
        space_group: int,
        wyckoff_letter: str,
    ):
        """Generate symmetry-equivalent positions for coordinates based on multiplicity.

        Args:
            coords: Original coordinates [batch_size, 3]
            space_group: Space group number
            wyckoff_letter: Wyckoff position letter

        Returns:
            Full set of symmetry-equivalent positions (approximate method)
        """
        # Get the multiplicity for this Wyckoff position
        multiplicity = self.get_wyckoff_multiplicity(space_group, wyckoff_letter)

        if multiplicity <= 1:
            return coords

        # Check if input is torch tensor or numpy array
        is_torch = False
        device = None
        dtype = None
        if TORCH_AVAILABLE:
            import torch

            is_torch = isinstance(coords, torch.Tensor)
            if is_torch:
                device = coords.device
                dtype = coords.dtype

        # Convert to numpy for processing if it's a torch tensor
        coords_np = coords.cpu().numpy() if is_torch else coords.copy()

        # Get constraints to understand which coordinates are fixed
        constraints = self.get_coordinate_constraints(space_group, wyckoff_letter)
        fixed_coords = [False, False, False]  # Whether x, y, z are fixed

        if constraints and len(constraints) > 0:
            for i, value in enumerate(constraints[0]):
                if i < 3:
                    fixed_coords[i] = value is not None

        # Generate equivalent positions based on the multiplicity and fixed coordinates
        equiv_positions = [coords_np]

        # This is a simplified approach that doesn't use actual symmetry operations
        # For a more accurate implementation, you would need the actual symmetry operations
        for i in range(1, multiplicity):
            # Generate a position by applying simple transformations
            # This is just an approximation and should be replaced with actual symmetry operations
            new_pos = coords_np.copy()

            # Apply transformations based on which coordinates are fixed
            for axis in range(3):
                if not fixed_coords[axis]:
                    # For free coordinates, we can generate equivalents
                    # This is a simplified approach - in reality, the symmetry operations are more complex
                    if i % 2 == 1:
                        # Simple reflection for demonstration
                        new_pos[:, axis] = 1.0 - new_pos[:, axis]

                    if i % 4 >= 2:
                        # Another transformation
                        new_pos[:, axis] = 0.5 + new_pos[:, axis] % 0.5

            equiv_positions.append(new_pos)

            # Ensure we don't exceed the multiplicity
            if len(equiv_positions) >= multiplicity:
                break

        # Combine all positions
        all_positions = np.vstack(equiv_positions)

        # Ensure coordinates are in range [0, 1)
        all_positions = all_positions % 1.0

        # Convert back to PyTorch tensor if input was a tensor
        if is_torch and TORCH_AVAILABLE:
            import torch

            return torch.tensor(all_positions, device=device, dtype=dtype)
        else:
            return all_positions
