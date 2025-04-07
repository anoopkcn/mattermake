from typing import List, Dict, Any, Optional
from pymatgen.core import Structure, Lattice


def extract_composition_string(structure: Structure) -> str:
    """Extract a formatted composition string from a structure"""
    comp = structure.composition

    elements = sorted(comp.elements, key=lambda e: e.symbol)

    # Format: "Element1_count1-Element2_count2-..."
    formula_parts = []
    for element in elements:
        count = comp[element]
        count_int = round(float(count))
        formula_parts.append(f"{element.symbol}_{count_int}")

    return "-".join(formula_parts)


class CrystalSequenceDecoder:
    """Decoder for crystal structure token sequences"""

    def __init__(self, tokenizer):
        """
        Initialize with a CrystalTokenizer instance

        Args:
            tokenizer: A CrystalTokenizer instance
        """
        self.tokenizer = tokenizer

        self.SEGMENT_SPECIAL = 0
        self.SEGMENT_SPACE_GROUP = 1
        self.SEGMENT_LATTICE = 2
        self.SEGMENT_ELEMENT = 3
        self.SEGMENT_WYCKOFF = 4
        self.SEGMENT_COORDINATE = 5

    def token_to_space_group(self, token_id: int) -> Optional[int]:
        """Convert a token ID to a space group number"""
        token_name = self.tokenizer.idx_to_token.get(token_id)
        if token_name and token_name.startswith("SG_"):
            return int(token_name[3:])
        return None

    def token_to_element(self, token_id: int) -> Optional[str]:
        """Convert a token ID to an element symbol"""
        token_name = self.tokenizer.idx_to_token.get(token_id)
        if token_name and token_name.startswith("ELEM_"):
            return token_name[5:]
        return None

    def token_to_wyckoff(self, token_id: int) -> Optional[str]:
        """Convert a token ID to a Wyckoff letter"""
        token_name = self.tokenizer.idx_to_token.get(token_id)
        if token_name and token_name.startswith("WYCK_"):
            return token_name[5:]
        return None

    def token_to_lattice_param(self, token_id: int, param_type: str) -> Optional[float]:
        """
        Convert a token ID to a lattice parameter value

        Args:
            token_id: Token ID
            param_type: One of 'a', 'b', 'c', 'alpha', 'beta', 'gamma'

        Returns:
            Lattice parameter value
        """
        token_name = self.tokenizer.idx_to_token.get(token_id)
        if token_name and token_name.startswith("LAT_"):
            bin_idx = int(token_name[4:])

            # Convert bin index back to actual value
            if param_type in ["a", "b", "c"]:
                # Lattice lengths (in Angstroms)
                return bin_idx * (100.0 / self.tokenizer.lattice_bins)
            else:
                # Lattice angles (in degrees)
                return bin_idx * (180.0 / self.tokenizer.lattice_bins)
        return None

    def token_to_coordinate(self, token_id: int) -> Optional[float]:
        """Convert a token ID to a fractional coordinate value"""
        token_name = self.tokenizer.idx_to_token.get(token_id)
        if token_name and token_name.startswith("COORD_"):
            # Extract the coordinate value from the token name
            return float(token_name[6:])
        return None

    def decode_sequence(
        self, tokens: List[int], segment_ids: List[int]
    ) -> Dict[str, Any]:
        """
        Decode a token sequence into a crystal structure

        Args:
            tokens: List of token IDs
            segment_ids: List of segment IDs

        Returns:
            Dictionary with decoded crystal structure information
        """
        # Initialize structure components
        space_group = None
        lattice_params = [None] * 6
        atoms = []

        # Variables to track current atom being processed
        current_element = None
        current_wyckoff = None
        current_coords = []

        # Process tokens
        for i, (token, segment) in enumerate(zip(tokens, segment_ids)):
            if (
                token == self.tokenizer.BOS_TOKEN
                or token == self.tokenizer.EOS_TOKEN
                or token == self.tokenizer.PAD_TOKEN
            ):
                continue

            if segment == self.SEGMENT_SPACE_GROUP:
                space_group = self.token_to_space_group(token)

            elif segment == self.SEGMENT_LATTICE:
                # Track which lattice parameter we're processing
                lattice_param_idx = sum(1 for p in lattice_params if p is not None)
                if lattice_param_idx < 6:
                    param_type = ["a", "b", "c", "alpha", "beta", "gamma"][
                        lattice_param_idx
                    ]
                    lattice_params[lattice_param_idx] = self.token_to_lattice_param(
                        token, param_type
                    )

            elif segment == self.SEGMENT_ELEMENT:
                # If we were processing an atom, save it before starting a new one
                if current_element is not None and len(current_coords) > 0:
                    atoms.append(
                        {
                            "element": current_element,
                            "wyckoff": current_wyckoff,
                            "coords": current_coords,
                        }
                    )
                    current_coords = []

                current_element = self.token_to_element(token)
                current_wyckoff = None

            elif segment == self.SEGMENT_WYCKOFF:
                current_wyckoff = self.token_to_wyckoff(token)

            elif segment == self.SEGMENT_COORDINATE:
                coord_value = self.token_to_coordinate(token)
                if coord_value is not None:
                    current_coords.append(coord_value)

        # Add the last atom if it exists
        if current_element is not None and len(current_coords) > 0:
            atoms.append(
                {
                    "element": current_element,
                    "wyckoff": current_wyckoff,
                    "coords": current_coords,
                }
            )

        # Create the structure
        try:
            structure = self.create_structure(space_group, lattice_params, atoms)
            return {
                "space_group": space_group,
                "lattice_params": lattice_params,
                "atoms": atoms,
                "structure": structure,
                "valid": True,
            }
        except ValueError as e:
            return {
                "space_group": space_group,
                "lattice_params": lattice_params,
                "atoms": atoms,
                "error": str(e),
                "valid": False,
                "error_type": "validation_error",
            }
        except Exception as e:
            return {
                "space_group": space_group,
                "lattice_params": lattice_params,
                "atoms": atoms,
                "error": str(e),
                "valid": False,
                "error_type": "general_error",
            }

    def has_realistic_lattice_parameters(self, a, b, c, alpha, beta, gamma):
        """Check if lattice parameters are within reasonable bounds"""
        # Typical ranges for most inorganic crystals (in Angstroms and degrees)
        return (
            0.5 < a < 50
            and 0.5 < b < 50
            and 0.5 < c < 50
            and 20 < alpha < 160
            and 20 < beta < 160
            and 20 < gamma < 160
        )

    def create_structure(
        self, space_group: int, lattice_params: List[float], atoms: List[Dict[str, Any]]
    ) -> Structure:
        """
        Create a pymatgen Structure from decoded components

        Args:
            space_group: Space group number
            lattice_params: List of lattice parameters [a, b, c, alpha, beta, gamma]
            atoms: List of atoms with element, wyckoff, and coords

        Returns:
            Pymatgen Structure
        """
        a, b, c, alpha, beta, gamma = lattice_params

        if not self.has_realistic_lattice_parameters(a, b, c, alpha, beta, gamma):
            raise ValueError(
                f"Unrealistic lattice parameters: a={a}, b={b}, c={c}, "
                f"alpha={alpha}, beta={beta}, gamma={gamma}"
            )

        lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

        species = []
        coords = []

        # Convert to full structure based on space group
        # For full implementation, this would use pymatgen symmetry operations
        # to generate all equivalent positions

        # For now, just use the provided coordinates TODO
        for atom in atoms:
            element = atom["element"]
            atom_coords = atom["coords"]

            if len(atom_coords) < 3:
                atom_coords = atom_coords + [0.0] * (3 - len(atom_coords))
            elif len(atom_coords) > 3:
                atom_coords = atom_coords[:3]

            species.append(element)
            coords.append(atom_coords)

        structure = Structure(lattice, species, coords)

        # TODO: Apply symmetry operations (if needed) this should enforce the specified space group

        return structure
