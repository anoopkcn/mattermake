from typing import List, Dict, Any, Optional
from pymatgen.core import Structure, Lattice
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import logging
import numpy as np

logger = logging.getLogger(__name__)


def process_structure_from_cif_string(
    material_id, cif_string, tokenizer, standardize=True, symprec=0.01
):
    """Process a structure from a CIF string"""
    try:
        structure = Structure.from_str(cif_string, fmt="cif")

        if standardize:
            sga = SpacegroupAnalyzer(structure, symprec=symprec)
            structure = sga.get_conventional_standard_structure()
            space_group = sga.get_space_group_number()
        else:
            space_group = None

        token_data = tokenizer.tokenize_structure(structure)

        return {
            "material_id": material_id,
            "formula": structure.composition.reduced_formula,
            "space_group": space_group,
            "token_data": token_data,
            "structure": structure,
        }
    except Exception as e:
        logger.error(f"Error processing material {material_id}: {e}")
        return None


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
        
        # Add the Wyckoff mapping for multiplicity handling
        from mattermake.utils.hct_wyckoff_mapping import SpaceGroupWyckoffMapping
        self.wyckoff_mapping = SpaceGroupWyckoffMapping()

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

    def token_to_wyckoff(self, token_id: int) -> Optional[tuple]:
        """Convert a token ID to a Wyckoff letter and multiplicity
        
        Returns:
            Tuple of (letter, multiplicity) or just letter if using old format
        """
        token_name = self.tokenizer.idx_to_token.get(token_id)
        
        # Handle combined Wyckoff-multiplicity tokens (e.g., "WYCK_a4")
        if token_name and token_name.startswith("WYCK_") and len(token_name) > 6:
            try:
                letter = token_name[5]
                multiplicity = int(token_name[6:])
                return (letter, multiplicity)
            except (ValueError, IndexError):
                # Fallback to just the letter
                return token_name[5] if len(token_name) > 5 else None
        
        # Handle original Wyckoff tokens (e.g., "WYCK_a")
        elif token_name and token_name.startswith("WYCK_"):
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
                wyckoff_result = self.token_to_wyckoff(token)
                
                # Check if it's a tuple (combined format) or string (old format)
                if isinstance(wyckoff_result, tuple):
                    letter, multiplicity = wyckoff_result
                    current_wyckoff = {
                        "letter": letter,
                        "multiplicity": multiplicity
                    }
                else:
                    current_wyckoff = wyckoff_result

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

        # Process atoms with Wyckoff position information
        for atom in atoms:
            element = atom["element"]
            wyckoff_data = atom["wyckoff"]
            atom_coords = atom["coords"]
            
            # Check if we have detailed Wyckoff data with multiplicity
            if isinstance(wyckoff_data, dict) and "multiplicity" in wyckoff_data:
                wyckoff_letter = wyckoff_data["letter"]
                multiplicity = wyckoff_data["multiplicity"]
                
                # Log the Wyckoff position and multiplicity info
                logger.debug(f"Processing Wyckoff position {wyckoff_letter} with multiplicity {multiplicity}")
                
                # Make sure coords are properly formatted
                if len(atom_coords) < 3:
                    atom_coords = atom_coords + [0.0] * (3 - len(atom_coords))
                elif len(atom_coords) > 3:
                    atom_coords = atom_coords[:3]
                
                # Use the multiplicity to generate equivalent positions
                try:
                    # Convert coords to numpy array for processing
                    coords_array = np.array([atom_coords])
                    
                    # Generate equivalent positions using the Wyckoff mapping
                    equiv_coords = self.wyckoff_mapping.generate_symmetry_equivalent_positions(
                        coords_array, space_group, wyckoff_letter
                    )
                    
                    # Add all equivalent positions
                    for coord in equiv_coords:
                        species.append(element)
                        coords.append(coord.tolist() if hasattr(coord, 'tolist') else coord)
                        
                    continue  # Skip the standard addition below
                except Exception as e:
                    # Log the error and fall back to adding just the original position
                    logger.warning(f"Error generating equivalent positions: {e}")
            
            # If no multiplicity data or generation failed, add the single position
            if len(atom_coords) < 3:
                atom_coords = atom_coords + [0.0] * (3 - len(atom_coords))
            elif len(atom_coords) > 3:
                atom_coords = atom_coords[:3]
                
            species.append(element)
            coords.append(atom_coords)

        structure = Structure(lattice, species, coords)

        # TODO: Apply symmetry operations (if needed) this should enforce the specified space group

        return structure
