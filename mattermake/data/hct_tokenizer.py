import torch
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from dataclasses import dataclass
from typing import List, Dict, Tuple
import logging
import spglib
from mattermake.utils.hct_wyckoff_mapping import SpaceGroupWyckoffMapping

logger = logging.getLogger(__name__)


@dataclass
class CrystalTokenData:
    """Container for tokenized crystal structure data"""

    sequence: List[int]  # The full token sequence
    segment_ids: List[
        int
    ]  # Identifies token types (composition, space group, lattice, element, etc.)
    masks: Dict[str, List[bool]]  # Masks for dynamic constraint handling

    # Original metadata for reference
    composition: Dict[str, int]  # Element counts in composition
    space_group: int
    lattice_params: List[float]
    atoms: List[Dict]

    def __len__(self):
        return len(self.sequence)


class CrystalTokenizer:
    """Tokenizer for converting crystal structures to token sequences with composition-first approach"""

    # Special tokens
    BOS_TOKEN = 0
    EOS_TOKEN = 1
    PAD_TOKEN = 2
    COMP_SEP_TOKEN = 3

    # Segment IDs
    SEGMENT_SPECIAL = 0
    SEGMENT_COMPOSITION = 1
    SEGMENT_SPACE_GROUP = 2
    SEGMENT_LATTICE = 3
    SEGMENT_ELEMENT = 4
    SEGMENT_WYCKOFF = 5
    SEGMENT_COORDINATE = 6

    def __init__(
        self,
        max_sequence_length: int = 256,
        coordinate_precision: int = 4,
        lattice_bins: int = 100,
        max_element_count: int = 12,  # Maximum count of each element to support
    ):
        self.max_sequence_length = max_sequence_length
        self.coordinate_precision = coordinate_precision
        self.lattice_bins = lattice_bins
        self.max_element_count = max_element_count
        
        # Initialize Wyckoff mapping
        self.wyckoff_mapping = SpaceGroupWyckoffMapping()

        # Initialize vocabulary
        self._init_vocabulary()

    def _init_vocabulary(self):
        """Initialize the token vocabulary with composition tokens"""
        self.vocab = {}
        self.idx_to_token = {}

        # Add special tokens
        self.vocab["<BOS>"] = self.BOS_TOKEN
        self.vocab["<EOS>"] = self.EOS_TOKEN
        self.vocab["<PAD>"] = self.PAD_TOKEN
        self.vocab["<COMP_SEP>"] = self.COMP_SEP_TOKEN

        self.idx_to_token[self.BOS_TOKEN] = "<BOS>"
        self.idx_to_token[self.EOS_TOKEN] = "<EOS>"
        self.idx_to_token[self.PAD_TOKEN] = "<PAD>"
        self.idx_to_token[self.COMP_SEP_TOKEN] = "<COMP_SEP>"

        elements = [
            "H",
            "He",
            "Li",
            "Be",
            "B",
            "C",
            "N",
            "O",
            "F",
            "Ne",
            "Na",
            "Mg",
            "Al",
            "Si",
            "P",
            "S",
            "Cl",
            "Ar",
            "K",
            "Ca",
            "Sc",
            "Ti",
            "V",
            "Cr",
            "Mn",
            "Fe",
            "Co",
            "Ni",
            "Cu",
            "Zn",
            "Ga",
            "Ge",
            "As",
            "Se",
            "Br",
            "Kr",
            "Rb",
            "Sr",
            "Y",
            "Zr",
            "Nb",
            "Mo",
            "Tc",
            "Ru",
            "Rh",
            "Pd",
            "Ag",
            "Cd",
            "In",
            "Sn",
            "Sb",
            "Te",
            "I",
            "Xe",
            "Cs",
            "Ba",
            "La",
            "Ce",
            "Pr",
            "Nd",
            "Pm",
            "Sm",
            "Eu",
            "Gd",
            "Tb",
            "Dy",
            "Ho",
            "Er",
            "Tm",
            "Yb",
            "Lu",
            "Hf",
            "Ta",
            "W",
            "Re",
            "Os",
            "Ir",
            "Pt",
            "Au",
            "Hg",
            "Tl",
            "Pb",
            "Bi",
        ]

        comp_offset = 10  # Leave room for special tokens
        self.composition_tokens = {}

        for i, element in enumerate(elements):
            for count in range(1, self.max_element_count + 1):
                token_name = f"COMP_{element}_{count}"
                token_id = comp_offset + (i * self.max_element_count) + count
                self.vocab[token_name] = token_id
                self.idx_to_token[token_id] = token_name
                self.composition_tokens[(element, count)] = token_id

        # Space group tokens (1-230)
        sg_offset = 1000  # Leave room for composition tokens
        for sg in range(1, 231):
            token_name = f"SG_{sg}"
            token_id = sg + sg_offset
            self.vocab[token_name] = token_id
            self.idx_to_token[token_id] = token_name

        # Element tokens
        element_offset = 1300
        self.element_tokens = {}
        for i, element in enumerate(elements):
            token_id = element_offset + i
            token_name = f"ELEM_{element}"
            self.vocab[token_name] = token_id
            self.idx_to_token[token_id] = token_name
            self.element_tokens[element] = token_id

        # Wyckoff tokens
        wyckoff_offset = 1500
        self.wyckoff_tokens = {}
        wyckoff_letters = "abcdefghijklmnopqrstuvwxyz"
        for i, letter in enumerate(wyckoff_letters):
            token_id = wyckoff_offset + i
            token_name = f"WYCK_{letter}"
            self.vocab[token_name] = token_id
            self.idx_to_token[token_id] = token_name
            self.wyckoff_tokens[letter] = token_id

        # Lattice parameter tokens (binned)
        lattice_offset = 1600
        self.lattice_tokens = {}
        for i in range(self.lattice_bins):
            token_id = lattice_offset + i
            token_name = f"LAT_{i}"
            self.vocab[token_name] = token_id
            self.idx_to_token[token_id] = token_name
            self.lattice_tokens[i] = token_id

        # Coordinate tokens (discretized)
        coord_offset = 1800
        self.coord_tokens = {}
        coord_bins = 10**self.coordinate_precision
        for i in range(coord_bins):
            value = i / coord_bins
            token_id = coord_offset + i
            token_name = f"COORD_{value:.{self.coordinate_precision}f}"
            self.vocab[token_name] = token_id
            self.idx_to_token[token_id] = token_name
            self.coord_tokens[i] = token_id

        # Combined Wyckoff-multiplicity tokens
        wyckoff_mult_offset = 3000
        self.wyckoff_mult_tokens = {}
        for sg in range(1, 231):
            for letter in "abcdefghijklmnopqrstuvwxyz":
                mult = self.wyckoff_mapping.get_wyckoff_multiplicity(sg, letter)
                if mult > 0:  # Valid combination
                    token_name = f"WYCK_{letter}{mult}"
                    token_id = wyckoff_mult_offset + (sg * 100) + ord(letter) - ord('a')
                    self.vocab[token_name] = token_id
                    self.idx_to_token[token_id] = token_name
                    self.wyckoff_mult_tokens[(sg, letter, mult)] = token_id

        self.vocab_size = len(self.vocab)
        logger.info(f"Vocabulary initialized with {self.vocab_size} tokens")

    def tokenize_structure(self, structure: Structure) -> CrystalTokenData:
        """Convert a pymatgen Structure to token sequence with composition first"""
        sga = SpacegroupAnalyzer(structure)
        spacegroup = sga.get_space_group_number()

        try:
            sym_structure = sga.get_symmetrized_structure()

            try:
                lattice = structure.lattice.matrix
                positions = structure.frac_coords
                numbers = [site.specie.Z for site in structure]
                cell = (lattice, positions, numbers)

                dataset = spglib.get_symmetry_dataset(cell, symprec=0.01)

                if dataset is None:
                    logger.info("Warning: Failed to get symmetry dataset from spglib")
                    wyckoff_sites = ["a"] * len(structure)
                else:
                    wyckoff_sites = dataset.wyckoffs
                    # logger.info(f"Found Wyckoff positions: {list(set(wyckoff_sites))}")
            except Exception as e:
                logger.info(f"Warning: spglib Wyckoff determination failed: {e}")
                wyckoff_sites = ["a"] * len(structure)
        except Exception as e:
            logger.info(f"Warning: Structure symmetrization failed: {e}")
            sym_structure = structure
            wyckoff_sites = ["a"] * len(structure)

        lattice = structure.lattice

        tokens = [self.BOS_TOKEN]
        segment_ids = [self.SEGMENT_SPECIAL]

        composition = structure.composition
        composition_dict = {}

        for element, count in sorted(composition.items()):
            element_symbol = str(element.symbol)
            count_int = round(float(count))  # Round to nearest integer
            composition_dict[element_symbol] = count_int

            if (element_symbol, count_int) in self.composition_tokens:
                token = self.composition_tokens[(element_symbol, count_int)]
                tokens.append(token)
                segment_ids.append(self.SEGMENT_COMPOSITION)
            else:
                remaining = count_int
                while remaining > 0:
                    current_count = min(remaining, self.max_element_count)
                    if (element_symbol, current_count) in self.composition_tokens:
                        token = self.composition_tokens[(element_symbol, current_count)]
                        tokens.append(token)
                        segment_ids.append(self.SEGMENT_COMPOSITION)
                    remaining -= current_count

        tokens.append(self.COMP_SEP_TOKEN)
        segment_ids.append(self.SEGMENT_SPECIAL)

        sg_token = self.vocab.get(
            f"SG_{spacegroup}", self.vocab.get("SG_1")
        )  # Default to SG 1 if not found
        tokens.append(sg_token)
        segment_ids.append(self.SEGMENT_SPACE_GROUP)

        for i, param in enumerate(lattice.parameters):
            if i < 3:  # a, b, c
                # Scale to 0-100 Å range and discretize
                bin_idx = min(
                    int(param * self.lattice_bins / 100), self.lattice_bins - 1
                )
            else:  # alpha, beta, gamma
                # Scale to 0-180° range and discretize
                bin_idx = min(
                    int(param * self.lattice_bins / 180), self.lattice_bins - 1
                )

            tokens.append(self.lattice_tokens[bin_idx])
            segment_ids.append(self.SEGMENT_LATTICE)

        site_data = []
        for i, site in enumerate(sym_structure.sites):
            element = site.specie.symbol
            wyckoff = wyckoff_sites[i] if i < len(wyckoff_sites) else "a"
            frac_coords = site.frac_coords
            site_data.append(
                {
                    "element": element,
                    "wyckoff": wyckoff,
                    "coords": frac_coords,
                    "index": i,
                }
            )

        sorted_sites = sorted(site_data, key=lambda x: (x["element"]))

        for site in sorted_sites:
            element_token = self.element_tokens.get(site["element"])
            if element_token:
                tokens.append(element_token)
                segment_ids.append(self.SEGMENT_ELEMENT)

            # Get Wyckoff letter and corresponding multiplicity
            wyckoff_letter = site["wyckoff"]
            # Get multiplicity from mapping
            multiplicity = self.wyckoff_mapping.get_wyckoff_multiplicity(
                spacegroup, wyckoff_letter
            )
            
            # Use combined Wyckoff-multiplicity token
            wyckoff_mult_token = self.wyckoff_mult_tokens.get((spacegroup, wyckoff_letter, multiplicity))
            if wyckoff_mult_token:
                tokens.append(wyckoff_mult_token)
                segment_ids.append(self.SEGMENT_WYCKOFF)
            else:
                # Fallback to just using Wyckoff letter if no combined token exists
                wyckoff_token = self.wyckoff_tokens.get(wyckoff_letter)
                if wyckoff_token:
                    tokens.append(wyckoff_token)
                    segment_ids.append(self.SEGMENT_WYCKOFF)

            coords = site["coords"]
            for coord in coords:
                coord_bin = min(
                    int(coord * 10**self.coordinate_precision),
                    10**self.coordinate_precision - 1,
                )
                tokens.append(self.coord_tokens[coord_bin])
                segment_ids.append(self.SEGMENT_COORDINATE)

        tokens.append(self.EOS_TOKEN)
        segment_ids.append(self.SEGMENT_SPECIAL)

        masks = self._create_constraint_masks(spacegroup, len(tokens))

        atoms_data = [
            {
                "element": site["element"],
                "wyckoff": site["wyckoff"],
                "coords": site["coords"],
            }
            for site in sorted_sites
        ]

        return CrystalTokenData(
            sequence=tokens,
            segment_ids=segment_ids,
            masks=masks,
            composition=composition_dict,
            space_group=spacegroup,
            lattice_params=list(lattice.parameters),
            atoms=atoms_data,
        )

    def _create_constraint_masks(
        self, space_group: int, seq_length: int
    ) -> Dict[str, List[bool]]:
        """Create masks for dynamic constraint handling during generation"""
        # Initialize masks
        masks = {
            "composition_mask": [False] * seq_length,
            "wyckoff_mask": [False] * seq_length,
            "lattice_mask": [False] * seq_length,
            "coordinate_mask": [False] * seq_length,
        }

        # Use the Wyckoff mapping to create dynamic constraints
        allowed_wyckoff = self.wyckoff_mapping.get_allowed_wyckoff_positions(space_group)
        
        # Create a mask for valid Wyckoff positions for this space group
        wyckoff_mask = self.wyckoff_mapping.create_wyckoff_mask(space_group)
        
        # At this point we have the allowed Wyckoff positions and a mask
        # The actual mapping of this to token positions would depend on how
        # we index into the sequence during generation
        # For now, we'll just store this for future use
        self._current_allowed_wyckoff = allowed_wyckoff
        self._current_wyckoff_mask = wyckoff_mask
        
        return masks

    def pad_sequence(
        self, token_data: CrystalTokenData
    ) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Pad token sequence to max_sequence_length"""
        seq_len = len(token_data.sequence)
        padded_seq = token_data.sequence + [self.PAD_TOKEN] * (
            self.max_sequence_length - seq_len
        )
        padded_seg = token_data.segment_ids + [self.SEGMENT_SPECIAL] * (
            self.max_sequence_length - seq_len
        )

        padded_masks = {}
        for mask_name, mask in token_data.masks.items():
            padded_mask = mask + [False] * (self.max_sequence_length - seq_len)
            padded_masks[mask_name] = torch.tensor(padded_mask, dtype=torch.bool)

        return (
            torch.tensor(padded_seq, dtype=torch.long),
            torch.tensor(padded_seg, dtype=torch.long),
            padded_masks,
        )
