import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any

from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HCTDataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed HCT data.
    Supports both individual structure files and batch files containing multiple structures.
    Each structure is a dictionary with keys: 'composition', 'spacegroup', 'lattice', 'atom_types',
    'wyckoff', 'atom_coords'.

    Optionally adds START and END tokens to the atom sequences using a
    unified scheme (PAD=0, START=-1, END=-2).
    Requires pre-processed data where Wyckoff indices start from 1.
    """

    def __init__(
        self,
        data_file: Path,
        add_atom_start_token: bool = True,
        add_atom_end_token: bool = True,
        start_token_idx: int = -1,  # DEFAULT START = -1
        end_token_idx: int = -2,  # DEFAULT END = -2
        # Define placeholder values for Coords corresponding to START/END
        start_coords: list[float] = [0.0, 0.0, 0.0],  # Placeholder coords
        end_coords: list[float] = [0.0, 0.0, 0.0],
    ):
        """
        Args:
            data_file (Path): Path to a single .pt file containing all structures.
            add_atom_start_token (bool): Prepend START token to atom sequences.
            add_atom_end_token (bool): Append END token to atom sequences.
            start_token_idx (int): Index used for the START token (-1).
            end_token_idx (int): Index used for the END token (-2).
            start_coords (list[float]): Coords used for the START atom position.
            end_coords (list[float]): Coords used for the END atom position.
        """
        super().__init__()
        self.data_file = data_file
        if not self.data_file.exists():
            log.error(f"Data file {data_file} does not exist.")
            raise FileNotFoundError(f"Data file {data_file} does not exist.")

        # Store token config
        self.add_start = add_atom_start_token
        self.add_end = add_atom_end_token
        self.start_idx = start_token_idx
        self.end_idx = end_token_idx
        # START/END for Wyckoff indices will use the same start_idx/end_idx
        self.start_coords = torch.tensor(start_coords, dtype=torch.float)
        self.end_coords = torch.tensor(end_coords, dtype=torch.float)

        # For compositional purposes
        self.hparams = {"element_vocab_size": 100}

        # --- Validation ---
        if self.start_idx == 0 or self.end_idx == 0:
            log.error("START/END tokens cannot be 0, as 0 is reserved for PAD.")
            raise ValueError("START/END tokens cannot be 0, as 0 is reserved for PAD.")
        if self.start_idx == self.end_idx:
            log.error("START and END tokens must be distinct.")
            raise ValueError("START and END tokens must be distinct.")

        # Load all structures into memory
        log.info(f"Loading structures from {data_file}...")
        try:
            self.structures = torch.load(self.data_file)
            self.total_structures = len(self.structures)
            log.info(
                f"Successfully loaded {self.total_structures} structures from {data_file}"
            )
        except Exception as e:
            log.error(f"Error loading data file {data_file}: {str(e)}")
            # Initialize with empty list as fallback
            self.structures = []
            self.total_structures = 0

        log.debug(f"START token: {start_token_idx}, END token: {end_token_idx}")

    def __len__(self) -> int:
        return self.total_structures

    def _get_structure_data(self, idx: int) -> Dict[str, Any]:
        """
        Get raw structure data from the preloaded structures list.

        Args:
            idx: The index of the structure in the dataset.

        Returns:
            Raw structure data dictionary
        """
        if idx < 0 or idx >= self.total_structures:
            log.error(f"Index {idx} out of bounds (0-{self.total_structures - 1})")
            return self._create_placeholder_structure(idx)

        # Return structure directly from the in-memory list
        return self.structures[idx]

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Loads and returns a single pre-processed crystal structure data point,
        optionally adding START/END tokens to atom sequences.

        Returns:
            Dict[str, Any]: A dictionary containing tensors for:
                - 'composition': (vocab_size,) tensor of int counts (long)
                - 'spacegroup': (1,) tensor containing the space group number (long) - needs adjustment if using 0-based indexing later
                - 'lattice': (6,) tensor containing lattice parameters (float)
                - 'atom_types': (seq_len,) tensor of atomic/token indices (long). Valid >= 1.
                - 'wyckoff': (seq_len,) tensor of Wyckoff/token indices (long). Valid >= 1.
                - 'atom_coords': (seq_len, 3) tensor of coordinates (float)
                - 'material_id': Original material ID (str)
        """
        try:
            log.debug(f"Loading structure {idx}")

            # Get structure data safely with retry logic
            try:
                data = self._get_structure_data(idx)
            except Exception as e:
                log.error(f"Error loading structure data for index {idx}: {str(e)}")
                # Create a minimal placeholder structure instead of failing
                return self._create_placeholder_structure(idx)

            # --- Load and prepare sequences ---
            try:
                atom_types = list(data.get("atom_types", []))
                atom_wyckoffs = list(data.get("wyckoff", []))
                raw_coords = data.get("atom_coords", [])
            except Exception as e:
                log.error(
                    f"Error extracting basic atom data for structure {idx}: {str(e)}"
                )
                return self._create_placeholder_structure(idx)

            # Validate indices are >= 1 (as expected from regenerated data)
            try:
                # Safely check for non-positive values without causing exceptions
                if atom_types and any(
                    at <= 0 for at in atom_types if isinstance(at, (int, float))
                ):
                    log.warning(
                        f"Structure at index {idx} contains non-positive atom type indices: {atom_types}. Will fix automatically."
                    )
                    # Fix atom type indices that are <= 0 by setting them to 1 (minimal valid value)
                    atom_types = [
                        max(1, at) if isinstance(at, (int, float)) else 1
                        for at in atom_types
                    ]

                if atom_wyckoffs and any(
                    aw <= 0 for aw in atom_wyckoffs if isinstance(aw, (int, float))
                ):
                    log.warning(
                        f"Structure at index {idx} contains non-positive Wyckoff indices: {atom_wyckoffs}. Will fix automatically."
                    )
                    # Fix Wyckoff indices that are <= 0 by setting them to 1 (minimal valid value)
                    atom_wyckoffs = [
                        max(1, aw) if isinstance(aw, (int, float)) else 1
                        for aw in atom_wyckoffs
                    ]
            except Exception as e:
                log.error(f"Error validating indices for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx)

            # Process coordinates safely
            try:
                if isinstance(raw_coords, torch.Tensor):
                    raw_coords = raw_coords.tolist()

                if not raw_coords or len(raw_coords) == 0:
                    atom_coords_list = []
                elif isinstance(raw_coords[0], (float, int)):
                    atom_coords_list = [
                        raw_coords[i : i + 3] for i in range(0, len(raw_coords), 3)
                    ]
                else:
                    atom_coords_list = list(raw_coords)

                # Check if coordinates list lengths match atom types/wyckoffs
                if len(atom_coords_list) != len(atom_types) or len(
                    atom_coords_list
                ) != len(atom_wyckoffs):
                    log.warning(
                        f"Coordinate list length ({len(atom_coords_list)}) doesn't match atom_types ({len(atom_types)}) "
                        f"or wyckoff ({len(atom_wyckoffs)}) for structure {idx}. Using minimal valid structure."
                    )
                    return self._create_placeholder_structure(idx)
            except Exception as e:
                log.error(f"Error processing coordinates for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx)

            # --- Add START/END tokens to atom sequences if configured ---
            try:
                if self.add_start:
                    atom_types.insert(0, self.start_idx)
                    atom_wyckoffs.insert(
                        0, self.start_idx
                    )  # Use same START token index
                    atom_coords_list.insert(0, self.start_coords.tolist())

                if self.add_end:
                    atom_types.append(self.end_idx)
                    atom_wyckoffs.append(self.end_idx)  # Use same END token index
                    atom_coords_list.append(self.end_coords.tolist())
            except Exception as e:
                log.error(
                    f"Error adding start/end tokens for structure {idx}: {str(e)}"
                )
                return self._create_placeholder_structure(idx)

            # --- Convert final sequences to Tensors with correct types ---
            try:
                final_data = {}
                final_data["atom_types"] = torch.tensor(atom_types, dtype=torch.long)
                final_data["wyckoff"] = torch.tensor(atom_wyckoffs, dtype=torch.long)

                if atom_coords_list:
                    final_data["atom_coords"] = torch.tensor(
                        atom_coords_list, dtype=torch.float
                    )
                else:
                    final_data["atom_coords"] = torch.empty((0, 3), dtype=torch.float)

                # --- Process fixed components ---
                # Handle composition tensor safely
                try:
                    comp = data.get(
                        "composition", torch.zeros(100, dtype=torch.long)
                    )  # Default to vocab size 100
                    if isinstance(comp, torch.Tensor):
                        final_data["composition"] = (
                            comp.clone().detach().to(dtype=torch.long)
                        )
                    else:
                        final_data["composition"] = torch.tensor(comp, dtype=torch.long)
                except Exception as e:
                    log.warning(
                        f"Error processing composition for structure {idx}: {str(e)}"
                    )
                    # Create a zero composition vector as fallback
                    final_data["composition"] = torch.zeros(100, dtype=torch.long)

                # Handle spacegroup safely
                try:
                    sg = data.get(
                        "spacegroup", 1
                    )  # Default to space group 1 if missing
                    if isinstance(sg, torch.Tensor):
                        final_data["spacegroup"] = (
                            sg.clone().detach().to(dtype=torch.long).view(1)
                        )
                    else:
                        final_data["spacegroup"] = torch.tensor([sg], dtype=torch.long)
                except Exception as e:
                    log.warning(
                        f"Error processing spacegroup for structure {idx}: {str(e)}"
                    )
                    final_data["spacegroup"] = torch.tensor(
                        [1], dtype=torch.long
                    )  # Default to space group 1

                # Handle lattice parameters safely
                try:
                    latt = data.get(
                        "lattice", torch.ones(6, dtype=torch.float)
                    )  # Default to unit cell if missing
                    if isinstance(latt, torch.Tensor):
                        final_data["lattice"] = (
                            latt.clone().detach().to(dtype=torch.float)
                        )
                    else:
                        final_data["lattice"] = torch.tensor(latt, dtype=torch.float)

                    if final_data["lattice"].shape != (6,):
                        log.warning(
                            f"Expected lattice parameters to have shape (6,), but got {final_data['lattice'].shape}"
                        )
                        final_data["lattice"] = torch.ones(
                            6, dtype=torch.float
                        )  # Use unit cell as fallback
                except Exception as e:
                    log.warning(
                        f"Error processing lattice for structure {idx}: {str(e)}"
                    )
                    final_data["lattice"] = torch.ones(
                        6, dtype=torch.float
                    )  # Use unit cell as fallback

                final_data["material_id"] = f"structure_{idx}"

                return final_data
            except Exception as e:
                log.error(
                    f"Error during final tensor conversion for structure {idx}: {str(e)}"
                )
                return self._create_placeholder_structure(idx)

        except Exception as e:
            log.error(f"Unhandled error loading structure {idx}: {str(e)}")
            # Create a minimal placeholder structure
            return self._create_placeholder_structure(idx)

    def _create_placeholder_structure(self, idx: int) -> Dict[str, Any]:
        """Creates a minimal valid structure when there's an error loading real data"""
        try:
            # Create empty default with correct shapes and START/END tokens
            atom_types = []
            atom_wyckoffs = []
            atom_coords = []

            if self.add_start:
                atom_types.append(self.start_idx)
                atom_wyckoffs.append(self.start_idx)
                atom_coords.append(self.start_coords.tolist())

            if self.add_end:
                atom_types.append(self.end_idx)
                atom_wyckoffs.append(self.end_idx)
                atom_coords.append(self.end_coords.tolist())

            # Use a fixed vocab size of 100 for composition vector
            vocab_size = 100

            return {
                "composition": torch.zeros(vocab_size, dtype=torch.long),
                "spacegroup": torch.tensor([1], dtype=torch.long),
                "lattice": torch.ones(6, dtype=torch.float),
                "atom_types": torch.tensor(atom_types, dtype=torch.long),
                "wyckoff": torch.tensor(atom_wyckoffs, dtype=torch.long),
                "atom_coords": torch.tensor(atom_coords, dtype=torch.float),
                "material_id": f"placeholder_{idx}",
            }
        except Exception as e:
            log.error(f"Error creating placeholder structure: {str(e)}")
            # Last resort fallback with minimal tensors
            return {
                "composition": torch.zeros(100, dtype=torch.long),
                "spacegroup": torch.tensor([1], dtype=torch.long),
                "lattice": torch.ones(6, dtype=torch.float),
                "atom_types": torch.tensor(
                    [self.start_idx, self.end_idx], dtype=torch.long
                ),
                "wyckoff": torch.tensor(
                    [self.start_idx, self.end_idx], dtype=torch.long
                ),
                "atom_coords": torch.zeros((2, 3), dtype=torch.float),
                "material_id": "error_placeholder",
            }
