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
    'atom_wyckoffs', 'atom_coords'.

    Optionally adds START and END tokens to the atom sequences using a
    unified scheme (PAD=0, START=-1, END=-2).
    Requires pre-processed data where Wyckoff indices start from 1.
    """

    def __init__(
        self,
        data_files: list[Path],
        add_atom_start_token: bool = True,
        add_atom_end_token: bool = True,
        start_token_idx: int = -1,  # Default START = -1
        end_token_idx: int = -2,  # Default END = -2
        # Define placeholder values for Coords corresponding to START/END
        start_coords: list[float] = [0.0, 0.0, 0.0],  # Placeholder coords
        end_coords: list[float] = [0.0, 0.0, 0.0],
        preload_data: bool = False,  # Whether to preload all data at initialization
    ):
        """
        Args:
            data_files (list[Path]): List of paths to the pre-processed .pt files.
            add_atom_start_token (bool): Prepend START token to atom sequences.
            add_atom_end_token (bool): Append END token to atom sequences.
            start_token_idx (int): Index used for the START token (-1).
            end_token_idx (int): Index used for the END token (-2).
            start_coords (list[float]): Coords used for the START atom position.
            end_coords (list[float]): Coords used for the END atom position.
            preload_data (bool): If True, load all data at init time; otherwise, load on-demand.
        """
        super().__init__()
        self.data_files = data_files
        if not self.data_files:
            log.error("No data files provided to HCTDataset.")
            raise ValueError("No data files provided to HCTDataset.")

        # Store token config
        self.add_start = add_atom_start_token
        self.add_end = add_atom_end_token
        self.start_idx = start_token_idx
        self.end_idx = end_token_idx
        # START/END for Wyckoff indices will use the same start_idx/end_idx
        self.start_coords = torch.tensor(start_coords, dtype=torch.float)
        self.end_coords = torch.tensor(end_coords, dtype=torch.float)
        self.preload_data = preload_data

        # --- Validation ---
        if self.start_idx == 0 or self.end_idx == 0:
            log.error("START/END tokens cannot be 0, as 0 is reserved for PAD.")
            raise ValueError("START/END tokens cannot be 0, as 0 is reserved for PAD.")
        if self.start_idx == self.end_idx:
            log.error("START and END tokens must be distinct.")
            raise ValueError("START and END tokens must be distinct.")

        # Load metadata about the files to support fast indexing
        # Each entry is (file_index, structure_index_within_file)
        self.structure_map = []
        self.total_structures = 0

        # If preloading, we'll store all structures here
        self.preloaded_data = [] if preload_data else None

        # Scan files to build index mapping
        for file_idx, filepath in enumerate(self.data_files):
            try:
                # Load just to count structures
                data = torch.load(filepath)

                # Handle both single-structure and multi-structure files
                if isinstance(data, list):
                    # Batch file with multiple structures
                    num_structures = len(data)
                    for struct_idx in range(num_structures):
                        self.structure_map.append((file_idx, struct_idx))

                    # Preload if requested
                    if preload_data:
                        self.preloaded_data.extend(data)
                else:
                    # Single structure file (as a dictionary)
                    self.structure_map.append((file_idx, None))

                    # Preload if requested
                    if preload_data:
                        self.preloaded_data.append(data)

                self.total_structures += num_structures if isinstance(data, list) else 1

            except Exception as e:
                log.error(f"Error scanning file {filepath}: {str(e)}")
                # Skip problematic files without failing
                continue

        log.info(
            f"Initialized HCTDataset with {len(data_files)} files containing {self.total_structures} structures"
        )
        log.debug(f"START token: {start_token_idx}, END token: {end_token_idx}")
        if preload_data:
            log.info(f"Preloaded {len(self.preloaded_data)} structures into memory")

    def __len__(self) -> int:
        return len(self.structure_map)

    def _get_structure_data(self, idx: int) -> Dict[str, Any]:
        """
        Get raw structure data from either preloaded cache or by loading from file.

        Args:
            idx: The index of the structure in the dataset.

        Returns:
            Raw structure data dictionary
        """
        # Look up which file and position this structure is in
        file_idx, struct_idx = self.structure_map[idx]
        filepath = self.data_files[file_idx]

        # Return from preloaded data if available
        if self.preload_data:
            return self.preloaded_data[idx]

        # Otherwise load from file
        try:
            data = torch.load(filepath)

            # If it's a batch file, extract the specific structure
            if isinstance(data, list):
                return data[struct_idx]
            else:
                return data

        except Exception as e:
            log.error(f"Error loading data from {filepath}: {str(e)}")
            raise

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
                - 'atom_wyckoffs': (seq_len,) tensor of Wyckoff/token indices (long). Valid >= 1.
                - 'atom_coords': (seq_len, 3) tensor of coordinates (float)
                - 'filepath': Original path of the loaded file (str)
        """
        # Get the specified structure by index
        file_idx, struct_idx = self.structure_map[idx]
        filepath = self.data_files[file_idx]

        try:
            log.debug(
                f"Loading structure {idx} (file: {filepath}, struct_idx: {struct_idx})"
            )
            data = self._get_structure_data(idx)

            # --- Load and prepare sequences ---
            atom_types = list(data["atom_types"])
            atom_wyckoffs = list(
                data["atom_wyckoffs"]
            )  # Expecting indices >= 1 from regenerated data
            raw_coords = data["atom_coords"]

            # Validate indices are >= 1 (as expected from regenrated data)
            if any(at <= 0 for at in atom_types):
                error_msg = f"Structure in {filepath} contains non-positive atom type indices: {atom_types}. Expected indices >= 1."
                log.error(error_msg)
                raise ValueError(error_msg)

            if any(aw <= 0 for aw in atom_wyckoffs):
                error_msg = f"Structure in {filepath} contains non-positive Wyckoff indices: {atom_wyckoffs}. Expected indices >= 1."
                log.error(error_msg)
                raise ValueError(error_msg)

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

            # --- Add START/END tokens to atom sequences if configured ---
            if self.add_start:
                atom_types.insert(0, self.start_idx)
                atom_wyckoffs.insert(0, self.start_idx)  # Use same START token index
                atom_coords_list.insert(0, self.start_coords.tolist())

            if self.add_end:
                atom_types.append(self.end_idx)
                atom_wyckoffs.append(self.end_idx)  # Use same END token index
                atom_coords_list.append(self.end_coords.tolist())

            # --- Convert final sequences to Tensors with correct types ---
            final_data = {}
            final_data["atom_types"] = torch.tensor(atom_types, dtype=torch.long)
            final_data["atom_wyckoffs"] = torch.tensor(atom_wyckoffs, dtype=torch.long)
            if atom_coords_list:
                final_data["atom_coords"] = torch.tensor(
                    atom_coords_list, dtype=torch.float
                )
            else:
                final_data["atom_coords"] = torch.empty((0, 3), dtype=torch.float)

            # --- Process fixed components ---
            # Handle composition tensor properly to avoid warnings
            comp = data["composition"]
            if isinstance(comp, torch.Tensor):
                final_data["composition"] = comp.clone().detach().to(dtype=torch.long)
            else:
                final_data["composition"] = torch.tensor(comp, dtype=torch.long)
                
            # Handle spacegroup
            sg = data["spacegroup"]
            if isinstance(sg, torch.Tensor):
                final_data["spacegroup"] = sg.clone().detach().to(dtype=torch.long).view(1)
            else:
                final_data["spacegroup"] = torch.tensor([sg], dtype=torch.long)
                
            # Handle lattice parameters
            latt = data["lattice"]
            if isinstance(latt, torch.Tensor):
                final_data["lattice"] = latt.clone().detach().to(dtype=torch.float)
            else:
                final_data["lattice"] = torch.tensor(latt, dtype=torch.float)
            if final_data["lattice"].shape != (6,):
                error_msg = f"Expected lattice parameters to have shape (6,), but got {final_data['lattice'].shape}"
                log.error(error_msg)
                raise ValueError(error_msg)

            final_data["filepath"] = str(filepath)

            return final_data
        except Exception as e:
            log.error(f"Error loading or processing structure {idx}: {str(e)}")
            if "data" in locals():
                log.error(
                    f"Original data keys: {list(data.keys()) if hasattr(data, 'keys') else 'Not a dictionary'}"
                )
                if isinstance(data, dict) and "atom_coords" in data:
                    log.error(f"Raw coords: {data.get('atom_coords')}")
            raise e
