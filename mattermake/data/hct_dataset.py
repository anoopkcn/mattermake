import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Dict, Any

from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


class HCTDataset(Dataset):
    """
    PyTorch Dataset for loading pre-processed HCT data.
    Expects data to be saved as individual .pt files, each containing a dictionary
    with keys: 'composition', 'spacegroup', 'lattice', 'atom_types',
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

        log.info(f"Initialized HCTDataset with {len(data_files)} files")
        log.debug(f"START token: {start_token_idx}, END token: {end_token_idx}")

        # --- Validation ---
        if self.start_idx == 0 or self.end_idx == 0:
            log.error("START/END tokens cannot be 0, as 0 is reserved for PAD.")
            raise ValueError("START/END tokens cannot be 0, as 0 is reserved for PAD.")
        if self.start_idx == self.end_idx:
            log.error("START and END tokens must be distinct.")
            raise ValueError("START and END tokens must be distinct.")

    def __len__(self) -> int:
        return len(self.data_files)

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
        filepath = self.data_files[idx]
        try:
            log.debug(f"Loading file: {filepath}")
            data = torch.load(filepath)

            # --- Load and prepare sequences ---
            atom_types = list(data["atom_types"])
            atom_wyckoffs = list(
                data["atom_wyckoffs"]
            )  # Expecting indices >= 1 from regenerated data
            raw_coords = data["atom_coords"]

            # Validate indices are >= 1 (as expected from regenrated data)
            if any(at <= 0 for at in atom_types):
                error_msg = f"File {filepath} contains non-positive atom type indices: {atom_types}. Expected indices >= 1."
                log.error(error_msg)
                raise ValueError(error_msg)

            if any(aw <= 0 for aw in atom_wyckoffs):
                error_msg = f"File {filepath} contains non-positive Wyckoff indices: {atom_wyckoffs}. Expected indices >= 1."
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
            final_data["composition"] = torch.tensor(
                data["composition"], dtype=torch.long
            )
            # Note: Spacegroup is 1-230. May need adjustment later if model expects 0-based indexing.
            final_data["spacegroup"] = torch.tensor(
                [data["spacegroup"]], dtype=torch.long
            )
            final_data["lattice"] = torch.tensor(data["lattice"], dtype=torch.float)
            if final_data["lattice"].shape != (6,):
                error_msg = f"Expected lattice parameters to have shape (6,), but got {final_data['lattice'].shape} in file {filepath}"
                log.error(error_msg)
                raise ValueError(error_msg)

            final_data["filepath"] = str(filepath)

            return final_data
        except Exception as e:
            log.error(f"Error loading or processing file: {filepath}")
            log.error(
                f"Original data keys: {list(data.keys()) if 'data' in locals() else 'N/A'}"
            )
            log.error(
                f"Raw coords: {data.get('atom_coords', 'N/A') if 'data' in locals() else 'N/A'}"
            )
            raise e
