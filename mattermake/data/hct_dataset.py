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
        try:
            # Get the specified structure by index
            file_idx, struct_idx = self.structure_map[idx]
            filepath = self.data_files[file_idx]

            log.debug(
                f"Loading structure {idx} (file: {filepath}, struct_idx: {struct_idx})"
            )
            
            # Get structure data safely with retry logic
            try:
                data = self._get_structure_data(idx)
            except Exception as e:
                log.error(f"Error loading structure data from file {filepath} (struct_idx={struct_idx}): {str(e)}")
                # Create a minimal placeholder structure instead of failing
                return self._create_placeholder_structure(idx, filepath)

            # --- Load and prepare sequences ---
            try:
                atom_types = list(data.get("atom_types", []))
                atom_wyckoffs = list(data.get("atom_wyckoffs", []))
                raw_coords = data.get("atom_coords", [])
            except Exception as e:
                log.error(f"Error extracting basic atom data for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx, filepath)

            # Validate indices are >= 1 (as expected from regenerated data)
            try:
                # Safely check for non-positive values without causing exceptions
                if atom_types and any(at <= 0 for at in atom_types if isinstance(at, (int, float))):
                    log.warning(f"Structure in {filepath} contains non-positive atom type indices: {atom_types}. Will fix automatically.")
                    # Fix atom type indices that are <= 0 by setting them to 1 (minimal valid value)
                    atom_types = [max(1, at) if isinstance(at, (int, float)) else 1 for at in atom_types]

                if atom_wyckoffs and any(aw <= 0 for aw in atom_wyckoffs if isinstance(aw, (int, float))):
                    log.warning(f"Structure in {filepath} contains non-positive Wyckoff indices: {atom_wyckoffs}. Will fix automatically.")
                    # Fix Wyckoff indices that are <= 0 by setting them to 1 (minimal valid value)
                    atom_wyckoffs = [max(1, aw) if isinstance(aw, (int, float)) else 1 for aw in atom_wyckoffs]
            except Exception as e:
                log.error(f"Error validating indices for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx, filepath)

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
                if len(atom_coords_list) != len(atom_types) or len(atom_coords_list) != len(atom_wyckoffs):
                    log.warning(
                        f"Coordinate list length ({len(atom_coords_list)}) doesn't match atom_types ({len(atom_types)}) "
                        f"or atom_wyckoffs ({len(atom_wyckoffs)}) for structure {idx}. Using minimal valid structure."
                    )
                    return self._create_placeholder_structure(idx, filepath)
            except Exception as e:
                log.error(f"Error processing coordinates for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx, filepath)

            # --- Add START/END tokens to atom sequences if configured ---
            try:
                if self.add_start:
                    atom_types.insert(0, self.start_idx)
                    atom_wyckoffs.insert(0, self.start_idx)  # Use same START token index
                    atom_coords_list.insert(0, self.start_coords.tolist())

                if self.add_end:
                    atom_types.append(self.end_idx)
                    atom_wyckoffs.append(self.end_idx)  # Use same END token index
                    atom_coords_list.append(self.end_coords.tolist())
            except Exception as e:
                log.error(f"Error adding start/end tokens for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx, filepath)

            # --- Convert final sequences to Tensors with correct types ---
            try:
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
                # Handle composition tensor safely
                try:
                    comp = data.get("composition", torch.zeros(100, dtype=torch.long))  # Default to vocab size 100
                    if isinstance(comp, torch.Tensor):
                        final_data["composition"] = comp.clone().detach().to(dtype=torch.long)
                    else:
                        final_data["composition"] = torch.tensor(comp, dtype=torch.long)
                except Exception as e:
                    log.warning(f"Error processing composition for structure {idx}: {str(e)}")
                    # Create a zero composition vector as fallback
                    final_data["composition"] = torch.zeros(100, dtype=torch.long)
                    
                # Handle spacegroup safely
                try:
                    sg = data.get("spacegroup", 1)  # Default to space group 1 if missing
                    if isinstance(sg, torch.Tensor):
                        final_data["spacegroup"] = sg.clone().detach().to(dtype=torch.long).view(1)
                    else:
                        final_data["spacegroup"] = torch.tensor([sg], dtype=torch.long)
                except Exception as e:
                    log.warning(f"Error processing spacegroup for structure {idx}: {str(e)}")
                    final_data["spacegroup"] = torch.tensor([1], dtype=torch.long)  # Default to space group 1
                    
                # Handle lattice parameters safely
                try:
                    latt = data.get("lattice", torch.ones(6, dtype=torch.float))  # Default to unit cell if missing
                    if isinstance(latt, torch.Tensor):
                        final_data["lattice"] = latt.clone().detach().to(dtype=torch.float)
                    else:
                        final_data["lattice"] = torch.tensor(latt, dtype=torch.float)
                        
                    if final_data["lattice"].shape != (6,):
                        log.warning(f"Expected lattice parameters to have shape (6,), but got {final_data['lattice'].shape}")
                        final_data["lattice"] = torch.ones(6, dtype=torch.float)  # Use unit cell as fallback
                except Exception as e:
                    log.warning(f"Error processing lattice for structure {idx}: {str(e)}")
                    final_data["lattice"] = torch.ones(6, dtype=torch.float)  # Use unit cell as fallback
                    
                final_data["filepath"] = str(filepath)

                return final_data
            except Exception as e:
                log.error(f"Error during final tensor conversion for structure {idx}: {str(e)}")
                return self._create_placeholder_structure(idx, filepath)
                
        except Exception as e:
            log.error(f"Unhandled error loading structure {idx}: {str(e)}")
            # Create a minimal placeholder structure
            return self._create_placeholder_structure(idx)
    
    def _create_placeholder_structure(self, idx: int, filepath: str = None) -> Dict[str, Any]:
        """Creates a minimal valid structure when there's an error loading real data"""
        try:
            if filepath is None and idx < len(self.structure_map):
                file_idx, _ = self.structure_map[idx]
                filepath = str(self.data_files[file_idx])
            elif filepath is None:
                filepath = "unknown"
                
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
                "atom_wyckoffs": torch.tensor(atom_wyckoffs, dtype=torch.long),
                "atom_coords": torch.tensor(atom_coords, dtype=torch.float),
                "filepath": filepath
            }
        except Exception as e:
            log.error(f"Error creating placeholder structure: {str(e)}")
            # Last resort fallback with minimal tensors
            return {
                "composition": torch.zeros(100, dtype=torch.long),
                "spacegroup": torch.tensor([1], dtype=torch.long),
                "lattice": torch.ones(6, dtype=torch.float),
                "atom_types": torch.tensor([self.start_idx, self.end_idx], dtype=torch.long),
                "atom_wyckoffs": torch.tensor([self.start_idx, self.end_idx], dtype=torch.long),
                "atom_coords": torch.zeros((2, 3), dtype=torch.float),
                "filepath": "error_placeholder"
            }
