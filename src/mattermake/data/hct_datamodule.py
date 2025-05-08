from lightning.pytorch import LightningModule
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from pathlib import Path
from typing import Optional, List, Dict, Any

from mattermake.data.hct_dataset import HCTDataset
from mattermake.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


# Using PAD = 0 scheme, requires indices >= 1 in data
ATOM_TYPE_PAD_IDX = 0
ATOM_WYCKOFF_PAD_IDX = 0
ATOM_COORD_PAD_VAL = 0.0  # Padding for continuous coordinates


def hct_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for HCT data using PAD=0 scheme.
    Handles padding of variable-length atom sequences.

    Args:
        batch (List[Dict[str, Any]]): A list of dictionaries from HCTDataset.__getitem__.

    Returns:
        Dict[str, Any]: A dictionary containing batched and padded tensors.
                        Includes 'atom_mask' for non-padded sequence elements.
    """
    try:
        # Filter out None items that might have been returned from __getitem__ due to errors
        valid_batch = [item for item in batch if item is not None]

        # If we have an empty batch after filtering, return minimal valid structure
        if not valid_batch:
            log.warning("Empty batch after filtering invalid items")
            return {
                "composition": torch.zeros(
                    (0, 100), dtype=torch.long
                ),  # Assuming vocab_size=100
                "spacegroup": torch.zeros((0, 1), dtype=torch.long),
                "lattice": torch.zeros((0, 6), dtype=torch.float),
                "atom_types": torch.zeros((0, 0), dtype=torch.long),
                "atom_wyckoffs": torch.zeros((0, 0), dtype=torch.long),
                "atom_coords": torch.zeros((0, 0, 3), dtype=torch.float),
                "atom_mask": torch.zeros((0, 0), dtype=torch.bool),
                "lengths": torch.zeros((0,), dtype=torch.long),
                "material_ids": [],
            }

        try:
            # Batch fixed-size tensors
            composition_batch = torch.stack(
                [item["composition"] for item in valid_batch]
            )
            spacegroup_batch = torch.stack([item["spacegroup"] for item in valid_batch])
            lattice_batch = torch.stack([item["lattice"] for item in valid_batch])
            material_ids = [item["material_id"] for item in valid_batch]

            # Batch variable-length tensors (Atom sequences)
            atom_types_list = [item["atom_types"] for item in valid_batch]
            atom_wyckoffs_list = [item["atom_wyckoffs"] for item in valid_batch]
            atom_coords_list = [item["atom_coords"] for item in valid_batch]

            # Get sequence lengths BEFORE padding (includes START/END tokens if added)
            lengths = torch.tensor(
                [len(seq) for seq in atom_types_list], dtype=torch.long
            )

            # Pad sequences using PAD=0 for indices
            atom_types_padded = pad_sequence(
                atom_types_list, batch_first=True, padding_value=ATOM_TYPE_PAD_IDX
            )
            atom_wyckoffs_padded = pad_sequence(
                atom_wyckoffs_list, batch_first=True, padding_value=ATOM_WYCKOFF_PAD_IDX
            )
            atom_coords_padded = pad_sequence(
                atom_coords_list, batch_first=True, padding_value=ATOM_COORD_PAD_VAL
            )

            # Create padding mask (True for data/tokens, False for padding)
            # Note: This mask marks PAD tokens (index 0) as False.
            max_len = lengths.max().item() if lengths.numel() > 0 else 0
            # Careful: Need to ensure the mask correctly identifies padding *index* 0, not just position 0
            # Simpler way using lengths:
            atom_mask = (
                torch.arange(max_len)[None, :] < lengths[:, None]
            )  # Shape: (batch_size, max_len)

            return {
                "composition": composition_batch,
                "spacegroup": spacegroup_batch,
                "lattice": lattice_batch,
                "atom_types": atom_types_padded,
                "atom_wyckoffs": atom_wyckoffs_padded,
                "atom_coords": atom_coords_padded,
                "atom_mask": atom_mask,  # Mask indicating non-padded atom entries (True where valid)
                "lengths": lengths,  # Original lengths of atom sequences (incl. START/END)
                "material_ids": material_ids,
            }
        except Exception as e:
            log.error(f"Error in collate_fn inner processing: {str(e)}")
            # Return a minimal but valid batch structure instead of failing
            log.warning("Returning fallback batch structure due to collate error")
            return {
                "composition": torch.zeros(
                    (1, 100), dtype=torch.long
                ),  # Minimal 1-item batch
                "spacegroup": torch.ones((1, 1), dtype=torch.long),  # Space group 1
                "lattice": torch.ones((1, 6), dtype=torch.float),  # Unit cell
                "atom_types": torch.ones((1, 1), dtype=torch.long),  # Single atom
                "atom_wyckoffs": torch.ones((1, 1), dtype=torch.long),  # Single Wyckoff
                "atom_coords": torch.zeros((1, 1, 3), dtype=torch.float),  # Origin
                "atom_mask": torch.ones((1, 1), dtype=torch.bool),  # Valid atom
                "lengths": torch.ones((1,), dtype=torch.long),  # Length 1
                "material_ids": ["fallback_structure"],
            }
    except Exception as e:
        log.error(f"Critical error in collate_fn outer processing: {str(e)}")
        # Return an absolute minimal structure for the entire batch
        return {
            "composition": torch.zeros((1, 100), dtype=torch.long),
            "spacegroup": torch.ones((1, 1), dtype=torch.long),
            "lattice": torch.ones((1, 6), dtype=torch.float),
            "atom_types": torch.ones((1, 1), dtype=torch.long),
            "atom_wyckoffs": torch.ones((1, 1), dtype=torch.long),
            "atom_coords": torch.zeros((1, 1, 3), dtype=torch.float),
            "atom_mask": torch.ones((1, 1), dtype=torch.bool),
            "lengths": torch.ones((1,), dtype=torch.long),
            "material_ids": ["emergency_fallback"],
        }


class HCTDataModule(LightningModule):
    """
    PyTorch Lightning DataModule for HCT using PAD=0, START=-1, END=-2 scheme.

    Loads pre-processed data saved as .pt files (expecting indices >= 1).
    Handles batching and padding of variable-length atom sequences.
    """

    def on_exception(self, exception):
        """Handle exceptions gracefully during DataModule operations"""
        log.error(f"Exception in DataModule: {str(exception)}")
        # We can add specific exception handling here if needed

    def __init__(
        self,
        processed_data_dir: str = "data/hct_data",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        # --- Atom Sequence Token Config ---
        add_atom_start_end_tokens: bool = True,
        atom_start_token_idx: int = -1,  # Default START = -1
        atom_end_token_idx: int = -2,  # Default END = -2
        # --- Vocabulary/Info Placeholders ---
        element_vocab_size: int = 100,  # Adjust based on actual elements + special tokens if needed
        # Max Wyckoff index will depend on the space group with the most sites + special tokens
        # max_wyckoff_index: int = ?, # Determine this from data or sg_to_symbols
        spacegroup_vocab_size: int = 230 + 1,  # 1-230 + potentially 0 if needed
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        log.info(
            f"Initializing HCTDataModule with batch_size={batch_size}, data_dir={processed_data_dir}"
        )

        # Set file paths for the single-file approach
        self.data_dir = Path(self.hparams.processed_data_dir)
        self.train_file = self.data_dir / "train.pt"
        self.val_file = self.data_dir / "val.pt"
        self.test_file = self.data_dir / "test.pt"

        self.train_dataset: Optional[HCTDataset] = None
        self.val_dataset: Optional[HCTDataset] = None
        self.test_dataset: Optional[HCTDataset] = None
        self.predict_dataset: Optional[HCTDataset] = None

        # Store dataset args for instantiation in setup()
        self.dataset_kwargs = {
            "add_atom_start_token": self.hparams.add_atom_start_end_tokens,
            "add_atom_end_token": self.hparams.add_atom_start_end_tokens,
            "start_token_idx": self.hparams.atom_start_token_idx,
            "end_token_idx": self.hparams.atom_end_token_idx,
            # start/end coords use defaults in HCTDataset
        }
        # Padding index is now consistently 0
        self.padding_idx = 0
        log.info("HCTDataModule initialization complete")

    def prepare_data(self):
        if not self.data_dir.is_dir():
            log.error(f"Processed data directory not found: {self.data_dir}")
            raise FileNotFoundError(
                f"Processed data directory not found: {self.data_dir}"
            )

        # Check for existence of required files
        required_files = [self.train_file, self.val_file, self.test_file]
        missing_files = [f for f in required_files if not f.exists()]
        if missing_files:
            missing_list = ", ".join(str(f) for f in missing_files)
            log.warning(f"Missing data files: {missing_list}")

    def setup(self, stage: Optional[str] = None):
        log.info(f"Setting up HCTDataModule for stage: {stage}")
        if stage == "fit" or stage is None:
            if self.train_file.exists():
                self.train_dataset = HCTDataset(self.train_file, **self.dataset_kwargs)
                log.info(
                    f"Loaded {len(self.train_dataset)} training samples from {self.train_file}"
                )
            else:
                log.warning(f"Training file not found: {self.train_file}")
                self.train_dataset = None

            if self.val_file.exists():
                self.val_dataset = HCTDataset(self.val_file, **self.dataset_kwargs)
                log.info(
                    f"Loaded {len(self.val_dataset)} validation samples from {self.val_file}"
                )
            else:
                log.warning(f"Validation file not found: {self.val_file}")
                self.val_dataset = None

        if stage == "test" or stage is None:
            if self.test_file.exists():
                self.test_dataset = HCTDataset(self.test_file, **self.dataset_kwargs)
                log.info(
                    f"Loaded {len(self.test_dataset)} test samples from {self.test_file}"
                )
            else:
                log.warning(f"Test file not found: {self.test_file}")
                self.test_dataset = None

        if stage == "predict":
            # Use test dataset for prediction by default
            if self.test_file.exists():
                self.predict_dataset = HCTDataset(self.test_file, **self.dataset_kwargs)
                log.info(
                    f"Loaded {len(self.predict_dataset)} prediction samples from {self.test_file}"
                )
            else:
                log.warning(f"Test file not found for prediction: {self.test_file}")
                self.predict_dataset = None

    def _get_dataloader(
        self, dataset: Optional[HCTDataset], shuffle: bool
    ) -> DataLoader:
        if not dataset:
            msg = f"{'Train' if shuffle else 'Validation/Test/Predict'} dataset not available. Check data paths and setup stage."
            log.error(msg)
            raise ValueError(msg)

        log.debug(
            f"Creating dataloader with batch_size={self.hparams.batch_size}, shuffle={shuffle}"
        )
        return DataLoader(
            dataset,
            batch_size=self.hparams.batch_size,
            shuffle=shuffle,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            collate_fn=hct_collate_fn,  # Use the updated collate function
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        log.debug("Creating train dataloader")
        return self._get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        log.debug("Creating validation dataloader")
        return self._get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        log.debug("Creating test dataloader")
        # Use fewer workers for testing to reduce risk of worker crashes
        original_workers = self.hparams.num_workers
        test_workers = (
            min(self.hparams.num_workers, 2) if self.hparams.num_workers > 0 else 0
        )
        self.hparams.num_workers = test_workers

        dataloader = self._get_dataloader(self.test_dataset, shuffle=False)

        # Restore original worker count
        self.hparams.num_workers = original_workers
        return dataloader

    def predict_dataloader(self) -> DataLoader:
        log.debug("Creating predict dataloader")
        return self._get_dataloader(self.predict_dataset, shuffle=False)
