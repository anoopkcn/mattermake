import pytorch_lightning as pl
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
    # Batch fixed-size tensors
    composition_batch = torch.stack([item["composition"] for item in batch])
    spacegroup_batch = torch.stack([item["spacegroup"] for item in batch])
    lattice_batch = torch.stack([item["lattice"] for item in batch])
    filepaths = [item["filepath"] for item in batch]

    # Batch variable-length tensors (Atom sequences)
    atom_types_list = [item["atom_types"] for item in batch]
    atom_wyckoffs_list = [item["atom_wyckoffs"] for item in batch]
    atom_coords_list = [item["atom_coords"] for item in batch]

    # Get sequence lengths BEFORE padding (includes START/END tokens if added)
    lengths = torch.tensor([len(seq) for seq in atom_types_list], dtype=torch.long)

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
    # One way: Check if the padded value is NOT the padding index.
    # atom_mask = (atom_types_padded != ATOM_TYPE_PAD_IDX) # Check against padding value
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
        "filepaths": filepaths,
    }


class HCTDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for HCT using PAD=0, START=-1, END=-2 scheme.

    Loads pre-processed data saved as .pt files (expecting indices >= 1).
    Handles batching and padding of variable-length atom sequences.
    """

    def __init__(
        self,
        processed_data_dir: str = "data/processed/hct",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        train_subdir: str = "train",
        val_subdir: str = "val",
        test_subdir: str = "test",
        file_extension: str = ".pt",
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

        self.data_dir = Path(self.hparams.processed_data_dir)
        self.train_path = self.data_dir / self.hparams.train_subdir
        self.val_path = self.data_dir / self.hparams.val_subdir
        self.test_path = self.data_dir / self.hparams.test_subdir

        self.train_dataset: Optional[HCTDataset] = None
        self.val_dataset: Optional[HCTDataset] = None
        self.test_dataset: Optional[HCTDataset] = None
        self.predict_dataset: Optional[HCTDataset] = None

        self.file_glob = f"*{self.hparams.file_extension}"

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
        for subdir_path in [self.train_path, self.val_path, self.test_path]:
            if not subdir_path.is_dir():
                log.warning(f"Data directory not found: {subdir_path}")

    def setup(self, stage: Optional[str] = None):
        log.info(f"Setting up HCTDataModule for stage: {stage}")
        if stage == "fit" or stage is None:
            train_files = sorted(list(self.train_path.glob(self.file_glob)))
            val_files = sorted(list(self.val_path.glob(self.file_glob)))
            if not train_files:
                log.warning(
                    f"No training files found in {self.train_path} matching {self.file_glob}"
                )
            if not val_files:
                log.warning(
                    f"No validation files found in {self.val_path} matching {self.file_glob}"
                )
            self.train_dataset = (
                HCTDataset(train_files, **self.dataset_kwargs) if train_files else None
            )
            self.val_dataset = (
                HCTDataset(val_files, **self.dataset_kwargs) if val_files else None
            )
            log.info(
                f"Loaded {len(self.train_dataset) if self.train_dataset else 0} training samples."
            )
            log.info(
                f"Loaded {len(self.val_dataset) if self.val_dataset else 0} validation samples."
            )

        if stage == "test" or stage is None:
            test_files = sorted(list(self.test_path.glob(self.file_glob)))
            if not test_files:
                log.warning(
                    f"No test files found in {self.test_path} matching {self.file_glob}"
                )
            self.test_dataset = (
                HCTDataset(test_files, **self.dataset_kwargs) if test_files else None
            )
            log.info(
                f"Loaded {len(self.test_dataset) if self.test_dataset else 0} test samples."
            )

        if stage == "predict":
            predict_files = sorted(
                list(self.test_path.glob(self.file_glob))
            )  # Or dedicated predict path
            if not predict_files:
                log.warning(
                    f"No prediction files found in {self.test_path} matching {self.file_glob}"
                )
            self.predict_dataset = (
                HCTDataset(predict_files, **self.dataset_kwargs)
                if predict_files
                else None
            )
            log.info(
                f"Loaded {len(self.predict_dataset) if self.predict_dataset else 0} prediction samples."
            )

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
        return self._get_dataloader(self.test_dataset, shuffle=False)

    def predict_dataloader(self) -> DataLoader:
        log.debug("Creating predict dataloader")
        return self._get_dataloader(self.predict_dataset, shuffle=False)
