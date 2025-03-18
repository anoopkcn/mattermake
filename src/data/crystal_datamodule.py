import os
from typing import Optional, List, Dict

import torch
from torch.utils.data import DataLoader
import lightning.pytorch as pl

from src.data.components.crystal_dataset import CrystalDataset
from src.data.components.transforms import (
    NormalizePositions,
    OneHotAtomTypes,
    RandomRotation,
    ComposeTransforms
)


class CrystalDataModule(pl.LightningDataModule):
    """
    LightningDataModule for crystal structure datasets.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        train_file: str = "train.pt",
        val_file: str = "val.pt",
        test_file: str = "test.pt",
        batch_size: int = 32,
        num_workers: int = 4,
        max_atoms: int = 80,
        num_atom_types: int = 95,
        include_properties: bool = True,
        use_data_augmentation: bool = True,
        pin_memory: bool = True,
    ):
        """
        Initialize the crystal data module.

        Args:
            data_dir: Directory containing the data files
            train_file: Name of the training data file
            val_file: Name of the validation data file
            test_file: Name of the test data file
            batch_size: Batch size for data loading
            num_workers: Number of workers for data loading
            max_atoms: Maximum number of atoms in a structure
            num_atom_types: Number of different atom types
            include_properties: Whether to include property data
            use_data_augmentation: Whether to use data augmentation
            pin_memory: Whether to pin memory for faster GPU transfer
        """
        super().__init__()
        self.save_hyperparameters()

        self.data_dir = data_dir
        self.train_file = os.path.join(data_dir, train_file)
        self.val_file = os.path.join(data_dir, val_file)
        self.test_file = os.path.join(data_dir, test_file)

        self.batch_size = batch_size
        self.num_workers = num_workers
        self.max_atoms = max_atoms
        self.num_atom_types = num_atom_types
        self.include_properties = include_properties
        self.pin_memory = pin_memory

        # Define transforms
        self.base_transforms = ComposeTransforms([
            NormalizePositions(),
            OneHotAtomTypes(num_atom_types=num_atom_types)
        ])

        self.train_transforms = ComposeTransforms([
            NormalizePositions(),
            OneHotAtomTypes(num_atom_types=num_atom_types),
            RandomRotation() if use_data_augmentation else lambda x: x
        ])

        # Placeholders for datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self):
        """Check if files exist."""
        for file_path in [self.train_file, self.val_file]:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Data file not found: {file_path}")

    def setup(self, stage: Optional[str] = None):
        """Set up datasets based on stage."""
        # Assign train/val datasets for use in dataloaders
        if stage == "fit" or stage is None:
            self.train_dataset = CrystalDataset(
                file_path=self.train_file,
                max_atoms=self.max_atoms,
                include_properties=self.include_properties,
                transform=self.train_transforms
            )

            self.val_dataset = CrystalDataset(
                file_path=self.val_file,
                max_atoms=self.max_atoms,
                include_properties=self.include_properties,
                transform=self.base_transforms
            )

        # Assign test dataset for use in dataloader(s)
        if stage == "test" or stage is None:
            if os.path.exists(self.test_file):
                self.test_dataset = CrystalDataset(
                    file_path=self.test_file,
                    max_atoms=self.max_atoms,
                    include_properties=self.include_properties,
                    transform=self.base_transforms
                )

    def train_dataloader(self):
        """Get train dataloader."""
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def val_dataloader(self):
        """Get validation dataloader."""
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def test_dataloader(self):
        """Get test dataloader."""
        if self.test_dataset is None:
            return None

        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            collate_fn=self.collate_fn
        )

    def collate_fn(self, batch: List[Dict]) -> Dict:
        """
        Custom collate function to handle batching of crystal structures.

        Args:
            batch: List of samples from the dataset

        Returns:
            Collated batch
        """
        # Extract crystal data
        crystals = [item["crystal"] for item in batch]

        # Stack atom types (shape: [batch_size, max_atoms, num_atom_types])
        if "atom_types_onehot" in crystals[0]:
            atom_types = torch.stack([c["atom_types_onehot"] for c in crystals])
        else:
            atom_types = torch.stack([c["atom_types"] for c in crystals])

        # Stack positions (shape: [batch_size, max_atoms, 3])
        positions = torch.stack([c["positions"] for c in crystals])

        # Stack lattice parameters (shape: [batch_size, 6])
        lattice = torch.stack([c["lattice"] for c in crystals])

        # Stack atom masks if available
        atom_masks = None
        if "atom_mask" in crystals[0]:
            atom_masks = torch.stack([c["atom_mask"] for c in crystals])

        # Create collated crystal data
        collated_crystal = {
            "atom_types": atom_types,
            "positions": positions,
            "lattice": lattice
        }

        if atom_masks is not None:
            collated_crystal["atom_mask"] = atom_masks

        # Handle conditional properties if present
        if "condition" in batch[0] and batch[0]["condition"] is not None:
            conditions = torch.stack([item["condition"] for item in batch])
            return {"crystal": collated_crystal, "condition": conditions}

        return {"crystal": collated_crystal}
