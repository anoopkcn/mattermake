#!/usr/bin/env python

import torch
import os
import sys
from pathlib import Path
import argparse
import random
from torch.utils.data import DataLoader

sys.path.append(os.path.abspath("."))  # Add project root to path

from mattermake.data.hct_dataset import HCTDataset
from mattermake.data.hct_datamodule import HCTDataModule, hct_collate_fn


def test_dataset(data_dir: str, num_samples: int = 5):
    """Test the HCTDataset with batch-processed data."""
    print(f"Testing HCTDataset with data from {data_dir}")

    # Test train dataset
    train_dir = Path(data_dir) / "train"
    if not train_dir.exists():
        print(f"Train directory not found: {train_dir}")
        return

    train_files = sorted(list(train_dir.glob("*.pt")))
    print(f"Found {len(train_files)} training files")

    # 1. Test dataset with first file
    print("Testing HCTDataset with first file")
    dataset = HCTDataset(train_files[0])
    print(f"Dataset contains {len(dataset)} structures")

    # 2. Test with another file if available
    print("Testing HCTDataset with second file")
    second_file = train_files[1] if len(train_files) > 1 else train_files[0]
    preload_dataset = HCTDataset(second_file)
    print(f"Second dataset contains {len(preload_dataset)} structures")

    # 3. Verify both methods give the same result
    if len(dataset) != len(preload_dataset):
        print(f"Dataset length mismatch: {len(dataset)} vs {len(preload_dataset)}")
    else:
        print(f"Dataset lengths match: {len(dataset)}")

    # 4. Check a few random samples
    print(f"Checking {num_samples} random samples")
    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    for idx in indices:
        sample = dataset[idx]
        preload_sample = preload_dataset[idx]

        # Print sample info
        print(f"Sample {idx}:")
        print(f"  - Material ID: {sample.get('material_id', 'Unknown')}")
        print(f"  - Space group: {sample['spacegroup'].item()}")
        print(f"  - Number of atoms: {len(sample['atom_types'])}")

        # Verify both methods give the same result
        atoms_match = torch.all(
            sample["atom_types"] == preload_sample["atom_types"]
        ).item()
        print(f"  - Preload data match: {atoms_match}")

    # 5. Test with DataLoader
    print("Testing with DataLoader")
    loader = DataLoader(dataset, batch_size=2, collate_fn=hct_collate_fn, shuffle=True)

    batch = next(iter(loader))
    print(f"Loaded batch with {len(batch['composition'])} samples")
    print(f"Batch keys: {list(batch.keys())}")
    print(f"Atom sequence shapes: {batch['atom_types'].shape}")

    return dataset


def test_datamodule(data_dir: str):
    """Test the HCTDataModule with batch-processed data."""
    print(f"Testing HCTDataModule with data from {data_dir}")

    # Create the datamodule
    datamodule = HCTDataModule(
        processed_data_dir=data_dir,
        batch_size=4,
        num_workers=0,  # Use 0 for easier debugging
    )

    # Setup the datamodule
    datamodule.prepare_data()
    datamodule.setup()

    # Check dataset sizes
    print(
        f"Train dataset size: {len(datamodule.train_dataset) if datamodule.train_dataset else 0}"
    )
    print(
        f"Val dataset size: {len(datamodule.val_dataset) if datamodule.val_dataset else 0}"
    )
    print(
        f"Test dataset size: {len(datamodule.test_dataset) if datamodule.test_dataset else 0}"
    )

    # Test the dataloaders
    if datamodule.train_dataset:
        train_loader = datamodule.train_dataloader()
        batch = next(iter(train_loader))
        print(f"Train batch size: {len(batch['composition'])}")
        print(f"Train batch atom types shape: {batch['atom_types'].shape}")

    if datamodule.val_dataset:
        val_loader = datamodule.val_dataloader()
        batch = next(iter(val_loader))
        print(f"Val batch size: {len(batch['composition'])}")

    if datamodule.test_dataset:
        test_loader = datamodule.test_dataloader()
        batch = next(iter(test_loader))
        print(f"Test batch size: {len(batch['composition'])}")

    return datamodule


def main():
    parser = argparse.ArgumentParser(description="Test HCT data loading")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/hct",
        help="Directory containing processed data with train/val/test subdirectories",
    )
    parser.add_argument(
        "--test_type",
        type=str,
        choices=["dataset", "datamodule", "both"],
        default="both",
        help="Which component to test",
    )
    parser.add_argument(
        "--num_samples", type=int, default=3, help="Number of random samples to check"
    )

    args = parser.parse_args()

    # Run the tests
    if args.test_type in ["dataset", "both"]:
        test_dataset(args.data_dir, args.num_samples)

    if args.test_type in ["datamodule", "both"]:
        test_datamodule(args.data_dir)

    print("Testing complete!")


if __name__ == "__main__":
    main()
