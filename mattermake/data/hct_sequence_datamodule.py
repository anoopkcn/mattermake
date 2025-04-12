import os
import torch
from typing import Optional, Dict, Callable
from torch.utils.data import DataLoader
from lightning.pytorch import LightningDataModule

import torch.serialization
from mattermake.data.hct_tokenizer import CrystalTokenData, CrystalTokenizer

from mattermake.data.hct_sequence_dataset import (
    CrystalSequenceDataset,
    collate_crystal_sequences,
)
from mattermake.utils.pylogger import get_pylogger

torch.serialization.add_safe_globals([CrystalTokenData])


class CrystalSequenceDataModule(LightningDataModule):
    """DataModule for crystal structure sequences"""

    def __init__(
        self,
        data_dir: str = "structure_tokens",
        processed_data_file: str = "processed_crystal_data.pt",
        batch_size: int = 32,
        num_workers: int = 4,
        pin_memory: bool = True,
        max_sequence_length: Optional[int] = None,
        train_filter: Optional[Callable] = None,
        val_filter: Optional[Callable] = None,
        test_filter: Optional[Callable] = None,
        data_augmentation: bool = False,
        weighted_sampling: bool = False,
        sampling_weights: Optional[Dict[str, float]] = None,
        shuffle: bool = True,
        drop_last: bool = False,
        cache_tokenized: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(
            logger=False,
            ignore=["train_filter", "val_filter", "test_filter"],
        )

        self.data_dir = data_dir
        self.processed_data_file = processed_data_file
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.max_sequence_length = max_sequence_length
        self.train_filter = train_filter
        self.val_filter = val_filter
        self.test_filter = test_filter
        self.data_augmentation = data_augmentation
        self.weighted_sampling = weighted_sampling
        self.sampling_weights = sampling_weights
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.cache_tokenized = cache_tokenized

        self.tokenizer = None
        self.data_train = None
        self.data_val = None
        self.data_test = None
        self.train_sampler = None

        self.log = get_pylogger(__name__)

    def prepare_data(self):
        """Check if processed data exists"""
        data_path = os.path.join(self.data_dir, self.processed_data_file)
        if not os.path.exists(data_path):
            raise FileNotFoundError(
                f"Processed data file not found: {data_path}. "
                f"Please run the data preparation script first."
            )

    def setup(self, stage: Optional[str] = None):
        """Set up the datasets for the requested stage"""
        data_path = os.path.join(self.data_dir, self.processed_data_file)
        data = torch.load(data_path, weights_only=False)

        if "tokenizer_config" in data:
            self.tokenizer = CrystalTokenizer(**data["tokenizer_config"])
            self.log.info(
                f"Initialized tokenizer with vocab size {self.tokenizer.vocab_size}"
            )

        if stage == "fit" or stage is None:
            if "train_data" in data:
                train_data = data["train_data"]
                self.log.info(
                    f"Setting up training dataset with {len(train_data)} structures"
                )
                self.data_train = CrystalSequenceDataset(
                    data=train_data,
                    transform=self._get_transforms(is_train=True)
                    if self.data_augmentation
                    else None,
                    filter_fn=self.train_filter,
                    max_sequence_length=self.max_sequence_length,
                    cache_tokenized=self.cache_tokenized,
                )
                self.log.info(
                    f"Training dataset contains {len(self.data_train)} structures after filtering"
                )

                if self.weighted_sampling:
                    self._setup_weighted_sampling()

            if "val_data" in data:
                val_data = data["val_data"]
                self.log.info(
                    f"Setting up validation dataset with {len(val_data)} structures"
                )
                self.data_val = CrystalSequenceDataset(
                    data=val_data,
                    transform=self._get_transforms(is_train=False),
                    filter_fn=self.val_filter,
                    max_sequence_length=self.max_sequence_length,
                    cache_tokenized=self.cache_tokenized,
                )
                self.log.info(
                    f"Validation dataset contains {len(self.data_val)} structures after filtering"
                )

        if stage == "test" or stage is None:
            if "test_data" in data:
                test_data = data["test_data"]
            elif "val_data" in data:
                test_data = data["val_data"]
            else:
                test_data = []

            if test_data:
                self.log.info(
                    f"Setting up test dataset with {len(test_data)} structures"
                )
                self.data_test = CrystalSequenceDataset(
                    data=test_data,
                    transform=self._get_transforms(is_train=False),
                    filter_fn=self.test_filter,
                    max_sequence_length=self.max_sequence_length,
                    cache_tokenized=self.cache_tokenized,
                )
                self.log.info(
                    f"Test dataset contains {len(self.data_test)} structures after filtering"
                )

    def _get_transforms(self, is_train=False):
        """Get data transforms based on configuration"""
        if not self.data_augmentation or not is_train:
            return None

        def transform_fn(item):
            return item

        return transform_fn

    def _setup_weighted_sampling(self):
        """Set up weighted sampling for imbalanced datasets"""
        if not self.data_train:
            return

        if self.sampling_weights:
            self.log.info("Using provided sampling weights")
            weights = []
            for idx in range(len(self.data_train)):
                space_group = self.data_train.space_groups[idx]
                if space_group in self.sampling_weights:
                    weights.append(self.sampling_weights[space_group])
                else:
                    weights.append(1.0)
        else:
            self.log.info("Setting up inverse frequency weighting by space group")
            space_groups = [sg for sg in self.data_train.space_groups if sg is not None]
            space_group_counts = {
                sg: space_groups.count(sg) for sg in set(space_groups)
            }

            total = len(space_groups)
            inv_freq = {sg: total / count for sg, count in space_group_counts.items()}

            max_weight = max(inv_freq.values())
            norm_weights = {sg: w / max_weight for sg, w in inv_freq.items()}

            weights = []
            for idx in range(len(self.data_train)):
                space_group = self.data_train.space_groups[idx]
                if space_group in norm_weights:
                    weights.append(norm_weights[space_group])
                else:
                    weights.append(1.0)

        from torch.utils.data import WeightedRandomSampler

        self.train_sampler = WeightedRandomSampler(
            weights=weights, num_samples=len(weights), replacement=True
        )

    def train_dataloader(self):
        """Get the training dataloader"""
        if not self.data_train:
            raise ValueError("Training data not set up. Call setup() first.")

        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            sampler=self.train_sampler
            if self.weighted_sampling and self.train_sampler
            else None,
            shuffle=self.shuffle and not self.weighted_sampling,
            drop_last=self.drop_last,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_crystal_sequences,
        )

    def val_dataloader(self):
        """Get the validation dataloader"""
        if not self.data_val:
            raise ValueError("Validation data not set up. Call setup() first.")

        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_crystal_sequences,
        )

    def test_dataloader(self):
        """Get the test dataloader"""
        if not self.data_test:
            raise ValueError("Test data not set up. Call setup() first.")

        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=False,
            drop_last=False,
            persistent_workers=self.num_workers > 0,
            collate_fn=collate_crystal_sequences,
        )

    def get_tokenizer(self):
        """Get the tokenizer used for this dataset"""
        return self.tokenizer

    def print_dataset_statistics(self):
        """Print detailed statistics about the datasets"""
        self.log.info("\n=== Dataset Statistics ===")

        if self.data_train:
            train_stats = self.data_train.get_statistics()
            self.log.info(f"\nTraining Dataset ({len(self.data_train)} structures):")
            self.log.info(f"  Unique formulas: {train_stats['unique_formulas']}")
            self.log.info(
                f"  Unique space groups: {train_stats['unique_space_groups']}"
            )

            # Print top 5 space groups
            top_sgs = sorted(
                train_stats["space_group_distribution"].items(),
                key=lambda x: x[1],
                reverse=True,
            )[:5]
            self.log.info(f"  Top 5 space groups: {top_sgs}")

        if self.data_val:
            val_stats = self.data_val.get_statistics()
            self.log.info(f"\nValidation Dataset ({len(self.data_val)} structures):")
            self.log.info(f"  Unique formulas: {val_stats['unique_formulas']}")
            self.log.info(f"  Unique space groups: {val_stats['unique_space_groups']}")

        if self.data_test:
            test_stats = self.data_test.get_statistics()
            self.log.info(f"\nTest Dataset ({len(self.data_test)} structures):")
            self.log.info(f"  Unique formulas: {test_stats['unique_formulas']}")
            self.log.info(f"  Unique space groups: {test_stats['unique_space_groups']}")

        self.log.info("\n")
