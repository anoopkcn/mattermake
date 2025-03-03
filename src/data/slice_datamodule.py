from typing import Optional

import torch
from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, Dataset


class SliceDataset(Dataset):
    def __init__(self, embeddings, slice_ids):
        self.embeddings = embeddings
        self.slice_ids = slice_ids

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return {
            "embedding": self.embeddings[idx] if torch.is_tensor(self.embeddings[idx]) else torch.tensor(self.embeddings[idx], dtype=torch.float),
            "slice_ids": torch.tensor(self.slice_ids[idx], dtype=torch.long),
        }


class SliceDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "data/slice",
        batch_size: int = 16,
        num_workers: int = 0,
        block_size: int = 1024,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.block_size = block_size

        self.data_train = None
        self.data_val = None

    def prepare_data(self):
        # Download or prepare data if needed
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            try:
                # Try to load PyTorch files first
                train_data = torch.load(f"{self.data_dir}/train.pt")
                val_data = torch.load(f"{self.data_dir}/val.pt")
            except (FileNotFoundError, RuntimeError):
                # Fall back to NumPy files if PyTorch files not found
                import numpy as np
                train_data = np.load(f"{self.data_dir}/train.npy", allow_pickle=True).item()
                val_data = np.load(f"{self.data_dir}/val.npy", allow_pickle=True).item()

            self.data_train = SliceDataset(train_data["embeddings"], train_data["slice_ids"])
            self.data_val = SliceDataset(val_data["embeddings"], val_data["slice_ids"])

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch):
        embeddings = torch.stack([item["embedding"] for item in batch])

        # Process slice_ids with padding
        slices = [item["slice_ids"] for item in batch]
        max_len = min(max(len(s) for s in slices), self.block_size)

        x = torch.zeros((len(batch), max_len), dtype=torch.long)
        y = torch.zeros((len(batch), max_len), dtype=torch.long)

        for i, slice_ids in enumerate(slices):
            slice_ids = slice_ids[:max_len]
            seq_len = len(slice_ids)
            if seq_len > 1:  # We need at least two tokens
                x[i, :seq_len-1] = slice_ids[:-1]
                y[i, :seq_len-1] = slice_ids[1:]

        return {
            "embeddings": embeddings,
            "input_ids": x,
            "target_ids": y
        }
