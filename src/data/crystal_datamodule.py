from typing import Optional
import torch
from torch_geometric.data import Dataset, Data
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader

from src.utils.crystal_to_graph import structure_to_quotient_graph


class CrystalDataset(Dataset):
    def __init__(
        self,
        structures,
        transform=None,
        pre_transform=None,
    ):
        super().__init__(transform, pre_transform)
        self.structures = structures

    def len(self):
        return len(self.structures)

    def get(self, idx):
        structure = self.structures[idx]

        node_features, edge_index, edge_features, cell_params = (
            structure_to_quotient_graph(structure)
        )

        data = Data(
            x=node_features,
            edge_index=edge_index,
            edge_attr=edge_features,
            cell_params=cell_params,
            num_nodes=node_features.size(0),
        )

        return data


class CrystalDataModule(LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data/structures",
        batch_size: int = 32,
        num_workers: int = 0,
    ):
        super().__init__()

        self.save_hyperparameters(logger=False)

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.data_train: Optional[Dataset] = None
        self.data_val: Optional[Dataset] = None
        self.data_test: Optional[Dataset] = None

    def prepare_data(self):
        """Download data if needed."""
        pass

    def setup(self, stage: Optional[str] = None):
        if stage == "fit" or stage is None:
            try:
                train_data = torch.load(f"{self.data_dir}/train.pt")
                val_data = torch.load(f"{self.data_dir}/val.pt")
                test_data = torch.load(f"{self.data_dir}/test.pt")
            except (FileNotFoundError, RuntimeError):
                import numpy as np

                train_data = np.load(
                    f"{self.data_dir}/train.npy", allow_pickle=True
                ).item()
                val_data = np.load(f"{self.data_dir}/val.npy", allow_pickle=True).item()
                test_data = np.load(
                    f"{self.data_dir}/test.npy", allow_pickle=True
                ).item()

            self.data_train = CrystalDataset(train_data["structures"])
            self.data_val = CrystalDataset(val_data["structures"])
            self.data_test = CrystalDataset(test_data["structures"])

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            shuffle=False,
        )
