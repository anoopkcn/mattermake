from typing import Optional, List
import torch
from dataclasses import dataclass
from torch_geometric.data import Dataset, Data
from lightning.pytorch import LightningDataModule
from torch_geometric.loader import DataLoader


@dataclass
class CrystalGraphData(Data):
    """Data structure for crystal graphs that inherits from torch_geometric.data.Data"""

    material_id: str

    def __init__(
        self,
        material_id: str,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        cell_params: torch.Tensor,
        num_nodes: int,
    ):
        super().__init__(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            cell_params=cell_params,
            num_nodes=num_nodes,
        )
        self.material_id = material_id


class CrystalDataset(Dataset):
    def __init__(
        self, data_list: List[CrystalGraphData], transform=None, pre_transform=None
    ):
        super().__init__(transform, pre_transform)
        self.data = data_list

    def len(self):
        return len(self.data)

    def get(self, idx):
        return self.data[idx]


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

                self.data_train = CrystalDataset(train_data)
                self.data_val = CrystalDataset(val_data)
                self.data_test = CrystalDataset(test_data)

            except Exception as e:
                raise RuntimeError(f"Error loading data from {self.data_dir}: {str(e)}")

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
