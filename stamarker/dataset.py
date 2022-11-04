from typing import List
import scanpy as sc
import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import Dataset
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch_geometric.loader import NeighborLoader
from torch_geometric.data import Data
from .utils import compute_spatial_net, stats_spatial_net, compute_edge_list


class Batch(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

    def to(self, device):
        res = dict()
        for key, value in self.items():
            if hasattr(value, "to"):
                res[key] = value.to(device)
            else:
                res[key] = value
        return Batch(**res)


class RepDataset(Dataset):
    def __init__(self,
                 x,
                 target_y,
                 ground_truth=None):
        assert (len(x) == len(target_y))
        self.x = x
        self.target_y = target_y
        self.ground_truth = ground_truth

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample_x, sample_y = self.x[idx, :], self.target_y[idx]
        if self.ground_truth is not None:
            sample_gt = self.ground_truth[idx]
        else:
            sample_gt = np.nan
        sample = {"x": sample_x, "y": sample_y, "ground_truth": sample_gt}
        return sample


class SpatialDataModule(pl.LightningDataModule):
    def __init__(self,
                 full_batch: bool = True,
                 batch_size: int = 1000,
                 num_neighbors: List[int] = None,
                 num_workers=None,
                 pin_memory=False) -> None:
        self.batch_size = batch_size
        self.full_batch = full_batch
        self.has_y = False
        self.train_dataset = None
        self.valid_dataset = None
        self.num_neighbors = num_neighbors
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.ann_data = None

    def prepare_data(self, n_top_genes: int = 3000, rad_cutoff: float = 50,
                     show_net_stats: bool = False, min_cells=50, min_counts=None) -> None:
        sc.pp.calculate_qc_metrics(self.ann_data, inplace=True)
        sc.pp.filter_genes(self.ann_data, min_cells=min_cells)
        if min_counts is not None:
            sc.pp.filter_cells(self.ann_data, min_counts=min_counts)
        print("After filtering: ", self.ann_data.shape)
        # Normalization
        sc.pp.highly_variable_genes(self.ann_data, flavor="seurat_v3", n_top_genes=n_top_genes)
        self.ann_data = self.ann_data[:, self.ann_data.var['highly_variable']]
        sc.pp.normalize_total(self.ann_data, target_sum=1e4)
        sc.pp.log1p(self.ann_data)
        compute_spatial_net(self.ann_data, rad_cutoff=rad_cutoff)
        if show_net_stats:
            stats_spatial_net(self.ann_data)
        # ---------------------------- generate data ---------------------
        edge_list = compute_edge_list(self.ann_data)
        self.train_dataset = Data(edge_index=torch.LongTensor(np.array([edge_list[0], edge_list[1]])),
                                  x=torch.FloatTensor(self.ann_data.X),
                                  y=None)

    def train_dataloader(self):
        if self.full_batch:
            loader = NeighborLoader(self.train_dataset, num_neighbors=[1],
                                    batch_size=len(self.train_dataset.x))
        else:
            loader = NeighborLoader(self.train_dataset, num_neighbors=self.num_neighbors, batch_size=self.batch_size)
        return loader

    def val_dataloader(self):
        if self.valid_dataset is None:
            loader = NeighborLoader(self.train_dataset, num_neighbors=[1],
                                    batch_size=len(self.train_dataset.x))
        else:
            raise NotImplementedError
        return loader

    def test_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError

    def predict_dataloader(self) -> EVAL_DATALOADERS:
        raise NotImplementedError
