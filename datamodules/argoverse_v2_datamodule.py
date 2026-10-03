from typing import Callable, Optional

import pytorch_lightning as pl
from torch_geometric.loader import DataLoader

from Datasets.dataset_pkl import AVDataset


class ArgoverseV2DataModule(pl.LightningDataModule):
    def __init__(
        self,
        root: str,
        processed: str,
        train_batch_size: int,
        val_batch_size: int,
        test_batch_size: int,
        shuffle: bool = True,
        num_workers: int = 0,
        pin_memory: bool = True,
        persistent_workers: bool = True,
        train_transform: Optional[Callable] = None,
        val_transform: Optional[Callable] = None,
        test_transform: Optional[Callable] = None,
        **kwargs,
    ) -> None:
        super().__init__()
        self.root = root
        self.processed = processed
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.test_batch_size = test_batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.persistent_workers = persistent_workers and num_workers > 0
        self.train_transform = train_transform
        self.val_transform = val_transform
        self.test_transform = test_transform

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            self.train_dataset = AVDataset(
                root=self.root,
                processed=self.processed,
                split="train",
                transform=self.train_transform,
            )
            self.val_dataset = AVDataset(
                root=self.root,
                processed=self.processed,
                split="val",
                transform=self.val_transform,
            )
        if stage in (None, "test", "predict"):
            self.test_dataset = AVDataset(
                root=self.root,
                processed=self.processed,
                split="test",
                transform=self.test_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.train_batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.val_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.test_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
        )
