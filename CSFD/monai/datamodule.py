import logging
from pytorch_lightning import LightningDataModule
import os
from typing import Union, Optional, Tuple
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
from .. import data as _data_module


__all__ = [
    "SpineScansDataset", "SpineScansDataModule"
]

class SpineScansDataset(Dataset):
    def __init__(
        self,
        volume_dir: str,
        df: pd.DataFrame,
        mode: str,
        # split: float,
        in_memory: bool = False,
        # random_state=42,
    ):
        self.volume_dir = volume_dir
        self.mode = mode
        self.in_memory = in_memory

        self.table = df

        # # shuffle data
        # self.table = self.table.sample(frac=1, random_state=random_state).reset_index(drop=True)

        # # split dataset
        # assert 0.0 <= split <= 1.0
        # frac = int(split * len(self.table))
        # self.table = self.table[:frac] if mode == 'train' else self.table[frac:]

        # populate images/labels
        self.label_names = sorted([c for c in self.table.columns if c.startswith("C")])
        self.labels = self.table[self.label_names].values if self.label_names else [None] * len(self.table)
        self.volumes = [os.path.join(volume_dir, f"{row['StudyInstanceUID']}.pt") for _, row in self.table.iterrows()]
        assert len(self.volumes) == len(self.labels)

    def __getitem__(self, idx: int) -> dict:
        label = self.labels[idx]
        vol_ = self.volumes[idx]
        if isinstance(vol_, str):
            # try:
            vol = torch.load(vol_).to(torch.float32)
            # except (EOFError, RuntimeError):
            #     print(f"failed loading: {vol_}")
        else:
            vol = vol_
        if self.in_memory:
            self.volumes[idx] = vol
        # in case of predictions, return image name as label
        label = label if label is not None else vol_
        return {"data": vol.unsqueeze(0), "label": label}

    def __len__(self) -> int:
        return len(self.volumes)


class SpineScansDataModule(LightningDataModule):
    def __init__(
        self,
        cfg,
        in_memory: bool = False,
        train_transforms=None,
        valid_transforms=None
    ):
        super().__init__()

        self.cfg = cfg

        self.train_dir = self.cfg.dataset.data_root_path / 'train_volumes'
        self.test_dir = self.cfg.dataset.data_root_path / 'test_volumes'
        self.df = _data_module.get_df(self.cfg)

        # other configs
        self.in_memory = in_memory
        self.num_workers = (
            self.cfg.dataset.num_workers
            if self.cfg.dataset.num_workers is not None
            else os.cpu_count()
        )

        # need to be filled in setup()
        self.test_table = []
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.label_names = {}
        self.train_transforms = train_transforms
        self.valid_transforms = valid_transforms

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def setup(self, stage=None):
        """Prepare datasets"""

        if stage == "fit":
            self.train_dataset = SpineScansDataset(
                volume_dir=self.train_dir,
                df=self.df,
                in_memory=self.in_memory,
                mode='train'
            )
            logging.info(f"training dataset: {len(self.train_dataset)}")
        elif stage == "validate":
            self.valid_dataset = SpineScansDataset(
                volume_dir=self.train_dir,
                df=self.df,
                in_memory=self.in_memory,
                mode='valid'
            )
            logging.info(f"validation dataset: {len(self.valid_dataset)}")
            self.label_names = sorted(set(self.train_dataset.label_names + self.valid_dataset.label_names))

        if not os.path.isdir(self.test_dir):
            logging.warning(f"Missing test folder: {self.test_dir}")
            return
        ls_cases = [os.path.basename(p) for p in self.test_dir.glob('*')]
        self.test_table = [dict(StudyInstanceUID=os.path.splitext(n)[0]) for n in ls_cases]
        self.test_dataset = SpineScansDataset(
            self.test_dir,
            df=pd.DataFrame(self.test_table),
            mode='test',
        )
        logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            # batch_transforms=self.train_transforms,
            batch_size=self.cfg.dataset.train_batch_size,
            num_workers=self.num_workers
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            # batch_transforms=self.valid_transforms,
            batch_size=self.cfg.dataset.valid_batch_size,
            num_workers=self.num_workers
        )

    def test_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            logging.warning('no testing data found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            # batch_transforms=self.valid_transforms,
            batch_size=self.cfg.dataset.test_batch_size,
            num_workers=self.num_workers
        )
