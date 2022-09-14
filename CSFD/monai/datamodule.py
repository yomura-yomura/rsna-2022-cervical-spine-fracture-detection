import logging
from pytorch_lightning import LightningDataModule
import os
import torch.utils.data
import numpy as np
import pandas as pd
import torch
import CSFD.data
import CSFD.data.three_dimensions
import CSFD.data.io_with_cfg
import CSFD.monai.transforms
from monai.data import CacheDataset, DataLoader


__all__ = [
    "CSFDDataModule"
]


class CSFDDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg_dataset,
    ):
        self.cfg_dataset = cfg_dataset

        if self.cfg_dataset.type_to_load == "npz":
            self.images_paths = df["np_images_path"].to_list()
        elif self.cfg_dataset.type_to_load == "dcm":
            self.images_paths = df["dcm_images_path"].to_list()
        else:
            raise ValueError(self.cfg_dataset.datatype_to_load)

        self.study_uid_list = df["StudyInstanceUID"].to_list()
        self.target_columns = (
            [None] * len(df)
            if (
                self.cfg_dataset.target_columns is None
                or
                np.all(np.isin(self.cfg_dataset.target_columns, df.columns)) == np.False_
            ) else
            df[list(self.cfg_dataset.target_columns)].to_numpy(int)
        )

    def __getitem__(self, idx: int) -> dict:
        uid = self.study_uid_list[idx]
        images = CSFD.data.io_with_cfg.load_3d_images(self.images_paths[idx], self.cfg_dataset)
        images = torch.Tensor(images).half()

        if self.target_columns[idx] is None:
            return {
                "uid": uid,
                "data": images
            }
        else:
            label = torch.Tensor(self.target_columns[idx]).half()
            return {
                "uid": uid,
                "data": images,
                "label": label
            }

    def __len__(self) -> int:
        return len(self.images_paths)


class CSFDDataModule(LightningDataModule):
    def __init__(self, cfg, df=None):
        super().__init__()

        self.cfg = cfg

        if df is None:
            self.df = CSFD.data.get_df(self.cfg.dataset)
        else:
            self.df = df

        # other configs
        self.num_workers = (
            self.cfg.dataset.num_workers
            if self.cfg.dataset.num_workers is not None
            else os.cpu_count()
        )

        # need to be defined in setup()
        self.train_dataset = None
        self.valid_dataset = None
        self.test_dataset = None
        self.label_names = {}
        # self.train_transforms = train_transforms
        # self.valid_transforms = valid_transforms

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CacheDataset(
                self.df[self.df["fold"] != self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=True),
                cache_rate=self.cfg.dataset.train_cache_rate
            )
            logging.info(f"training dataset: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.valid_dataset = CacheDataset(
                self.df[self.df["fold"] == self.cfg.dataset.cv.fold].to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False),
                cache_rate=self.cfg.dataset.valid_cache_rate
            )
            logging.info(f"validation dataset: {len(self.valid_dataset)}")
            self.label_names = self.cfg.dataset.target_columns
        if stage == "predict":
            self.test_dataset = CSFDDataset(
                self.df,
                self.cfg.dataset
            )
            logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.cfg.dataset.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.valid_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def predict_dataloader(self):
        if not self.test_dataset:
            logging.warning('no testing data found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.test_batch_size,
            num_workers=self.num_workers
        )
