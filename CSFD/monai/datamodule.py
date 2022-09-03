import logging
from pytorch_lightning import LightningDataModule
import os
from typing import Optional
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import torch
import CSFD.data
import CSFD.data.three_dimensions


__all__ = [
    "CSFDDataModule"
]


class CSFDDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        cfg_dataset,
    ):
        # assert (
        #     cfg_dataset.target_columns is None
        #     or
        #     np.all(np.isin(cfg_dataset.target_columns, df.columns))
        # )

        self.cfg_dataset = cfg_dataset

        if self.cfg_dataset.type_to_load == "npz":
            self.images_paths = df["np_images_path"].to_list()
        elif self.cfg_dataset.type_to_load == "dcm":
            self.images_paths = df["dcm_images_path"].to_list()
        else:
            raise ValueError(cfg_dataset.datatype_to_load)

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

        if self.cfg_dataset.type_to_load == "npz":
            vol = np.load(self.images_paths[idx])["arr_0"]
            if len(vol) != self.cfg_dataset.depth:
                vol = CSFD.data.three_dimensions.resize_depth(
                    vol,
                    self.cfg_dataset.depth, self.cfg_dataset.depth_range,
                    self.cfg_dataset.enable_depth_resized_with_cv2
                )
        elif self.cfg_dataset.type_to_load == "dcm":
            vol = CSFD.data.three_dimensions.load_3d_images(
                self.images_paths[idx],
                self.cfg_dataset.image_2d_shape,
                self.cfg_dataset.enable_depth_resized_with_cv2,
                self.cfg_dataset.data_type,
                depth=self.cfg_dataset.depth, depth_range=self.cfg_dataset.depth_range,
                n_jobs=1
            )
        else:
            raise RuntimeError

        assert vol.dtype == np.dtype(self.cfg_dataset.data_type)

        vol = vol[np.newaxis, ...]
        vol = torch.Tensor(vol).half()

        if self.target_columns[idx] is None:
            return {
                "uid": uid,
                "data": vol
            }
        else:
            label = torch.Tensor(self.target_columns[idx])
            return {
                "uid": uid,
                "data": vol,
                "label": label
            }

    def __len__(self) -> int:
        return len(self.images_paths)


class CSFDDataModule(LightningDataModule):
    def __init__(self, cfg, df):
        super().__init__()

        self.cfg = cfg
        self.df = df

        # self.df = _data_module.get_df(self.cfg.dataset)

        # other configs
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
        # self.train_transforms = train_transforms
        # self.valid_transforms = valid_transforms

    @property
    def num_labels(self) -> int:
        return len(self.label_names)

    def setup(self, stage=None):
        if stage == "fit":
            self.train_dataset = CSFDDataset(
                self.df[self.df["fold"] != self.cfg.dataset.cv.fold],
                self.cfg.dataset
            )
            logging.info(f"training dataset: {len(self.train_dataset)}")
        if stage in ("fit", "validate"):
            self.valid_dataset = CSFDDataset(
                self.df[self.df["fold"] == self.cfg.dataset.cv.fold],
                self.cfg.dataset
            )
            logging.info(f"validation dataset: {len(self.valid_dataset)}")
            self.label_names = self.cfg.dataset.target_columns
        if stage == "predict":
            self.test_dataset = CSFDDataset(
                self.df,
                self.cfg.dataset
            )
            logging.info(f"test dataset: {len(self.test_dataset)}")

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            batch_size=self.cfg.dataset.train_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.valid_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.valid_batch_size,
            num_workers=self.num_workers,
            drop_last=True
        )

    def predict_dataloader(self) -> Optional[DataLoader]:
        if not self.test_dataset:
            logging.warning('no testing data found')
            return
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            batch_size=self.cfg.dataset.test_batch_size,
            num_workers=self.num_workers
        )
