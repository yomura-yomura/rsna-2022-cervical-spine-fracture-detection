import logging
import warnings

from pytorch_lightning import LightningDataModule
import os
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import torch
import torch.nn.functional
import CSFD.data
import CSFD.data.three_dimensions
import CSFD.data.io_with_cfg
import CSFD.monai.transforms
from monai.data import CacheDataset, DataLoader


__all__ = [
    "CSFDDataModule"
]


# class CSFDDataset(torch.utils.data.Dataset):
#     def __init__(
#         self,
#         df: pd.DataFrame,
#         cfg_dataset,
#     ):
#         self.cfg_dataset = cfg_dataset
#
#         if self.cfg_dataset.type_to_load == "npz":
#             self.images_paths = df["np_images_path"].to_list()
#         elif self.cfg_dataset.type_to_load == "dcm":
#             self.images_paths = df["dcm_images_path"].to_list()
#         else:
#             raise ValueError(self.cfg_dataset.datatype_to_load)
#
#         self.study_uid_list = df["StudyInstanceUID"].to_list()
#         self.target_columns = (
#             [None] * len(df)
#             if (
#                 self.cfg_dataset.target_columns is None
#                 or
#                 np.all(np.isin(self.cfg_dataset.target_columns, df.columns)) == np.False_
#             ) else
#             df[list(self.cfg_dataset.target_columns)].to_numpy(int)
#         )
#
#     def __getitem__(self, idx: int) -> dict:
#         uid = self.study_uid_list[idx]
#         images = CSFD.data.io_with_cfg.load_3d_images(self.images_paths[idx], self.cfg_dataset)
#         images = torch.Tensor(images).half()
#
#         if self.target_columns[idx] is None:
#             return {
#                 "uid": uid,
#                 "data": images
#             }
#         else:
#             label = torch.Tensor(self.target_columns[idx]).half()
#             return {
#                 "uid": uid,
#                 "data": images,
#                 "label": label
#             }
#
#     def __len__(self) -> int:
#         return len(self.images_paths)


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
        # self.label_names = {}
        # self.train_transforms = train_transforms
        # self.valid_transforms = valid_transforms

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
        if stage == "predict":
            self.test_dataset = CacheDataset(
                self.df.to_dict("records"),
                CSFD.monai.transforms.get_transforms(self.cfg, is_train=False),
                cache_rate=0
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



class Cropped3DDataset(Dataset):
    def __init__(self, cfg, dataset: CacheDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_uid_array = np.array([record["StudyInstanceUID"] for record in self.dataset.data])
        self.semantic_segmentation_bb_df = CSFD.data.io.load_semantic_segmentation_bb_df(
            self.cfg.dataset.semantic_segmentation_bb_path
        )
        self.unique_id = np.unique(self.semantic_segmentation_bb_df[["StudyInstanceUID", "type"]].to_numpy(str), axis=0)
        assert self.cfg.dataset.target_columns == ['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        self.vertebrae_types = self.cfg.dataset.target_columns[1:]

    def __len__(self):
        # return len(self.dataset) * len(self.vertebrae_types)
        return len(self.unique_id)

    def __getitem__(self, idx):
        # dataset_idx = idx // len(self.vertebrae_types)
        # vertebrae_idx = idx % len(self.vertebrae_types)
        target_uid = self.unique_id[idx, 0]
        dataset_idx = int(np.argmax(self.dataset_uid_array == target_uid))

        record = self.dataset[dataset_idx]
        # vertebrae_type = self.vertebrae_types[vertebrae_idx]
        vertebrae_type = self.unique_id[idx, 1]

        matched_ss_bb_df = self.semantic_segmentation_bb_df[
            self.semantic_segmentation_bb_df["StudyInstanceUID"] == record["uid"]
        ]

        idf = matched_ss_bb_df[matched_ss_bb_df["type"] == vertebrae_type]
        if len(idf) == 0:
            warnings.warn(f"no matched at {idx}")
            data = torch.empty([0, 0, 0, 0])
        else:
            assert len(np.unique(idf[["x0", "y0", "x1", "y1"]].values, axis=0)) == 1
            x0, y0, x1, y1 = idf[["x0", "y0", "x1", "y1"]].iloc[0]
            data = record["data"][:, idf["slice_number"].to_numpy(), x0:x1, y0:y1]

            # common_shape = np.array([41, 122, 142])
            margins_to_reshape = [
                (
                    int(np.floor((self.cfg.dataset.common_shape_for_bb[i] - data.shape[1 + i]) / 2)),
                    int(np.ceil((self.cfg.dataset.common_shape_for_bb[i] - data.shape[1 + i]) / 2))
                )
                for i in np.arange(3)[::-1].tolist()
            ]
            margins_to_reshape.append((0, 0))

            data = torch.nn.functional.pad(
                data,
                pad=tuple(
                    margin
                    for margins in margins_to_reshape
                    for margin in margins
                )
            )

        if "label" in record.keys():
            label = record["label"][self.cfg.dataset.target_columns.index(vertebrae_type)]
            return {
                "data": data,
                "label": torch.tensor([label])
            }
        else:
            return {
                "data": data
            }

class CSFDCropped3DDataModule(CSFDDataModule):
    def setup(self, stage=None):
        super(CSFDCropped3DDataModule, self).setup(stage)

        if self.cfg.dataset.semantic_segmentation_bb_path is not None:
            if self.train_dataset is not None:
                self.train_dataset = Cropped3DDataset(self.cfg, self.train_dataset)
            if self.valid_dataset is not None:
                self.valid_dataset = Cropped3DDataset(self.cfg, self.valid_dataset)
            if self.test_dataset is not None:
                self.test_dataset = Cropped3DDataset(self.cfg, self.test_dataset)
