import logging
import warnings

import pandas as pd
from pytorch_lightning import LightningDataModule
import os
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional
import CSFD.data
import CSFD.data.io.three_dimensions
import CSFD.data.io_with_cfg.three_dimensions
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


class BaseCroppedDataset(Dataset):
    def __init__(self, cfg, dataset: CacheDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_uid_array = np.array([record["StudyInstanceUID"] for record in self.dataset.data])
        self.semantic_segmentation_bb_df = CSFD.data.io.two_dimensions.load_semantic_segmentation_bb_df(
            self.cfg.dataset.semantic_segmentation_bb_path
        )

        # UNet学習時の設定, TODO: hard-coded
        self.default_depth_range = [0.1, 0.9]
        self.default_data_shape = [128, 256, 256]

    def _scale_depths_fitting_to_new_depth_range(self, depths, new_depth):
        if self.cfg.dataset.depth_range is None and self.cfg.dataset.depth is None:
            org_slice_numbers = np.arange(new_depth)
            start_i, end_i = map(int, np.quantile(org_slice_numbers, self.default_depth_range))
            depths = tuple(map(int, np.quantile(np.arange(start_i, end_i), np.array(depths) / self.default_data_shape[0])))
        elif self.cfg.dataset.depth_range == self.default_depth_range:
            pass
        else:
            raise NotImplementedError
        return depths

    def _calc_2d_ranges_to_crop(self, x0, x1, y0, y1, width, height):
        scales_to_resize = np.array([width, height]) / self.default_data_shape[1:]

        x0 *= scales_to_resize[0]
        x1 *= scales_to_resize[0]
        y0 *= scales_to_resize[1]
        y1 *= scales_to_resize[1]
        x0 = int(x0)
        x1 = int(x1)
        y0 = int(y0)
        y1 = int(y1)
        return x0, x1, y0, y1

    def _pad_images_with_zeros(self, images):
        """
        images: 2D or 3D
        """
        common_shape = self.cfg.dataset.common_shape_for_bb

        margins_to_reshape = [
            (
                int(np.floor((common_shape[i] - images.shape[i]) / 2)) if common_shape[i] is not None else 0,
                int(np.ceil((common_shape[i] - images.shape[i]) / 2)) if common_shape[i] is not None else 0
            )
            for i in (-np.arange(1, 1 + len(images.shape[1:]))).tolist()
        ]
        margins_to_reshape.append((0, 0))

        images = torch.nn.functional.pad(
            images,
            pad=tuple(
                margin
                for margins in margins_to_reshape
                for margin in margins
            )
        )
        return images


class Cropped3DDataset(BaseCroppedDataset):
    def __init__(self, cfg, dataset: CacheDataset):
        super(Cropped3DDataset, self).__init__(cfg, dataset)
        
        self.unique_id = np.unique(self.semantic_segmentation_bb_df[["StudyInstanceUID", "type"]].to_numpy(str), axis=0)
        assert self.cfg.dataset.target_columns == ['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        self.vertebrae_types = self.cfg.dataset.target_columns[1:]

    def __len__(self):
        # return len(self.dataset) * len(self.vertebrae_types)
        return len(self.unique_id)

    def _get_record(self, idx) -> dict:
        target_uid = self.unique_id[idx, 0]
        dataset_idx = int(np.argmax(self.dataset_uid_array == target_uid))
        return self.dataset[dataset_idx]

    def _get_vertebrae_type(self, idx) -> str:
        return self.unique_id[idx, 1]

    def __getitem__(self, idx) -> dict:
        record = self._get_record(idx)
        vertebrae_type = self._get_vertebrae_type(idx)

        matched_ss_bb_df = self.semantic_segmentation_bb_df[
            self.semantic_segmentation_bb_df["StudyInstanceUID"] == record["uid"]
        ]
        matched_ss_bb_df = matched_ss_bb_df[matched_ss_bb_df["type"] == vertebrae_type]

        if len(matched_ss_bb_df) == 0:
            warnings.warn(f"no matched at {idx}")
            data = torch.empty([0, 0, 0, 0])
        else:
            assert len(np.unique(matched_ss_bb_df[["x0", "y0", "x1", "y1"]].values, axis=0)) == 1
            slice_numbers = matched_ss_bb_df["slice_number"].to_numpy()
            d0 = slice_numbers.min()
            d1 = slice_numbers.max()
            x0, y0, x1, y1 = matched_ss_bb_df[["x0", "y0", "x1", "y1"]].iloc[0]

            org_data_shape = np.array(record["data"].shape[1:])

            d0, d1 = self._scale_depths_fitting_to_new_depth_range([d0, d1], org_data_shape[0])
            x0, x1, y0, y1 = self._calc_2d_ranges_to_crop(x0, x1, y0, y1, org_data_shape[1], org_data_shape[2])

            data = record["data"][:, d0:d1, x0:x1, y0:y1]
            data = self._pad_images_with_zeros(data)

        new_record = {
            "data": data
        }
        if "label" in record.keys():
            new_record["label"] = torch.tensor([record["label"][self.cfg.dataset.target_columns.index(vertebrae_type)]])
        return new_record


class CSFDCropped3DDataModule(CSFDDataModule):
    def setup(self, stage=None):
        super(CSFDCropped3DDataModule, self).setup(stage)

        assert self.cfg.dataset.semantic_segmentation_bb_path is not None

        if self.train_dataset is not None:
            self.train_dataset = Cropped3DDataset(self.cfg, self.train_dataset)
        if self.valid_dataset is not None:
            self.valid_dataset = Cropped3DDataset(self.cfg, self.valid_dataset)
        if self.test_dataset is not None:
            self.test_dataset = Cropped3DDataset(self.cfg, self.test_dataset)


class Cropped2DDataset(BaseCroppedDataset):
    def __init__(self, cfg, dataset: CacheDataset):
        assert cfg.dataset.cropped_2d_labels_path is not None
        assert cfg.dataset.common_shape_for_bb[0] is None
        super(Cropped2DDataset, self).__init__(cfg, dataset)

        cropped_2d_labels_df = pd.read_csv(self.cfg.dataset.cropped_2d_labels_path)
        self.semantic_segmentation_bb_df = pd.merge(
            self.semantic_segmentation_bb_df, cropped_2d_labels_df,
            on=["StudyInstanceUID", "slice_number", "type"], how="inner"
        )
        assert len(self.semantic_segmentation_bb_df) == len(cropped_2d_labels_df)
        self.semantic_segmentation_bb_df = self.semantic_segmentation_bb_df[
            self.semantic_segmentation_bb_df["StudyInstanceUID"].isin(self.dataset_uid_array)
        ]

    def __len__(self):
        return len(self.semantic_segmentation_bb_df)

    def __getitem__(self, idx):
        (
            target_uid, slice_number, label,
            x0, y0, x1, y1
        ) = self.semantic_segmentation_bb_df[[
            "StudyInstanceUID", "slice_number", "label",
            "x0", "y0", "x1", "y1"
        ]].iloc[idx]
        record: dict = self.dataset[int(np.argmax(self.dataset_uid_array == target_uid))]

        org_data_shape = np.array(record["data"].shape[1:])

        slice_number = self._scale_depths_fitting_to_new_depth_range([slice_number], org_data_shape[0])[0]
        x0, x1, y0, y1 = self._calc_2d_ranges_to_crop(x0, x1, y0, y1, org_data_shape[1], org_data_shape[2])

        data = record["data"][:, slice_number, x0:x1, y0:y1]
        data = self._pad_images_with_zeros(data)

        new_record = {
            "data": data
        }

        if "label" in record.keys():
            new_record["label"] = torch.Tensor([label])

        return new_record


class CSFDCropped2DDataModule(CSFDDataModule):
    def setup(self, stage=None):
        super(CSFDCropped2DDataModule, self).setup(stage)

        assert self.cfg.dataset.semantic_segmentation_bb_path is not None

        if self.train_dataset is not None:
            self.train_dataset = Cropped2DDataset(self.cfg, self.train_dataset)
        if self.valid_dataset is not None:
            self.valid_dataset = Cropped2DDataset(self.cfg, self.valid_dataset)
        if self.test_dataset is not None:
            self.test_dataset = Cropped2DDataset(self.cfg, self.test_dataset)
