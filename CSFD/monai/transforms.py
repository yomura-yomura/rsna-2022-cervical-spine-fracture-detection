import monai.transforms
import pandas as pd
import numpy as np
import warnings
import CSFD.data.three_dimensions
import CSFD.data.io_with_cfg
from monai.transforms import (
    Compose,
    RandScaleIntensityd, RandShiftIntensityd,
    RandAffined,
    EnsureTyped,
    Lambdad,
    ToTensord
)
import torch


def get_transforms(cfg, is_train):
    transforms = [
        LoadImage(cfg.dataset)
    ]

    if is_train:
        if hasattr(cfg.train.augmentation, "scale_intensity"):
            transforms.append(
                RandScaleIntensityd(
                    keys="data",
                    **cfg.train.augmentation.scale_intensity.kwargs
                )
            )
        if hasattr(cfg.train.augmentation, "shift_intensity"):
            transforms.append(
                RandShiftIntensityd(
                    keys="data",
                    **cfg.train.augmentation.shift_intensity.kwargs
                )
            )
        if hasattr(cfg.train.augmentation, "affine"):
            transforms.append(
                RandAffined(
                    keys="data",
                    rotate_range=np.deg2rad(cfg.train.augmentation.affine.rotate_range_in_deg),
                    translate_range=np.multiply(
                        [cfg.dataset.depth, *cfg.dataset.image_2d_shape],
                        cfg.train.augmentation.affine.translate_range_in_scale
                    ),
                    **cfg.train.augmentation.affine.kwargs
                )
            )

    transforms += [
        # Lambdad(keys="data", func=normalize_image_wise),
        EnsureTyped(keys=("data", "label"), dtype=torch.float16, allow_missing_keys=True),
        ToTensord(keys=("data", "label"))
    ]
    transforms = Compose(transforms)
    transforms.set_random_state(seed=cfg.train.seed)
    return transforms


def normalize_image_wise(data):
    if isinstance(data, monai.data.MetaTensor):
        data = data.numpy()
    data /= data.max(axis=-1, keepdims=True).max(axis=-2, keepdims=True)
    return data


class LoadImage(monai.transforms.MapTransform):
    def __init__(self, cfg_dataset):
        keys = ["StudyInstanceUID", "np_images_path", "dcm_images_path"]
        super().__init__(keys, allow_missing_keys=True)
        self.cfg_dataset = cfg_dataset

    def __call__(self, row: dict):
        uid = row["StudyInstanceUID"]
        target_columns = (
            None
            if (
                self.cfg_dataset.target_columns is None
                or
                np.all(np.isin(self.cfg_dataset.target_columns, list(row.keys()))) == np.False_
            ) else
            [row[col] for col in self.cfg_dataset.target_columns]
        )

        if self.cfg_dataset.type_to_load == "npz":
            images_path = row["np_images_path"]
        elif self.cfg_dataset.type_to_load == "dcm":
            images_path = row["dcm_images_path"]
        else:
            raise ValueError(self.cfg_dataset.datatype_to_load)

        images = CSFD.data.io_with_cfg.load_3d_images(images_path, self.cfg_dataset)

        if target_columns is None:
            return {
                "uid": uid,
                "data": images
            }
        else:
            return {
                "uid": uid,
                "data": images,
                "label": target_columns
            }
