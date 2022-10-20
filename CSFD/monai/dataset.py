from monai.data import CacheDataset
from torch.utils.data import Dataset
import numpy as np
import torch
import torch.nn.functional
import warnings
import pathlib
import pandas as pd
import CSFD.data.io.three_dimensions
import os
import tqdm


__all__ = ["BaseCroppedDataset", "Cropped3DDataset", "Cropped2DDataset", "CacheDataset"]


class BaseCroppedDataset(Dataset):
    def __init__(self, cfg, dataset: CacheDataset):
        self.cfg = cfg
        self.dataset = dataset
        self.dataset_uid_array = np.array([record["StudyInstanceUID"] for record in self.dataset.data])

        semantic_segmentation_bb_df = CSFD.data.io.two_dimensions.load_semantic_segmentation_bb_df(
            self.cfg.dataset.semantic_segmentation_bb_path
        )
        self.semantic_segmentation_bb_df = semantic_segmentation_bb_df[
            semantic_segmentation_bb_df["StudyInstanceUID"].isin(self.dataset_uid_array)
        ]

        # UNet学習時の設定
        unet_ss_cfg = CSFD.data.io.load_yaml_config(
            pathlib.Path(self.cfg.dataset.semantic_segmentation_bb_path).with_name("UNet.yaml"),
            show_warning=False
        )
        self.default_depth_range = unet_ss_cfg.dataset.depth_range
        self.default_data_shape = [unet_ss_cfg.dataset.depth, *unet_ss_cfg.dataset.image_2d_shape]

    @staticmethod
    def get_slice_numbers_along_original_idx(original_depth, depth_range, depth):
        original_slice_numbers = np.arange(original_depth)
        start_i, end_i = np.quantile(original_slice_numbers, depth_range or [0, 1])
        return np.linspace(start_i, end_i, depth or original_depth)

    def scale_default_depths_fitting_to_current_depth(self, default_depths, original_depth: int):
        default_slice_numbers = self.get_slice_numbers_along_original_idx(
            original_depth, self.default_depth_range, self.default_data_shape[0]
        )
        original_slice_numbers_at_default_slice_numbers = default_slice_numbers[default_depths]

        target_slice_numbers = self.get_slice_numbers_along_original_idx(
            original_depth, self.cfg.dataset.depth_range, self.cfg.dataset.depth
        )

        mask = (
            (original_slice_numbers_at_default_slice_numbers < target_slice_numbers[0])
            |
            (target_slice_numbers[-1] < original_slice_numbers_at_default_slice_numbers)
        )
        if np.any(mask):
            warnings.warn(f"{np.count_nonzero(mask):,} given depths will be ignored", UserWarning)
            original_slice_numbers_at_default_slice_numbers = original_slice_numbers_at_default_slice_numbers[~mask]
        return np.searchsorted(target_slice_numbers, original_slice_numbers_at_default_slice_numbers)  # TODO: roughly calculated

    def calc_2d_ranges_to_crop(self, x0, x1, y0, y1, width, height):
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

        assert self.cfg.dataset.target_columns == ['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']
        self.vertebrae_types = self.cfg.dataset.target_columns[1:]
        self.unique_id = np.unique(self.semantic_segmentation_bb_df[["StudyInstanceUID", "type"]].to_numpy(str), axis=0)

        self.label_df = pd.melt(
            pd.DataFrame(dataset.data),
            id_vars=["StudyInstanceUID", "fold"], value_vars=cfg.dataset.target_columns,
            var_name="type", value_name="label"
        ).set_index(["StudyInstanceUID", "type"]).loc[self.unique_id.tolist()]

    def __len__(self):
        return len(self.unique_id)

    def _get_dataset_idx(self, idx):
        target_uid = self.unique_id[idx, 0]
        return int(np.argmax(self.dataset_uid_array == target_uid))

    def _get_record(self, idx) -> dict:
        return self.dataset[self._get_dataset_idx(idx)]

    def _get_n_depths(self, idx) -> int:
        return len(os.listdir(self.dataset.data[self._get_dataset_idx(idx)]["dcm_images_path"]))

    def _get_vertebrae_type(self, idx) -> str:
        return self.unique_id[idx, 1]

    def __getitem__(self, idx) -> dict:
        record = self._get_record(idx)
        vertebrae_type = self._get_vertebrae_type(idx)
        original_depth = self._get_n_depths(idx)

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

            d0, d1 = self.scale_default_depths_fitting_to_current_depth([d0, d1], original_depth)
            x0, x1, y0, y1 = self.calc_2d_ranges_to_crop(x0, x1, y0, y1, org_data_shape[1], org_data_shape[2])

            data = record["data"][:, d0:d1, x0:x1, y0:y1]
            data = self._pad_images_with_zeros(data)

        new_record = {
            "data": data
        }
        if "label" in record.keys():
            new_record["label"] = torch.Tensor([
                record["label"][self.cfg.dataset.target_columns.index(vertebrae_type)]
            ])
        return new_record


class Cropped2DDataset(BaseCroppedDataset):
    def __init__(self, cfg, dataset: CacheDataset):
        assert cfg.dataset.cropped_2d_labels_path is not None
        assert cfg.dataset.common_shape_for_bb[0] is None
        super(Cropped2DDataset, self).__init__(cfg, dataset)

        cropped_2d_labels_df = pd.read_csv(self.cfg.dataset.cropped_2d_labels_path)
        # assert np.all(cropped_2d_labels_df.columns == ['StudyInstanceUID', 'slice_number', 'label'])
        cropped_2d_labels_df = cropped_2d_labels_df[
            cropped_2d_labels_df["StudyInstanceUID"].isin(self.dataset_uid_array)
        ]

        self.semantic_segmentation_bb_df = self.semantic_segmentation_bb_df[
            self.semantic_segmentation_bb_df["StudyInstanceUID"].isin(cropped_2d_labels_df["StudyInstanceUID"])
        ]
        self.cropped_2d_labels_df = cropped_2d_labels_df.set_index(["StudyInstanceUID", "slice_number"])

    def __len__(self):
        return len(self.semantic_segmentation_bb_df)

    def get_matched_with_uid(self, uid):
        # matched_ss_bb_df = self.semantic_segmentation_bb_df[self.semantic_segmentation_bb_df["StudyInstanceUID"] == uid]
        if self.cfg.dataset.cropped_2d_images_path is not None:
            target_fn = pathlib.Path(self.cfg.dataset.cropped_2d_images_path) / f"{uid}.npz"
            if target_fn.exists():
                loaded_data = dict(np.load(target_fn))
                # loaded_data["type"] = matched_ss_bb_df.set_index("slice_number").loc[loaded_data["slice_number"]]["type"].to_numpy(str)
                return loaded_data
            else:
                raise FileNotFoundError(target_fn)
        else:
            raise NotImplementedError


    def __getitem__(self, idx):
        (
            target_uid, slice_number, vertebrae_type,
            x0, y0, x1, y1
        ) = self.semantic_segmentation_bb_df[[
            "StudyInstanceUID", "slice_number", "type",
            "x0", "y0", "x1", "y1"
        ]].iloc[idx]


        if self.cfg.dataset.cropped_2d_images_path is not None:
            target_fn = pathlib.Path(self.cfg.dataset.cropped_2d_images_path) / f"{target_uid}.npz"
            if target_fn.exists():
                loaded_data = np.load(target_fn)
                sel = (loaded_data["slice_number"] == slice_number) & (loaded_data["vertebrae_type"] == vertebrae_type)
                matched_count = np.count_nonzero(sel)
                if matched_count > 0:
                    assert matched_count == 1, matched_count
                    return dict(
                        data=torch.tensor(loaded_data["data"][sel][0][np.newaxis, ...]),
                        label=torch.Tensor(loaded_data["label"][sel])
                    )
                else:
                    warnings.warn(
                        f"no matched cropped-2d images with slice_number = {slice_number} and vertebrae_type = {vertebrae_type}",
                        UserWarning
                    )

        record: dict = self.dataset[int(np.argmax(self.dataset_uid_array == target_uid))]
        org_data_shape = np.array(record["data"].shape[1:])

        slice_number = self.scale_default_depths_fitting_to_current_depth([slice_number], org_data_shape[0])[0]
        x0, x1, y0, y1 = self.calc_2d_ranges_to_crop(x0, x1, y0, y1, org_data_shape[1], org_data_shape[2])

        try:
            label = self.cropped_2d_labels_df.loc[target_uid, slice_number][0]
        except KeyError:
            label = 0

        data = record["data"][:, slice_number, x0:x1, y0:y1]
        data = self._pad_images_with_zeros(data)

        new_record = {
            "data": data
        }

        if "label" in record.keys():
            new_record["label"] = torch.Tensor([label])

        return new_record

    def save(self):
        assert self.cfg.dataset.cropped_2d_images_path is not None
        cropped_2d_images_path = pathlib.Path(self.cfg.dataset.cropped_2d_images_path)
        cropped_2d_images_path.mkdir(exist_ok=True, parents=True)
        for uid, df in tqdm.tqdm(
                self.semantic_segmentation_bb_df.reset_index(drop=True).groupby("StudyInstanceUID"), desc="saving"
        ):
            target_fn = cropped_2d_images_path / f"{uid}.npz"

            data_to_save = dict(
                data=[None] * len(df),
                label = [None] * len(df),
                slice_number = [None] * len(df),
                vertebrae_type = [None] * len(df)
            )

            if target_fn.exists():
                old_data = np.load(str(target_fn))
                for k in data_to_save.keys():
                    if k in old_data.keys():
                        data_to_save[k] = old_data[k]

            for i, df_idx in enumerate(tqdm.tqdm(df.index)):
                if data_to_save["data"][i] is None or data_to_save["label"][i] is None:
                    record = self[df_idx]
                    data_to_save["data"][i] = record["data"].numpy()[0, ...]
                    data_to_save["label"][i] = record["label"].numpy()
                if data_to_save["slice_number"][i] is None:
                    data_to_save["slice_number"][i] = df.loc[df_idx, "slice_number"]
                if data_to_save["vertebrae_type"][i] is None:
                    data_to_save["vertebrae_type"][i] = df.loc[df_idx, "type"]

            np.savez_compressed(
                target_fn,
                **data_to_save
            )
