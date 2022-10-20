import gc
import sys
import warnings
import cv2
import pandas as pd
import pydicom
import tqdm
import numpy as np
import pathlib
import joblib
import skimage.exposure
from CSFD.data import io as _io_module
from CSFD.data import io_with_cfg as _io_with_cfg_module


def save_all_3d_images(
    images_dir_path, output_dir_path, image_2d_shape, enable_depth_resized_with_cv2, data_type,
    depth, depth_range, height_range, width_range, use_windowing,
    uid_list=None, n_jobs=-1
):
    images_dir_path = pathlib.Path(images_dir_path)
    output_dir_path = pathlib.Path(output_dir_path)
    output_dir_path.mkdir(parents=True, exist_ok=True)

    if uid_list is None:
        dicom_paths = list(images_dir_path.glob("*"))
    else:
        dicom_paths = []
        for uid in uid_list:
            dicom_path = images_dir_path / uid
            if not dicom_path.exists():
                warnings.warn(f"{dicom_path} not found in {images_dir_path}", UserWarning)
                continue
            dicom_paths.append(dicom_path)

    for dicom_path in tqdm.tqdm(dicom_paths, file=sys.stdout, desc="creating 3d images"):
        output_path = output_dir_path / f"{dicom_path.name}.npz"
        if output_path.exists():
            continue

        images = load_3d_images(
            dicom_path, image_2d_shape, enable_depth_resized_with_cv2, data_type,
            n_jobs=n_jobs, depth=depth, depth_range=depth_range,
            height_range=height_range, width_range=width_range,
            widowing=use_windowing
        )
        np.savez_compressed(output_path, images)
        # np.savez(output_path, images)

        # might be needed for kaggle notebook
        del images
        gc.collect()


def resize_depth(images: np.ndarray, depth, depth_range, enable_depth_resized_with_cv2):
    assert images.ndim >= 3  # (..., depth, h/w, w/h)

    if depth_range is not None:
        assert len(depth_range) == 2
        start_idx, end_idx = np.quantile(np.arange(images.shape[-3]), depth_range).astype(int)
        images = images[..., start_idx:end_idx, :, :]

    if depth is None:
        return images

    # if images.shape[-3] < depth:
    #     warnings.warn("images.shape[-3] < given depth", UserWarning)

    if enable_depth_resized_with_cv2:
        images = images.swapaxes(-3, -2)
        *left_shapes, images_height, images_depth, images_width = images.shape
        images = images.reshape((-1, images_depth, images_width))
        images = np.stack([
            cv2.resize(image, (images_width, depth), interpolation=cv2.INTER_AREA)
            for image in images
        ], axis=0)
        images = images.reshape((*left_shapes, images_height, depth, images_width))
        images = images.swapaxes(-3, -2)
        return images
    else:
        indices = np.quantile(
            np.arange(images.shape[-3]), np.linspace(0, 1, depth)
        ).astype(int)
        return images[..., indices, :, :]


def get_dicom_paths(dicom_dir_path: pathlib.Path):
    dicom_paths = sorted(
        dicom_dir_path.glob("*"),
        key=lambda p: int(p.name.split(".")[0])
    )
    if (
        pydicom.dcmread(dicom_paths[0]).get("ImagePositionPatient")[2]
        >
        pydicom.dcmread(dicom_paths[-1]).get("ImagePositionPatient")[2]
    ):
        return dicom_paths[::-1]
    return dicom_paths


def load_3d_images(
    dicom_dir_path, image_2d_shape=None, enable_depth_resized_with_cv2=True, data_type="f4",
    n_jobs=-1, depth=None, depth_range=None, height_range=None, width_range=None,
    voi_lut=False, widowing=False
):
    dicom_dir_path = pathlib.Path(dicom_dir_path)
    if not dicom_dir_path.exists():
        raise FileNotFoundError(dicom_dir_path)
    dicom_paths = get_dicom_paths(dicom_dir_path)

    if n_jobs == 1:
        images = [
            _io_module.two_dimensions.load_image(
                dicom_path, image_2d_shape, data_type,
                height_range, width_range,
                voi_lut, True,
                widowing
            )
            for dicom_path in dicom_paths
        ]
    else:
        images = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_io_module.two_dimensions.load_image)(
                dicom_path, image_2d_shape, data_type,
                height_range, width_range,
                voi_lut, True,
                widowing
            )
            for dicom_path in dicom_paths
        )
    images = np.stack(images, axis=0)

    return resize_depth(images, depth, depth_range, enable_depth_resized_with_cv2)


def get_df(
        data_root_path, dataset_type,
        type_to_load="dcm",
        train_3d_images=None, data_type=None,
        depth=None, depth_range=None,
        image_2d_shape=None,
        height_range=None, width_range=None,
        enable_depth_resized_with_cv2=True,
        use_windowing=False,

        cv=None, target_columns=None,
        use_segmentation=False, train_segmentations_path=None,

        ignore_invalids=True, n_jobs_to_save_images=-1,
        cropped_2d_labels_path=None
):
    df = _io_module.two_dimensions.get_df(
        data_root_path, dataset_type, cv, target_columns,
        ignore_invalid=ignore_invalids
    )

    if use_segmentation:
        for p in (pathlib.Path(data_root_path) / "segmentations").glob("*.nii"):
            df.loc[df["StudyInstanceUID"] == p.name[:-4], "nil_segmentations_path"] = p
        df = df.dropna()

    if cropped_2d_labels_path is not None:
        cropped_2d_labels_df = pd.read_csv(cropped_2d_labels_path)
        df = df[df["StudyInstanceUID"].isin(cropped_2d_labels_df["StudyInstanceUID"])]

    if train_segmentations_path:
        train_segmentations_path = pathlib.Path(train_segmentations_path)
        df["npz_segmentations_path"] = [train_segmentations_path / f"{uid}.npz" for uid in df["StudyInstanceUID"]]
        does_exist = df["npz_segmentations_path"].map(lambda p: p.exists())
        if np.any(does_exist):
            warnings.warn(f"{np.count_nonzero(does_exist):,} npz_segmentations_path not found.")
            df.loc[~does_exist, "npz_segmentations_path"] = np.nan

    if type_to_load not in ("npz", "dcm"):
        raise ValueError(type_to_load)

    if type_to_load == "npz":
        depth_dir = (
            "_".join([
                f"{depth}" if depth is not None else "normal",
                f"{'-'.join(map(str, depth_range)) if depth_range is not None else 'normal'}"
            ])
        )
        height_dir = (
            "_".join([
                f"{image_2d_shape[0]}" if image_2d_shape is not None else "normal",
                f"{'-'.join(map(str, height_range or [0, 1])) if height_range is not None else 'normal'}"
            ])
        )
        width_dir = (
            "_".join([
                f"{image_2d_shape[1]}" if image_2d_shape is not None else "normal",
                f"{'-'.join(map(str, width_range or [0, 1])) if width_range is not None else 'normal'}"
            ])
        )
        if train_3d_images is None:
            raise ValueError(f"train_3d_images must not be None")
        output_dir_path = (
            pathlib.Path(train_3d_images)
            / ("windowing" if use_windowing else "normal")
            / "_".join(map(str, image_2d_shape or ["normal"]))
            / f"d{depth_dir}"
            / f"h{height_dir}"
            / f"w{width_dir}"
            / np.dtype(data_type).name
        )
        save_all_3d_images(
            images_dir_path=pathlib.Path(data_root_path) / f"{dataset_type}_images",
            output_dir_path=output_dir_path,
            image_2d_shape=image_2d_shape,
            enable_depth_resized_with_cv2=enable_depth_resized_with_cv2,
            data_type=data_type,
            depth=depth,
            depth_range=depth_range,
            height_range=height_range, width_range=width_range,
            use_windowing=use_windowing,
            uid_list=df["StudyInstanceUID"], n_jobs=n_jobs_to_save_images
        )
        df["np_images_path"] = df["StudyInstanceUID"].map(lambda uid: output_dir_path / f"{uid}.npz")
        if np.all(df["np_images_path"].map(lambda p: p.exists())) == np.False_:
            raise FileNotFoundError

    df["dcm_images_path"] = df["StudyInstanceUID"].map(
        lambda uid: pathlib.Path(data_root_path) / f"{dataset_type}_images" / uid
    )

    return df
