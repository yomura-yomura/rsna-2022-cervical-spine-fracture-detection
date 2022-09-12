import gc
import sys
import warnings
import cv2
import pydicom
import tqdm
import numpy as np
import pathlib
import joblib
import skimage.exposure
from . import io as _io_module


def save_all_3d_images(
    images_dir_path, output_dir_path, image_2d_shape, enable_depth_resized_with_cv2, data_type,
    depth, depth_range, height_range, width_range,
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
            height_range=height_range, width_range=width_range
        )
        np.savez_compressed(output_path, images)
        # np.savez(output_path, images)

        # might be needed for kaggle notebook
        del images
        gc.collect()


def resize_depth(images: np.ndarray, depth, depth_range, enable_depth_resized_with_cv2):
    assert images.ndim == 3  # (depth, h/w, w/h)

    if depth_range is not None:
        assert len(depth_range) == 2
        start_idx, end_idx = np.quantile(np.arange(len(images)), depth_range).astype(int)
        images = images[start_idx:end_idx]

    if depth is None:
        return images

    if len(images) < depth:
        warnings.warn("len(images) < given depth", UserWarning)

    if enable_depth_resized_with_cv2:
        return np.stack([
            cv2.resize(image, (image.shape[1], depth), interpolation=cv2.INTER_AREA)
            for image in np.rollaxis(images, axis=1)
        ], axis=1)
    else:
        indices = np.quantile(
            np.arange(len(images)), np.linspace(0, 1, depth)
        ).astype(int)
        return images[indices]


def _get_dicom_paths(dicom_dir_path: pathlib.Path):
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
    n_jobs=-1, depth=None, depth_range=None, height_range=None, width_range=None
):
    dicom_dir_path = pathlib.Path(dicom_dir_path)
    if not dicom_dir_path.exists():
        raise FileNotFoundError(dicom_dir_path)
    dicom_paths = _get_dicom_paths(dicom_dir_path)

    if n_jobs == 1:
        images = [
            _io_module.load_image(dicom_path, image_2d_shape, data_type, height_range, width_range)
            for dicom_path in dicom_paths
        ]
    else:
        images = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_io_module.load_image)(dicom_path, image_2d_shape, data_type, height_range, width_range)
            for dicom_path in dicom_paths
        )
    images = np.stack(images, axis=0)

    return resize_depth(images, depth, depth_range, enable_depth_resized_with_cv2)


def get_df(cfg_dataset, ignore_invalids=True, n_jobs_to_save_images=-1):
    df = _io_module.get_df(cfg_dataset, ignore_invalid=ignore_invalids)

    if cfg_dataset.use_segmentations:
        for p in (pathlib.Path(cfg_dataset.data_root_path) / "segmentations").glob("*.nii"):
            df.loc[df["StudyInstanceUID"] == p.name[:-4], "nil_images_path"] = p
        df = df.dropna()

    if cfg_dataset.type_to_load == "npz":
        depth_dir = (
            "_".join([
                f"{cfg_dataset.depth}",
                f"{'-'.join(map(str, cfg_dataset.depth_range))}"
            ])
            if cfg_dataset.save_images_with_specific_depth else
            "normal"
        )
        height_dir = (
            "_".join([
                f"{cfg_dataset.image_2d_shape[0]}" if cfg_dataset.image_2d_shape is not None else "normal",
                f"{'-'.join(map(str, cfg_dataset.height_range or [0, 1]))}"
            ])
            if cfg_dataset.save_images_with_specific_height else
            "normal"
        )
        width_dir = (
            "_".join([
                f"{cfg_dataset.image_2d_shape[1]}" if cfg_dataset.image_2d_shape is not None else "normal",
                f"{'-'.join(map(str, cfg_dataset.width_range or [0, 1]))}"
            ])
            if cfg_dataset.save_images_with_specific_width else
            "normal"
        )

        output_dir_path = (
            pathlib.Path(cfg_dataset.train_3d_images)
            / "_".join(map(str, cfg_dataset.image_2d_shape or ["normal"]))
            / depth_dir
            / height_dir
            / width_dir
            / np.dtype(cfg_dataset.data_type).name
        )
        save_all_3d_images(
            images_dir_path=pathlib.Path(cfg_dataset.data_root_path) / f"{cfg_dataset.type}_images",
            output_dir_path=output_dir_path,
            image_2d_shape=cfg_dataset.image_2d_shape,
            enable_depth_resized_with_cv2=cfg_dataset.enable_depth_resized_with_cv2,
            data_type=cfg_dataset.data_type,
            depth=cfg_dataset.depth,
            depth_range=cfg_dataset.depth_range,
            height_range=cfg_dataset.height_range, width_range=cfg_dataset.width_range,
            uid_list=df["StudyInstanceUID"],
            n_jobs=n_jobs_to_save_images
        )
        df["np_images_path"] = df["StudyInstanceUID"].map(lambda uid: output_dir_path / f"{uid}.npz")
        if np.all(df["np_images_path"].map(lambda p: p.exists())) == np.False_:
            raise FileNotFoundError
    elif cfg_dataset.type_to_load == "dcm":
        df["dcm_images_path"] = df["StudyInstanceUID"].map(
            lambda uid: pathlib.Path(cfg_dataset.data_root_path) / f"{cfg_dataset.type}_images" / uid
        )
    else:
        raise ValueError(cfg_dataset.type_to_load)

    return df
