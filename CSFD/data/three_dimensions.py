import gc
import sys
import warnings
import cv2
import pydicom
import tqdm
import numpy as np
import pathlib
import joblib
from . import io as _io_module


def save_all_3d_images(
    images_dir_path, output_dir_path, image_2d_shape, enable_depth_resized_with_cv2, data_type,
    uid_list=None, n_jobs=-1
):
    depth = None
    depth_range = None

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
            n_jobs=n_jobs, depth=depth, depth_range=depth_range
        )
        np.savez_compressed(output_path, images)
        # np.savez(output_path, images)

        # might be needed for kaggle notebook
        del images
        gc.collect()


def resize_depth(images: np.ndarray, depth, depth_range, enable_depth_resized_with_cv2):
    assert images.ndim == 3  # (depth, h/w, w/h)

    if depth_range is not None:
        start_idx, end_idx = np.quantile(np.arange(len(images)), depth_range).astype(int)
        images = images[start_idx:end_idx]

    if depth is None:
        return images

    if len(images) < depth:
        warnings.warn("len(images) < depth", UserWarning)

    if enable_depth_resized_with_cv2:
        # print("cv2 resized")
        return np.stack([
            cv2.resize(image, (depth, image.shape[1]), interpolation=cv2.INTER_AREA)
            for image in np.rollaxis(images, axis=1)
        ], axis=1)
    else:
        assert len(depth_range) == 2
        indices = np.quantile(
            np.arange(len(images)), np.linspace(0, 1, depth)
        ).astype(int)
        return images[indices]


def _get_dicom_paths(dicom_dir_path):
    dicom_paths = sorted(
        pathlib.Path(dicom_dir_path).glob("*"),
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
    dicom_dir_path, image_2d_shape, enable_depth_resized_with_cv2, data_type,
    n_jobs=-1, depth=None, depth_range=(0.1, 0.9)
):
    dicom_paths = _get_dicom_paths(dicom_dir_path)

    if n_jobs == 1:
        images = [
            _io_module.load_image(dicom_path, image_2d_shape, data_type)
            for dicom_path in dicom_paths
        ]
    else:
        images = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_io_module.load_image)(dicom_path, image_2d_shape, data_type)
            for dicom_path in dicom_paths
        )

    images = np.stack(images, axis=0)
    return resize_depth(images, depth, depth_range, enable_depth_resized_with_cv2)


def get_df(dataset_cfg, ignore_invalids=True):
    df = _io_module.get_df(dataset_cfg, ignore_invalid=ignore_invalids)
    if dataset_cfg.type_to_load == "npz":
        depth_dir = dataset_cfg.depth if dataset_cfg.save_images_with_specific_depth else "normal"
        output_dir_path = (
            pathlib.Path(dataset_cfg.train_3d_images)
            / "_".join(map(str, dataset_cfg.image_2d_shape))
            / f"{depth_dir}"
            / np.dtype(dataset_cfg.data_type).name
        )
        save_all_3d_images(
            images_dir_path=pathlib.Path(dataset_cfg.data_root_path) / f"{dataset_cfg.type}_images",
            output_dir_path=output_dir_path,
            image_2d_shape=dataset_cfg.image_2d_shape,
            enable_depth_resized_with_cv2=dataset_cfg.enable_depth_resized_with_cv2,
            data_type=dataset_cfg.data_type,
            uid_list=df["StudyInstanceUID"]
        )
        df["np_images_path"] = df["StudyInstanceUID"].map(lambda uid: output_dir_path / f"{uid}.npz")
        if np.all(df["np_images_path"].map(lambda p: p.exists())) == np.False_:
            raise FileNotFoundError
    elif dataset_cfg.type_to_load == "dcm":
        df["dcm_images_path"] = df["StudyInstanceUID"].map(
            lambda uid: pathlib.Path(dataset_cfg.data_root_path) / f"{dataset_cfg.type}_images" / uid
        )
    else:
        raise ValueError(dataset_cfg.type_to_load)
    return df
