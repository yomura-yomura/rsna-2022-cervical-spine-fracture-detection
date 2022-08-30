import gc
import sys
import warnings

import tqdm
import numpy as np
import pathlib
import joblib
from . import io as _io_module


def save_all_3d_images(images_dir_path, output_dir_path, uid_list=None, n_jobs=-1, depth=None):
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

        images = load_3d_images(dicom_path, n_jobs=n_jobs, depth=depth)
        # np.savez_compressed(output_path, images)
        np.savez(output_path, images)

        # might be needed for kaggle notebook
        del images
        gc.collect()


def load_3d_images(dicom_dir_path, n_jobs=-1, image_2d_shape=(256, 256), depth=None):
    dicom_paths = sorted(pathlib.Path(dicom_dir_path).glob("*"), key=lambda p: int(p.name.split(".")[0]))
    # print(len(dicom_paths), dicom_dir_path)
    if depth is not None:
        # instead of zooming whole dicom series, load only part of the images
        indices = np.quantile(np.arange(len(dicom_paths)), np.linspace(0.1, 0.9, depth)).astype(int)
        dicom_paths = [dicom_paths[i] for i in indices]

    if n_jobs == 1:
        images = [_io_module.load_image(dicom_path, image_2d_shape) for dicom_path in dicom_paths]
    else:
        images = joblib.Parallel(n_jobs=n_jobs)(
            joblib.delayed(_io_module.load_image)(dicom_path, image_2d_shape)
            for dicom_path in dicom_paths
        )
    return np.stack(images, axis=0)


def get_df(dataset_cfg):
    df = _io_module.get_df(dataset_cfg)
    if dataset_cfg.type_to_load == "npz":
        save_all_3d_images(
            images_dir_path=pathlib.Path(dataset_cfg.data_root_path) / f"{dataset_cfg.type}_images",
            output_dir_path=dataset_cfg.train_3d_images,
            uid_list=df["StudyInstanceUID"]
        )
        df["np_images_path"] = df["StudyInstanceUID"].map(
            lambda uid: pathlib.Path(dataset_cfg.train_3d_images) / f"{uid}.npz"
        )
        if np.all(df["np_images_path"].map(lambda p: p.exists())) == np.False_:
            raise FileNotFoundError
    elif dataset_cfg.type_to_load == "dcm":
        df["dcm_images_path"] = df["StudyInstanceUID"].map(
            lambda uid: pathlib.Path(dataset_cfg.data_root_path) / f"{dataset_cfg.type}_images" / uid
        )
    else:
        raise ValueError(dataset_cfg.type_to_load)
    return df
