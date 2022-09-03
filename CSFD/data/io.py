import pandas as pd
import sklearn.model_selection
import omegaconf
import pathlib
import numpy as np
import pydicom
import cv2


__all__ = ["load_yaml_config", "get_df", "load_image", "get_submission_df"]


_folds_csv_root_path = pathlib.Path(__file__).resolve().parent / "_folds_csv"


invalid_study_uid_list = [
    "1.2.826.0.1.3680043.20574"
]


def load_yaml_config(path):
    cfg = omegaconf.OmegaConf.load(path)
    omegaconf.OmegaConf.set_struct(cfg, True)

    def _validate_cfg(cfg, key, default_value):
        if not hasattr(cfg, key):
            setattr(cfg, key, default_value)

    # Needed just for compatibility
    with omegaconf.open_dict(cfg):
        for cfg_key, default_map_dict in {
            "dataset": {
                "type_to_load": "npz",
                "image_2d_shape": [256, 256],
                "enable_depth_resized_with_cv2": False,
                "data_type": "u1",
                "depth_range": [0.1, 0.9],
                "height_range": None,
                "width_range": None,
                "save_images_with_specific_depth": False,
            },
            "model": {
                "use_multi_sample_dropout": False,
            },
            "train": {
                "early_stopping": False,
            }
        }.items():
            cfg_key = getattr(cfg, cfg_key)
            for key, default_value in default_map_dict.items():
                _validate_cfg(cfg_key, key, default_value)

        if cfg.model.name.startswith("resnet"):
            cfg.model.kwargs = dict(n_input_channels=1)

    return cfg


def drop_invalids(*dfs):
    df = dfs[0]
    ret = [df_[~df["StudyInstanceUID"].isin(invalid_study_uid_list)] for df_ in dfs]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def get_df(dataset_cfg, ignore_invalid=True):
    df = pd.read_csv(pathlib.Path(dataset_cfg.data_root_path) / f"{dataset_cfg.type}.csv")

    if dataset_cfg.type == "train":
        if hasattr(dataset_cfg, "debug") and dataset_cfg.debug:
            df = df.iloc[:10]
        else:
            _folds_csv_path = (
                _folds_csv_root_path / "_".join([
                    f"{dataset_cfg.cv.type}",
                    f"nFolds{dataset_cfg.cv.n_folds}",
                    f"Seed{dataset_cfg.cv.seed}"
                ])
            ).with_suffix(".csv")

            if _folds_csv_path.exists():
                df = pd.concat([df, pd.read_csv(_folds_csv_path)], axis=1)
            else:
                if dataset_cfg.cv.type == "KFold":
                    kf = sklearn.model_selection.KFold(
                        n_splits=dataset_cfg.cv.n_folds,
                        shuffle=True, random_state=dataset_cfg.cv.seed
                    )
                else:
                    raise NotImplementedError(f"Unexpected dataset_cfg.cv.type: {dataset_cfg.cv.type}")

                df["fold"] = -1
                for fold, (_, valid_indices) in enumerate(
                        kf.split(
                            df.drop(columns=dataset_cfg.target_columns),
                            df[list(dataset_cfg.target_columns)]
                        )
                ):
                    df.loc[valid_indices, "fold"] = fold
                assert np.all(df["fold"] >= 0)
                df["fold"].to_csv(_folds_csv_path, index=False)
                print(f"[Info] {_folds_csv_root_path} has been created.")

        if ignore_invalid:
            df = drop_invalids(df)
    elif dataset_cfg.type == "test":
        if len(get_submission_df(dataset_cfg)) == 3:
            df = pd.DataFrame({
                "row_id": [
                    '1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'
                ],
                "StudyInstanceUID": [
                    '1.2.826.0.1.3680043.22327', '1.2.826.0.1.3680043.25399', '1.2.826.0.1.3680043.5876'
                ],
                "prediction_type": [
                    "C1", "C1", "C1"
                ]
            })
    else:
        raise ValueError(f"Unexpected dataset_cfg.type: {dataset_cfg.type}")
    return df


def load_image(dicom_path, image_shape=(256, 256), data_type="u1",
               height_range=None, width_range=None):
    # data_type = np.dtype(data_type)

    dicom = pydicom.read_file(dicom_path)
    data = dicom.pixel_array.astype("f4")
    data -= np.min(data)
    if np.max(data) != 0:
        data /= np.max(data)
    data *= 255
    data = data.astype(data_type)
    # data = dicom.pixel_array
    # data = data - np.min(data)
    # if np.max(data) != 0:
    #     data = data / np.max(data)
    # data = (data * 255).astype(data_type)

    if height_range is not None:
        data = data[np.quantile(np.arange(len(data)), height_range).astype(int), :]
    if width_range is not None:
        data = data[:, np.quantile(np.arange(len(data)), height_range).astype(int)]

    if image_shape is not None:
        data = cv2.resize(data, image_shape, interpolation=cv2.INTER_AREA)
    return data


def get_submission_df(dataset_cfg):
    df = pd.read_csv(pathlib.Path(dataset_cfg.data_root_path) / "sample_submission.csv")
    if len(df) == 3:
        df = pd.DataFrame({
            "row_id": [
                '1.2.826.0.1.3680043.22327_C1', '1.2.826.0.1.3680043.25399_C1', '1.2.826.0.1.3680043.5876_C1'
            ],
            "fractured": [
                0.5, 0.5, 0.5
            ]
        })

    return df
