import warnings
import pandas as pd
import sklearn.model_selection
import pathlib
import numpy as np
import pydicom
import pydicom.pixel_data_handlers.util
import cv2
import nibabel as nib


__all__ = ["get_df", "load_image", "get_submission_df"]


_folds_csv_root_path = pathlib.Path(__file__).resolve().parent.parent / "_folds_csv"


invalid_study_uid_list = [
    "1.2.826.0.1.3680043.20574"
]


def drop_invalids(*dfs):
    df = dfs[0]
    ret = [df_[~df["StudyInstanceUID"].isin(invalid_study_uid_list)] for df_ in dfs]
    if len(ret) == 1:
        return ret[0]
    else:
        return ret


def get_df(
    data_root_path, dataset_type, cv=None, target_columns=None,
    ignore_invalid=True
):
    df = pd.read_csv(pathlib.Path(data_root_path) / f"{dataset_type}.csv")

    if dataset_type == "train":
        if cv is not None:
            _folds_csv_path = (
                _folds_csv_root_path / "_".join([
                    f"{cv.type}",
                    f"nFolds{cv.n_folds}",
                    f"Seed{cv.seed}"
                ])
            ).with_suffix(".csv")

            if _folds_csv_path.exists():
                df = pd.concat([df, pd.read_csv(_folds_csv_path)], axis=1)
            else:
                if cv.type == "KFold":
                    kf = sklearn.model_selection.KFold(
                        n_splits=cv.n_folds,
                        shuffle=True, random_state=cv.seed
                    )
                    y = df[list(target_columns)]
                elif cv.type == "StratifiedKFold":
                    kf = sklearn.model_selection.StratifiedKFold(
                        n_splits=cv.n_folds,
                        shuffle=True, random_state=cv.seed
                    )
                    y = df["patient_overall"]
                else:
                    raise NotImplementedError(f"Unexpected dataset_cfg.cv.type: {cv.type}")

                df["fold"] = -1
                for fold, (_, valid_indices) in enumerate(
                        kf.split(
                            df.drop(columns=target_columns),
                            y
                        )
                ):
                    df.loc[valid_indices, "fold"] = fold
                assert np.all(df["fold"] >= 0)
                df["fold"].to_csv(_folds_csv_path, index=False)
                print(f"[Info] {_folds_csv_root_path} has been created.")

        if ignore_invalid:
            df = drop_invalids(df)

    elif dataset_type == "test":
        if len(get_submission_df(data_root_path)) == 3:
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
        raise ValueError(f"Unexpected dataset_cfg.type: {dataset_type}")
    return df


def _get_first_of_dicom_field_as_int(x):
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def load_image(
        dicom_path, image_shape=(256, 256), data_type="u1",
        height_range=None, width_range=None,
        voi_lut=False, fix_monochrome=True,
        windowing=False
):
    dicom = pydicom.read_file(dicom_path)
    if voi_lut:
        image = pydicom.pixel_data_handlers.util.apply_voi_lut(dicom.pixel_array, dicom).astype("f4")
    else:
        image = dicom.pixel_array.astype("f4")

    if windowing:
        image *= float(dicom.get("RescaleSlope"))
        image += float(dicom.get("RescaleIntercept"))

        center = _get_first_of_dicom_field_as_int(dicom.get("WindowCenter"))
        width = _get_first_of_dicom_field_as_int(dicom.get("WindowWidth"))
        image[:] = np.clip(image, center - width / 2, center + width / 2)

    # depending on this value, X-ray may look inverted - fix that:
    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        print("[Info] fix_monochrome")
        image[:] = np.amax(image) - image

    image -= np.min(image)
    if np.max(image) != 0:
        image /= np.max(image)
    image *= 255
    image = image.astype(data_type)

    if height_range is not None:
        start_ih, end_ih = np.quantile(np.arange(image.shape[0]), height_range).astype(int)
        image = image[start_ih:end_ih, :]
    if width_range is not None:
        start_iw, end_iw = np.quantile(np.arange(image.shape[1]), width_range).astype(int)
        image = image[:, start_iw:end_iw]

    if image_shape is not None:
        image = resize_hw(image, image_shape)
    return image


def resize_hw(image, image_shape):
    assert image.ndim == 2  # (h/w, w/h)
    if image.shape[-2] < image_shape[1]:
        warnings.warn("image.shape[-2] < given image_shape[1]", UserWarning)
    if image.shape[-1] < image_shape[0]:
        warnings.warn("image.shape[-1] < given image_shape[0]", UserWarning)
    return cv2.resize(image, (image_shape[1], image_shape[0]), interpolation=cv2.INTER_AREA)


def get_submission_df(data_root_path):
    df = pd.read_csv(pathlib.Path(data_root_path) / "sample_submission.csv")
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


def load_segmentations(nil_path, separate_in_channels=False):
    nil_file = nib.load(nil_path)
    segmentations = np.asarray(nil_file.get_fdata(dtype="f2"), dtype="u1")
    # segmentations[:] = np.flip(segmentations, axis=-1)
    segmentations[:] = np.rot90(segmentations, axes=(0, 1))
    segmentations = np.rollaxis(segmentations, axis=-1)
    segmentations[segmentations > 7] = 0  # exclude labels of T1 to T12

    if separate_in_channels:
        segmentations = np.stack(
            [segmentations == label for label in np.unique(segmentations[segmentations > 0])],
            axis=0
        )
        segmentations = segmentations.astype("u1")

    segmentations *= 255

    return segmentations


def load_semantic_segmentation_bb_df(csv_path):
    semantic_segmentation_bb_df = pd.read_csv(csv_path)
    semantic_segmentation_bb_df = semantic_segmentation_bb_df.dropna()
    semantic_segmentation_bb_df["x0"] = np.floor(semantic_segmentation_bb_df["x0"]).astype(int)
    semantic_segmentation_bb_df["y0"] = np.floor(semantic_segmentation_bb_df["y0"]).astype(int)
    # semantic_segmentation_bb_df["x1"] = np.ceil(semantic_segmentation_bb_df["x1"]).astype(int)
    # semantic_segmentation_bb_df["y1"] = np.ceil(semantic_segmentation_bb_df["y1"]).astype(int)
    semantic_segmentation_bb_df["x1"] = np.floor(semantic_segmentation_bb_df["x1"]).astype(int)
    semantic_segmentation_bb_df["y1"] = np.floor(semantic_segmentation_bb_df["y1"]).astype(int)
    return semantic_segmentation_bb_df
