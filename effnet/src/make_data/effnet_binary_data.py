import glob
import os
import re
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../"))
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import utils
from sklearn.model_selection import StratifiedGroupKFold
from tqdm.notebook import tqdm

from CSFD.data import three_dimensions as th_dim


class EffnetDataSet(torch.utils.data.Dataset):
    def __init__(self, df, path, transforms=None):
        super().__init__()
        self.df = df
        self.path = path
        self.transforms = transforms

    def __getitem__(self, i):
        path = os.path.join(
            self.path, self.df.iloc[i].StudyInstanceUID, f"{self.df.iloc[i].Slice}.dcm"
        )

        try:
            img = utils.load_dicom(path)[0]
            # Pytorch uses (batch, channel, height, width) order. Converting (height, width, channel) -> (channel, height, width)
            img = np.transpose(img, (2, 0, 1))
            if self.transforms is not None:
                img = self.transforms(torch.as_tensor(img))
        except Exception as ex:
            print(ex)
            return None

        if "patient_overall" in self.df:
            frac_targets = torch.as_tensor(
                self.df.iloc[i][["patient_overall"]].astype("float32").values
            )
            return img, frac_targets
        return img

    def __len__(self):
        return len(self.df)


def preprocess(
    df_train, df_train_slices, df_train_box, df_test, TEST_IMAGES_PATH, N_FOLDS
):

    # detect reversed slice uid
    uid_list = list(df_train_box["StudyInstanceUID"].unique())
    df_train = df_train.query("StudyInstanceUID == @uid_list")
    #df_train = df_train.query("StudyInstanceUID == @uid_list | patient_overall == 0")[
    #    ["StudyInstanceUID", "patient_overall"]
    #].reset_index(drop=True)
    #df_train =pd.concat([df_train.query("patient_overall == 1").reset_index(drop =True),
    #                     df_train.query("patient_overall == 0")[:235].reset_index(drop =True)],axis = 0).reset_index(drop =True)
    df_train_box = df_train_box[["StudyInstanceUID", "slice_number", "is_reversed"]]

    # Pick reversed Slice
    list_reversed_uid = df_train_box.query("is_reversed == True")[
        "StudyInstanceUID"
    ].unique()

    train_uid = list(df_train["StudyInstanceUID"].unique())
    df_train_temp = (
        df_train_slices[["StudyInstanceUID", "Slice"]]
        .query("StudyInstanceUID == @train_uid")
        .merge(df_train, on="StudyInstanceUID", how="left")
    )
    df_train_box = df_train_box.rename(columns={"slice_number": "new_slice"})
    list_df_trains = []
    for uid in list_reversed_uid:
        df_temp = df_train_temp.query("StudyInstanceUID == @uid")
        df_temp["new_slice"] = df_temp["Slice"].values[::-1]
        list_df_trains.append(df_temp)
    df_others = df_train_temp.query("StudyInstanceUID not in  @list_reversed_uid")
    df_others["new_slice"] = df_others["Slice"]
    list_df_trains.append(df_others)
    df_train = pd.concat(list_df_trains)
    df_train = df_train.reindex(
        columns=["StudyInstanceUID", "patient_overall", "Slice", "new_slice"]
    )
    df_train = df_train.drop("patient_overall", axis=1)
    df_train_box["patient_overall"] = 1
    df_train = df_train.merge(
        df_train_box.drop("is_reversed", axis=1),
        on=["StudyInstanceUID", "new_slice"],
        how="left",
    ).fillna(0)

    # reject bug StudyInstanceUID
    df_train = df_train.query(
        'StudyInstanceUID != "1.2.826.0.1.3680043.20574"'
    ).reset_index(drop=True)

    # Add Fold columns
    cv = StratifiedGroupKFold(n_splits=N_FOLDS, random_state=44, shuffle=True)
    for k, (_, test_idx) in enumerate(
        cv.split(df_train, y=df_train.patient_overall, groups=df_train.StudyInstanceUID)
    ):
        df_train.loc[test_idx, "split"] = k
    if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
        # test_images and test.csv are inconsistent in the dev dataset, fixing labels for the dev run.
        df_test = pd.DataFrame(
            {
                "row_id": [
                    "1.2.826.0.1.3680043.22327_C1",
                    "1.2.826.0.1.3680043.25399_C1",
                    "1.2.826.0.1.3680043.5876_C1",
                ],
                "StudyInstanceUID": [
                    "1.2.826.0.1.3680043.22327",
                    "1.2.826.0.1.3680043.25399",
                    "1.2.826.0.1.3680043.5876",
                ],
                "prediction_type": ["C1", "C1", "patient_overall"],
            }
        )

    test_slices = glob.glob(f"{TEST_IMAGES_PATH}/*/*")
    test_slices = [
        re.findall(f"{TEST_IMAGES_PATH}/(.*)/(.*).dcm", s)[0] for s in test_slices
    ]
    df_test_slices = pd.DataFrame(
        data=test_slices, columns=["StudyInstanceUID", "Slice"]
    )
    df_test = (
        df_test.set_index("StudyInstanceUID")
        .join(df_test_slices.set_index("StudyInstanceUID"))
        .reset_index()
    )
    return df_train, df_train_slices, df_test, df_test_slices


# For debug
if __name__ == "__main__":
    import torchvision as tv

    cfg = utils.load_yaml(
        Path(
            "/home/jumpei.uchida/develop/kaggle_1080ti_1_2/rsna-2022-cervical-spine-fracture-detection/effnet/config/config.yaml"
        )
    )
    # DATA PATH
    RSNA_2022_PATH = cfg["data"]["RSNA_2022_PATH"]
    TRAIN_IMAGES_PATH = f"{RSNA_2022_PATH}/train_images"
    TEST_IMAGES_PATH = f"{RSNA_2022_PATH}/test_images"
    METADATA_PATH = cfg["data"]["METADATA_PATH"]
    N_FOLDS = int(cfg["model"]["N_FOLDS"])
    WEIGHTS = tv.models.efficientnet.EfficientNet_V2_S_Weights.DEFAULT
    # PARAMETER OF EFFNET

    # Read csv data for slicing
    df_train = pd.read_csv(f"{RSNA_2022_PATH}/train.csv")
    df_train_slices = pd.read_csv(f"{METADATA_PATH}/train_segmented.csv")
    df_test = pd.read_csv(f"{RSNA_2022_PATH}/test.csv")
    df_train_box = pd.read_csv(f"{RSNA_2022_PATH}/cropped_2d_labels.csv")

    # PreProcess and Effnetdata
    df_train, df_train_slices, df_test, df_test_slices = preprocess(
        df_train=df_train,
        df_train_slices=df_train_slices,
        df_train_box=df_train_box,
        df_test=df_test,
        TEST_IMAGES_PATH=TEST_IMAGES_PATH,
        N_FOLDS=N_FOLDS,
    )

    ds_train = EffnetDataSet(df_train, TRAIN_IMAGES_PATH, WEIGHTS.transforms())
    ds_test = EffnetDataSet(df_test, TEST_IMAGES_PATH, WEIGHTS.transforms())
