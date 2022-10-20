import pathlib

import CSFD.data.io
import pandas as pd
import pydicom
import tqdm


df = CSFD.data.io.three_dimensions.get_df(
    data_root_path="../data/rsna-2022-cervical-spine-fracture-detection",
    dataset_type="train", type_to_load="dcm"
)

data_path = pathlib.Path("dicom_metadata_csv")
data_path.mkdir(exist_ok=True)

for dcm_images_path in tqdm.tqdm(df["dcm_images_path"]):
    dcm_paths = CSFD.data.io.three_dimensions.get_dicom_paths(dcm_images_path)
    uid = dcm_images_path.name

    target_csv = data_path / f"{uid}.csv"

    if target_csv.exists():
        continue

    df_list = []
    for slice_number, dcm_path in enumerate(dcm_paths):
        d = pydicom.read_file(dcm_path)
        df_list.append(
            pd.DataFrame([
                (uid, slice_number, i.keyword, i.tag, i.value)
                for i in d.iterall() if i.keyword != "PixelData"
            ], columns=["UID", "slice_number", "key", "tag", "value"])
        )
    df = pd.concat(df_list)
    df.to_csv(target_csv, index=False)
