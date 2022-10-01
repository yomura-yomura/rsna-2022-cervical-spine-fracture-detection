import collections

import plotly.express as px
import plotly_utility
import pathlib
import pandas as pd
import tqdm

dicom_metadata_path = pathlib.Path("dicom_metadata_csv")

values_dict = collections.defaultdict(list)
for p in tqdm.tqdm(list(dicom_metadata_path.glob("*.csv"))):
    df = pd.read_csv(p)
    for key in (
        "PhotometricInterpretation",
        "WindowCenter", "WindowWidth", "RescaleIntercept", "RescaleSlope"
    ):
        values_dict[key] += df[df["key"] == key]["value"].tolist()
    # values += df[df["key"] == "WindowWidth"]["value"].tolist()


for k in values_dict.keys():
    print(k)
    print(pd.value_counts(values_dict[k]))
    print()

# px.histogram(
#     x=
# )