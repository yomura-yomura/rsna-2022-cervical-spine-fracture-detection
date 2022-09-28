import plotly.express as px
import plotly_utility
import pathlib
import pandas as pd


dicom_metadata_path = pathlib.Path("dicom_metadata_csv")

values = []
for p in dicom_metadata_path.glob("*.csv"):
    df = pd.read_csv(p)
    values += df[df["key"] == "RescaleSlope"]["value"].tolist()
    # values += df[df["key"] == "WindowWidth"]["value"].tolist()


# px.histogram(
#     x=
# )