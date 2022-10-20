import CSFD.metric.numpy
import CSFD.data.io
import pandas as pd
import sys


fn = sys.argv[1]

df = CSFD.data.io.three_dimensions.get_df(
    data_root_path="rsna-2022-cervical-spine-fracture-detection", dataset_type="train",
    type_to_load="dcm"
)

predicted_df = pd.read_csv(fn)
df = pd.merge(predicted_df[["uid"]], df, left_on="uid", right_on="StudyInstanceUID", how="left")
assert (df["StudyInstanceUID"].values == predicted_df["uid"].values).all()

cv = CSFD.metric.numpy.competition_loss(
    predicted_df[['Overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].values,
    df[['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']].values
)
print(f"{cv:.4f}")
