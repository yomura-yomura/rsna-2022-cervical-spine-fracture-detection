import pandas as pd
import numpy as np
import CSFD.data.io.three_dimensions
import plotly.express as px
import plotly_utility
import CSFD.metric.numpy


predicted_df = pd.read_csv("binary_effnetv2_unet_pred.csv")
predicted_df["fold"] = predicted_df["fold"].astype(int)
predicted_df = predicted_df.rename(columns={"overall": "patient_overall"})
assert len(predicted_df["StudyInstanceUID"].unique()) == len(predicted_df)

df = CSFD.data.io.three_dimensions.get_df("../rsna-2022-cervical-spine-fracture-detection", "train")
df = pd.merge(df, predicted_df["StudyInstanceUID"], on="StudyInstanceUID", how="right")
assert np.all(df["StudyInstanceUID"] == predicted_df["StudyInstanceUID"])
assert np.all(predicted_df.index == df.index)

target_columns = ['patient_overall', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7']

fig = px.histogram(
    x=predicted_df["patient_overall"], color=df["patient_overall"], barmode="overlay",
    histnorm="probability density", marginal="box",
    labels={"x": "predicted probability", "color": "patient_overall"}
)
plotly_utility.offline.mpl_plot(fig)

df["fold"] = predicted_df["fold"]

count_df = df.groupby("fold")["patient_overall"].value_counts().rename("count").reset_index()
count_df["percent"] = count_df.groupby("fold")["count"].transform(lambda c: c / c.sum())
count_df["fold"] = count_df["fold"].astype(str)
count_df["percent_str"] = count_df["percent"].apply(lambda p: f"{p:.0%}")
fig = px.bar(
    count_df,
    x="patient_overall", y="count", color="fold", barmode="group",
    text="percent_str"
)
fig.show()

predicted_df["loss"] = CSFD.metric.numpy.competition_loss(
    predicted_df[target_columns].values, df[target_columns].values, reduction=None
)
fig = px.histogram(
    predicted_df, x="loss", color=df["patient_overall"].astype(str),
    facet_row="fold", barmode="overlay",
    log_y=True,
    category_orders={"fold": [0, 1, 2]}, labels={"color": "patient_overall"}
)
fig.show()
