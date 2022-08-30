import plotly.express as px
import plotly_utility
import numpy as np
import CSFD.data
import attrdict

dataset_cfg = attrdict.AttrDict(
    type="train",
    data_root_path="../rsna-2022-cervical-spine-fracture-detection",
    target_columns=["C1", "C2", "C3", "C4", "C5", "C6", "C7"],
    cv=dict(
        type="KFold",
        seed=42,
        n_folds=5,
        fold=0
    )
)

df = CSFD.data.get_df(dataset_cfg)
assert np.all(df[dataset_cfg.target_columns].values.any(axis=1) == df["patient_overall"])

fig = px.histogram(
    x=df["patient_overall"])
plotly_utility.offline.mpl_plot(fig)

