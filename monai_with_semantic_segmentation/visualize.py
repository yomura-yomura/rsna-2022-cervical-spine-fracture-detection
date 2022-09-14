import CSFD.data.three_dimensions
import CSFD.bounding_box
import numpy as np
import itertools
import tqdm
import plotly.express as px
import plotly_utility
import pandas as pd
import pathlib

cfg = CSFD.data.load_yaml_config("SEResNext50.yaml")
df = CSFD.data.three_dimensions.get_df(cfg.dataset)

train_segmentations_path = pathlib.Path(cfg.dataset.train_segmentations_path)

for name in (p.name for p in train_segmentations_path.parent.glob("fold*") if p.is_dir()):
    train_segmentations_path = train_segmentations_path.with_name(name)
    target_csv_path = train_segmentations_path.with_name(f"segmentations-info-{train_segmentations_path.name}.csv")

    if not target_csv_path.exists():
        data = []
        for p in tqdm.tqdm(df["npz_segmentations_path"]):
            for i, segmentations in enumerate(np.load(p)["arr_0"]):
                count = np.count_nonzero(segmentations)

                row = {"name": p.name, "type": f"C{i + 1}", "count": count}
                if count > 0:
                    bb = CSFD.bounding_box.get_3d_bounding_box(segmentations)
                    row["shape0"], row["shape1"], row["shape2"] = segmentations[tuple(itertools.starmap(slice, bb))].shape
                    row["mean0"], row["mean1"], row["mean2"] = np.mean(bb, axis=1)
                else:
                    print(f"count == 0")
                data.append(row)

        segmentation_df = pd.DataFrame(data)
        segmentation_df.to_csv(target_csv_path, index=False)
    else:
        segmentation_df = pd.read_csv(target_csv_path)


import plotly_utility.express as pux

fig = pux.histogram(
    segmentation_df,
    x="count", facet_row="type", log_y=True, nbins=100,
    width=800, height=1000
)
plotly_utility.offline.mpl_plot(fig)


fig = pux.histogram(
    segmentation_df.melt(
        id_vars="type",
        value_vars=["shape0", "shape1", "shape2"],
        var_name="shape", value_name="x"
    ),
    title="Shape",
    x="x", facet_col="shape", facet_row="type",
    use_different_bin_widths=True, nbins=20,
    width=800, height=1000
)
fig.update_xaxes(title="depth", matches="x1", col=1)
fig.update_xaxes(title="height", matches="x2", col=2)
fig.update_xaxes(title="width", matches="x3", col=3)
plotly_utility.subplots.update_xaxes(fig, "inside", title=None)
plotly_utility.offline.mpl_plot(fig)


fig = pux.histogram(
    segmentation_df.melt(
        id_vars="type",
        value_vars=["mean0", "mean1", "mean2"],
        var_name="mean", value_name="x"
    ),
    title="Mean Positions",
    x="x", facet_col="mean", facet_row="type",
    use_different_bin_widths=True, nbins=20,
    width=800, height=1000
)
fig.update_xaxes(title="depth", matches="x1", col=1)
fig.update_xaxes(title="height", matches="x2", col=2)
fig.update_xaxes(title="width", matches="x3", col=3)
plotly_utility.subplots.update_xaxes(fig, "inside", title=None)
plotly_utility.offline.mpl_plot(fig)


fig = pux.histogram(
    segmentation_df[segmentation_df["count"] < 1000],
    title="count < 1000",
    x="count", facet_row="type", log_y=True, nbins=100,
    width=800, height=1000
)
plotly_utility.offline.mpl_plot(fig)
plotly_utility.offline.mpl_plot(
    px.histogram(segmentation_df[segmentation_df["count"] < 1000].sort_values("type"), x="type")
)

