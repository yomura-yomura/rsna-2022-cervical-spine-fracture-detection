import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import numpy as np
import itertools
import tqdm
import plotly.express as px
import plotly_utility
import pandas as pd
import pathlib


def get_semantic_segmentation_bb_df(df):
    df_list = []
    for uid, p in zip(df["StudyInstanceUID"], tqdm.tqdm(df["npz_segmentations_path"])):
        for vertex_i, segmentations in enumerate(np.load(p)["arr_0"]):
            if segmentations.any() == np.False_:
                print("counts == 0")
                continue
            depth_range, height_range, width_range = CSFD.bounding_box.get_3d_bounding_box(segmentations)
            counts = np.count_nonzero(segmentations, axis=(-2, -1))
            df = pd.DataFrame({
                "StudyInstanceUID": uid,
                "slice_number": np.arange(depth_range[0], depth_range[1] + 1),
                "type": f"C{vertex_i + 1}",
                "x0": height_range[0], "x1": height_range[1],
                "y0": width_range[0], "y1": width_range[1],
                "count": counts[depth_range[0]: depth_range[1] + 1]
            })
            df_list.append(df)
    return pd.concat(df_list)


cfg = CSFD.data.io.load_yaml_config("SEResNext50.yaml")

train_segmentations_root_path = pathlib.Path("../semantic_segmentation/models/UNet_128x256x256")
cfg.dataset.train_segmentations_path = train_segmentations_root_path / "predicted_data/uint8/fold0/"

df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)


# train_segmentations_path = pathlib.Path(cfg.dataset.train_segmentations_path)

# train_segmentations_root_path = pathlib.Path("../semantic_segmentation/models/UNet_256x256x256")
train_segmentations_predicted_uint8_path = train_segmentations_root_path / "predicted_data" / "uint8" / "fold0"

for name in (p.name for p in train_segmentations_predicted_uint8_path.parent.glob("fold*") if p.is_dir()):
    train_segmentations_predicted_uint8_path = train_segmentations_predicted_uint8_path.with_name(name)
    target_csv_path = train_segmentations_predicted_uint8_path.with_name(
        f"segmentations-info-{train_segmentations_predicted_uint8_path.name}.csv"
    )

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

    target_segmentation_bb_root_path = train_segmentations_root_path / "semantic_segmentation_bb"
    target_segmentation_bb_root_path.mkdir(exist_ok=True)

    target_segmentation_bb_path = target_segmentation_bb_root_path / f"train_semantic_segmentation_bb_{name}.csv"

    if target_segmentation_bb_path.exists():
        semantic_segmentation_bb_df = pd.read_csv(target_segmentation_bb_path)
    else:
        print(f"creating {target_segmentation_bb_path}")
        semantic_segmentation_bb_df = get_semantic_segmentation_bb_df(df)
        semantic_segmentation_bb_df.to_csv(target_segmentation_bb_path, index=False)

    print(name)
    break


ss_cfg = CSFD.data.io.load_yaml_config(train_segmentations_predicted_uint8_path / "UNet.yaml")
# new_df.loc[:, df.columns] = df

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

for i in range(3):
    segmentation_df[f"min{i}"] = segmentation_df[f"mean{i}"] - segmentation_df[f"shape{i}"] / 2
    segmentation_df[f"max{i}"] = segmentation_df[f"mean{i}"] + segmentation_df[f"shape{i}"] / 2


for i, size in enumerate([ss_cfg.dataset.depth, *ss_cfg.dataset.image_2d_shape]):
    segmentation_df[f"min_scale{i}"] = segmentation_df[f"min{i}"] / size
    segmentation_df[f"max_scale{i}"] = segmentation_df[f"max{i}"] / size

    plotly_utility.offline.mpl_plot(
        pux.histogram(
            segmentation_df.melt(
                "type", [f"min_scale{i}", f"max_scale{i}"], value_name=f"min/max_scale{i}"
            ),
            x=f"min/max_scale{i}", color="type"
        ).update_xaxes(range=[0, 1], dtick=0.1)
    )


for i in range(3):
    print(np.nanquantile(segmentation_df[f"shape{i}"], [0.8, 0.9, 0.95, 0.99]))

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


count_zero_segmentations_df = segmentation_df[segmentation_df["count"] == 0]["name"].value_counts().reset_index()
print(count_zero_segmentations_df)
uid_list = count_zero_segmentations_df["index"].map(lambda npz_name: npz_name[:-4])

uid = uid_list[0]
npz_path = df[df["StudyInstanceUID"] == uid]["np_images_path"].iloc[0]
images = np.load(npz_path)["arr_0"]
import plotly.express as px
import plotly_utility

fig = px.imshow(images[60], title=uid, color_continuous_scale=px.colors.sequential.gray)
plotly_utility.offline.mpl_plot(fig)



segmentation_df