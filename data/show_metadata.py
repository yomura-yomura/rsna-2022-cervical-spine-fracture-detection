import pandas as pd
import pathlib
import CSFD.data.io_with_cfg
import CSFD.visualization.two_dimensions
import omegaconf
import tqdm
import numpy as np
import plotly.express as px
import plotly_utility


cfg = omegaconf.DictConfig(dict(
    dataset=dict(
        data_root_path="rsna-2022-cervical-spine-fracture-detection",
        type_to_load="npz",
        type="train",
        target_columns=None,
        train_3d_images="3d_train_images_v4",
        use_windowing=True
    )
))
CSFD.data.io.add_default_values_if_not_defined(cfg, show_warnings=False)
print(cfg)

train_bb_df = pd.read_csv(pathlib.Path(cfg.dataset.data_root_path) / "train_bounding_boxes.csv")
df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
df = df[df["StudyInstanceUID"].isin(train_bb_df["StudyInstanceUID"])]

target_fn = pathlib.Path("cropped_2d_labels/cropped_2d_labels.csv")
if target_fn.exists():
    train_bb_df = pd.read_csv(target_fn)
else:
    for uid, dcm_images_path in tqdm.tqdm(df.groupby(["StudyInstanceUID", "dcm_images_path"]).indices):
        paths = CSFD.data.io.three_dimensions.get_dicom_paths(dcm_images_path)
        is_reversed = int(paths[0].name[:-4]) < int(paths[-1].name[:-4])
        train_bb_df.loc[
            train_bb_df["StudyInstanceUID"] == uid,
            "is_reversed"
        ] = is_reversed
        slice_number_mapping = {
            int(p.name[:-4]): i
            for i, p in enumerate(paths)
        }
        train_bb_df.loc[
            train_bb_df["StudyInstanceUID"] == uid,
            "slice_number"
        ] = train_bb_df.loc[
            train_bb_df["StudyInstanceUID"] == uid,
            "slice_number"
        ].map(slice_number_mapping)

    train_bb_df.to_csv(target_fn, index=False)




count_df = train_bb_df.groupby("StudyInstanceUID")["slice_number"].count()


# uid = "1.2.826.0.1.3680043.21561"
# uid = count_df.index[2]
# uid: str = train_bb_df[train_bb_df["is_reversed"]]["StudyInstanceUID"].unique()[0]


for uid in tqdm.tqdm(train_bb_df["StudyInstanceUID"].unique()):
    output_dir_path = pathlib.Path("pictures_images_with_bb") / uid
    output_dir_path.mkdir(exist_ok=True, parents=True)

    images = np.load(df[df["StudyInstanceUID"] == uid]["np_images_path"].iloc[0])["arr_0"]
    matched_bb_df = train_bb_df[train_bb_df["StudyInstanceUID"] == uid]

    max_n_pics_per_image = 20
    n_pics = int(np.ceil(len(matched_bb_df) / max_n_pics_per_image))


    for i_pic in range(n_pics):
        output_fn_path = output_dir_path / f"{i_pic:>02d}.svg"
        # if output_fn_path.exists():
        #     continue
        matched_images = images[matched_bb_df["slice_number"]][max_n_pics_per_image * i_pic: max_n_pics_per_image * (i_pic + 1)]

        default_facet_col_wrap = 5
        n_rows = matched_images.shape[0] // default_facet_col_wrap + 1
        if n_rows == 1:
            n_cols = matched_images.shape[0]
        else:
            n_cols = default_facet_col_wrap

        fig = px.imshow(
            matched_images,
            title=f"{uid} (#{i_pic + 1} of {n_pics})",
            facet_col=0, facet_col_wrap=n_cols,
            color_continuous_scale="gray",
            width=2100 * n_cols // default_facet_col_wrap,
            height=2000 * n_rows // (max_n_pics_per_image // default_facet_col_wrap),
            template="plotly_dark"
        )

        for i_subplot, (row, col) in enumerate(plotly_utility.subplots.get_subplot_coordinates(fig)):
            df_idx = max_n_pics_per_image * i_pic + i_subplot
            if len(matched_bb_df) <= df_idx:
                continue
            record = matched_bb_df.iloc[df_idx]

            fig.add_trace(
                CSFD.visualization.two_dimensions.get_bb_trace(
                    x_range=(record["x"], record["x"] + record["width"]),
                    y_range=(record["y"], record["y"] + record["height"]),
                    image_shape=images.shape[1:]
                ),
                row=row, col=col
            )
            # break
        fig.write_image(output_fn_path)
    # plotly_utility.offline.mpl_plot(fig, scale=10)

