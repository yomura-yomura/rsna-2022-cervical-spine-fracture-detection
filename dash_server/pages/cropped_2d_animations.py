import os.path
import pathlib
import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import CSFD.visualization.two_dimensions
import plotly.graph_objs as go
import numpy as np
import re
import dash
import warnings
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State
from .common import *


__version__ = "0.0"


dash.register_page(__name__)
module_uid = __name__.replace(".", "_")

default_query_kwargs = dict(
    uid="1", model_version="v1 (256x256x256)"
)


def layout(**given_kwargs):
    given_kwargs = validate_layout_kwargs(given_kwargs, default_query_kwargs)

    return html.Div([
        html.H2(f"Animations of Cropped 2D (v{__version__})"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=[], value=given_kwargs["uid"],
            clearable=False
        ),
        dcc.Dropdown(
            id="my-dropdown-ss-model-version",
            options=list(ss_model_dict.keys()),
            value=given_kwargs["model_version"]
        ),
        html.Div([
            *get_elements_of_status_bar(f"{module_uid}"),
            dcc.Loading(
                id=f"loading-{module_uid}",
                children=get_dcc_graph(f"{module_uid}"),
                type="circle"
            )
        ])
    ])


def get_cropped_2d_images_path(unet_ss_model_name):
    new_cropped_2d_labels = f"../data/cropped_2d_images/{unet_ss_model_name}"
    assert os.path.exists(new_cropped_2d_labels), new_cropped_2d_labels
    return new_cropped_2d_labels


@callback(
    Output(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="options"),
    Input(component_id="my-dropdown-ss-model-version", component_property="value")
)
def change_uid_list(ss_model_version):
    unet_ss_model_name = validate_callback_kwargs(
        ss_model_version=ss_model_version
    )
    cropped_2d_images_path = pathlib.Path(get_cropped_2d_images_path(unet_ss_model_name))
    available_uid_list = tuple(p.name[:-4] for p in cropped_2d_images_path.glob("*.npz"))
    return [target for target in targets if target.split()[0] in available_uid_list]


@callback(
    Output(component_id=f"my-graph-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-ss-model-version", component_property="value")],
    **get_kwargs_to_callback_for_progress_bar(f"{module_uid}")
)
def plot(set_progress, target, ss_model_version):
    uid, unet_ss_model_name = validate_callback_kwargs(
        uid_dropdown_text=target, ss_model_version=ss_model_version
    )

    progress = Progress(set_progress, 6)
    progress.step("creating dataloader...")

    cfg.dataset.semantic_segmentation_bb_path = f"../semantic_segmentation/models/{unet_ss_model_name}/semantic_segmentation_bb/train_semantic_segmentation_bb_fold0.csv"
    cfg.dataset.common_shape_for_bb = [None, 256, 256]
    cfg.dataset.cropped_2d_images_path = get_cropped_2d_images_path(unet_ss_model_name)

    datamodule = CSFD.monai.datamodule.CSFDCropped2DDataModule(cfg, df)
    datamodule.setup("predict")
    dataset = datamodule.test_dataset

    progress.step("stacking data...")

    assert np.all(cfg.dataset.target_columns[1:] == [f"C{i}" for i in np.arange(7) + 1])

    matched_data = dataset.get_matched_with_uid(uid)
    data_list = []
    for vertebrae_type in cfg.dataset.target_columns[1:]:
        sel = matched_data["vertebrae_type"] == vertebrae_type
        data_list.append(matched_data["data"][sel][np.argsort(matched_data["slice_number"][sel])])

    progress.step("padding data with zeros...")

    max_depth = max(data.shape[0] for data in data_list)
    data_list = [
        np.concatenate([data, np.zeros((max_depth - data.shape[0], 256, 256))])
        # np.pad(data, [(0, max_depth - data.shape[0]), (0, 0), (0, 0)])
        for data in data_list
    ]

    progress.step("creating figure...")

    stacked_data = np.stack(data_list, axis=0)

    facet_col_wrap = 4
    fig = px.imshow(
        stacked_data,
        facet_col=0, facet_col_wrap=facet_col_wrap,
        animation_frame=1,
        color_continuous_scale="gray",
        range_color=[0, 1], template="plotly_dark"
    )
    fig.data = fig.data[:7]
    fig.for_each_annotation(lambda a: a.update(text=f'C{int(a.text.split("=")[1]) + 1}'))

    # fig.layout.sliders[0]["active"] = 20
    # # fig._data = fig.frames[20].data
    #
    # import plotly.graph_objs as go
    # fig2 = go.Figure(frames=fig.frames, layout=fig.layout)
    # for trace in fig.frames[20].data:
    #     fig2.add_trace(trace)
    # fig2._grid_ref = fig._grid_ref

    progress.step("adding bounding box to figure...")
    record: dict = dataset.dataset[int(np.argmax(dataset.dataset_uid_array == uid))]
    org_data_shape = record["data"].shape[1:]
    del record

    ss_bb_df = dataset.semantic_segmentation_bb_df.set_index(["StudyInstanceUID", "slice_number", "type"])
    ss_bb_df = ss_bb_df.loc[(uid, matched_data["slice_number"], matched_data["vertebrae_type"])].reset_index()
    ss_bb_df["slice_number_in_org"] = dataset.scale_default_depths_fitting_to_current_depth(matched_data["slice_number"], org_data_shape[0])
    for i, (vertebrae_type, line_color) in enumerate(zip(cfg.dataset.target_columns[1:], px.colors.qualitative.Plotly)):
        row = 2 - i // facet_col_wrap
        col = 1 + i % facet_col_wrap
        fig.add_trace(
            CSFD.visualization.two_dimensions.get_bb_trace(
                name=vertebrae_type,
                line_color=line_color,
                visible='legendonly'
            ),
            row=row, col=col
        )

        matched_ss_bb_df = ss_bb_df[ss_bb_df["type"] == vertebrae_type].sort_values("slice_number")
        for idx, frame in enumerate(fig.frames):
            copied_trace = go.Contour(fig.data[-1])

            if idx < len(matched_ss_bb_df):
                ss_bb_idx = matched_ss_bb_df.index[idx]

                try:
                    matched_c2d_labels_df = dataset.cropped_2d_labels_df.loc[
                        (uid, matched_ss_bb_df.loc[ss_bb_idx, "slice_number_in_org"])
                    ]
                except KeyError:
                    matched_c2d_labels_df = None

                if matched_c2d_labels_df is not None:
                    x0 = matched_c2d_labels_df["x"]
                    y0 = matched_c2d_labels_df["y"]
                    x1 = x0 + matched_c2d_labels_df["width"]
                    y1 = y0 + matched_c2d_labels_df["height"]

                    x0_to_crop, x1_to_crop, y0_to_crop, y1_to_crop = dataset.calc_2d_ranges_to_crop(
                        *matched_ss_bb_df.loc[ss_bb_idx, ["x0", "x1", "y0", "y1"]],
                        org_data_shape[1], org_data_shape[2]
                    )

                    x_side_margin = 0.5 * max(256 - (x1_to_crop - x0_to_crop), 0)
                    y_side_margin = 0.5 * max(256 - (y1_to_crop - y0_to_crop), 0)

                    x0 -= x0_to_crop
                    x1 -= x0_to_crop
                    y0 -= y0_to_crop
                    y1 -= y0_to_crop
                    x0 += x_side_margin
                    x1 += x_side_margin
                    y0 += y_side_margin
                    y1 += y_side_margin

                    # x0, y0 = y0, x0
                    # x1, y1 = y1, x1
                    # x0 = int(x0)
                    # x1 = int(x1)
                    # y0 = int(y0)
                    # y1 = int(y1)
                    x_range = (x0, x1)
                    y_range = (y0, y1)
                    print(idx, x_range, y_range)
                    # x = np.arange(min(x0, 0), max(256, x1 + 1))
                    # y = np.arange(min(y0, 0), max(256, y1 + 1))
                    # z = np.zeros((len(x), len(y)))
                    # z[np.isin(x, np.arange(x0, x1 + 1)), np.isin(y, y0)] = 1
                    # z[np.isin(x, np.arange(x0, x1 + 1)), np.isin(y, y1)] = 1
                    # z[np.isin(x, x0), np.isin(y, np.arange(y0, y1 + 1))] = 1
                    # z[np.isin(x, x1), np.isin(y, np.arange(y0, y1 + 1))] = 1
                    #
                    x, y, z = CSFD.visualization.two_dimensions._get_xyz_for_contour_plot(
                        x_range, y_range, image_shape=(256, 256)
                    )
                    copied_trace.x = x
                    copied_trace.y = y
                    copied_trace.z = z
                    copied_trace.visible = True

            frame.data += (copied_trace,)

    progress.step("fetching images from the server...")
    return fig