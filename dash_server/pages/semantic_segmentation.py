import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import numpy as np
import dash
import warnings
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State
from .common import *
import pathlib
import pandas as pd
import CSFD.visualization.three_dimensions
import plotly.graph_objs as go


__version__ = "0.0"


dash.register_page(__name__)
module_uid = __name__.replace(".", "_")


def layout(
        uid="1", model_version="v1 (256x256x256)", **kwargs
):
    if len(kwargs) > 0:
        warnings.warn(f"{kwargs} not expected")

    default_target = parse_uid_query(uid)

    return html.Div([
        html.H2(f"Semantic Segmentation (v{__version__})"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=targets, value=default_target
        ),
        dcc.Dropdown(
            id="my-dropdown-ss-model-version",
            options=list(ss_model_dict.keys()),
            value=model_version
        ),
        html.Div([
            dbc.Row([
                dbc.Col([
                    *get_elements_of_status_bar(f"sh-{module_uid}"),
                    dcc.Loading(
                        id=f"loading-sh-{module_uid}",
                        children=get_dcc_graph(f"sh-{module_uid}"),
                        type="circle"
                    )
                ]),
                dbc.Col([
                    *get_elements_of_status_bar(f"3dss-{module_uid}"),
                    dcc.Loading(
                        id=f"loading-3dss-{module_uid}",
                        children=get_dcc_graph(f"3dss-{module_uid}"),
                        type="circle"
                    )
                ])
            ])
        ]),
        *get_elements_of_status_bar(f"sa-{module_uid}"),
        dcc.Loading(
            id=f"loading-sa-{module_uid}",
            children=get_dcc_graph(f"sa-{module_uid}"),
            type="circle"
        )
    ])


@callback(
    Output(component_id=f"my-graph-sa-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-ss-model-version", component_property="value")],
    **get_kwargs_to_callback_for_progress_bar(f"sa-{module_uid}")
)
def plot1(set_progress, target, ss_model_version):
    uid, unet_ss_model_name = validate_callback_kwargs(
        uid_dropdown_text=target, ss_model_version=ss_model_version
    )
    ss_model_path = pathlib.Path("../semantic_segmentation/models/") / unet_ss_model_name
    ss_model_uint8_path = ss_model_path / "predicted_data" / "uint8" / "fold0"
    progress = Progress(set_progress, 4)

    cfg.dataset.type_to_load = "dcm"

    ss_cfg = CSFD.data.io.load_yaml_config(ss_model_uint8_path / "UNet.yaml", show_warning=False)

    cfg.dataset.depth = ss_cfg.dataset.depth
    cfg.dataset.depth_range = ss_cfg.dataset.depth_range
    cfg.dataset.image_2d_shape = ss_cfg.dataset.image_2d_shape

    segmentations = np.load(ss_model_uint8_path / f"{uid}.npz")["arr_0"]

    progress.step("creating dataloader...")

    datamodule = CSFD.monai.datamodule.CSFDDataModule(cfg, df)
    datamodule.setup("predict")
    dataset : CSFD.monai.datamodule.CacheDataset = datamodule.test_dataset

    idx = [i for i, record in enumerate(dataset.data) if record["StudyInstanceUID"] == uid][0]
    record = dataset[idx]

    progress.step("creating slicing-animation figure...")

    slicing_animation_fig = px.imshow(
        record["data"][0].numpy(),
        animation_frame=0,
        color_continuous_scale="gray",
        range_color=[0, 1], template="plotly_dark"
    )
    slicing_animation_fig._data = (
        *slicing_animation_fig._data,
        *(
            go.Contour(
                name=f"C{i + 1}",
                # z=np.full(segmentations.shape[1:], np.nan),
                showlegend=True, visible="legendonly"
            )
            for i in range(segmentations.shape[0])
        )
    )

    progress.step("adding segmentations into figure...")
    assert len(slicing_animation_fig.frames) == segmentations.shape[1], (len(slicing_animation_fig.frames), segmentations.shape[1])
    for segmentation, frame in zip(
            np.rollaxis(segmentations, axis=1),
            slicing_animation_fig.frames
    ):
        frame.data = (
            *frame.data,
            *(
                go.Contour(
                    name=f"C{i + 1}",
                    z=seg, showscale=False,
                    contours=dict(start=0, end=1, size=2, coloring='lines'),
                    line=dict(width=2),
                    colorscale=[
                        [0, px.colors.qualitative.Plotly[i]],
                        [1, px.colors.qualitative.Plotly[i]]
                    ],
                    showlegend=True, visible=True if seg.any() else "legendonly"
                )
                if seg.any() else
                go.Contour(name=f"C{i + 1}", visible='legendonly')
                for i, seg in enumerate(segmentation)
            )
        )
    slicing_animation_fig.layout.coloraxis.colorbar.x = 0.9

    progress.step("fetching images from the server...")
    return slicing_animation_fig


def get_segmentations(target, ss_model_version):
    uid, unet_ss_model_name = validate_callback_kwargs(
        uid_dropdown_text=target, ss_model_version=ss_model_version
    )
    ss_model_path = pathlib.Path("../semantic_segmentation/models/") / unet_ss_model_name
    ss_model_uint8_path = ss_model_path / "predicted_data" / "uint8" / "fold0"
    segmentations = np.load(ss_model_uint8_path / f"{uid}.npz")["arr_0"]
    return segmentations


@callback(
    Output(component_id=f"my-graph-sh-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-ss-model-version", component_property="value")],
    **get_kwargs_to_callback_for_progress_bar(f"sh-{module_uid}")
)
def plot2(set_progress, target, ss_model_version):
    segmentations = get_segmentations(target, ss_model_version)
    progress = Progress(set_progress, 3)

    progress.step("creating a dataframe counting semantic-segmentation pixels...")

    ss_count_df = pd.DataFrame({
        f"C{i + 1}": np.count_nonzero(segmentations, axis=(-2, -1))[i]
        for i in range(7)
    }).stack().reset_index()

    progress.step("creating semantic-segmentation-pixels histogram figure...")

    ss_count_fig = px.bar(
        ss_count_df,
        x="level_0", y=0, color="level_1", barmode="overlay",
        labels={"level_0": "depth", "0": "count"},
        template="plotly_dark"
    )

    progress.step("fetching images from the server...")
    return ss_count_fig


@callback(
    Output(component_id=f"my-graph-3dss-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-ss-model-version", component_property="value")],
    **get_kwargs_to_callback_for_progress_bar(f"3dss-{module_uid}")
)
def plot3(set_progress, target, ss_model_version):
    segmentations = get_segmentations(target, ss_model_version)
    progress = Progress(set_progress, 2)

    progress.step("creating 3d-semantic-segmentation figure...")

    three_dimensions_ss_fig = CSFD.visualization.three_dimensions.get_fig_of_surface_mesh(
        segmentations
    )
    three_dimensions_ss_fig.update_layout(
        scene=dict(
            camera=dict(
                up=dict(x=1, y=0, z=0),
                eye=dict(zip(("x", "y", "z"), (0.5, 1, 1)))
            ),
            xaxis_title="bottom to up",
            yaxis_title="front to back?",
            zaxis_title="left to right?"
        )
    )

    progress.step("fetching images from the server...")
    return three_dimensions_ss_fig
