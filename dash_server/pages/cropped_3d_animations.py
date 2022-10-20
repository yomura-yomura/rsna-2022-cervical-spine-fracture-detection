import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import numpy as np
import re
import dash
import warnings
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, callback, State
from .common import *


__version__ = "1.0"


dash.register_page(__name__)
module_uid = __name__.replace(".", "_")

default_query_kwargs = dict(
    uid="1", model_version="v1 (256x256x256)", preprocessing_type="windowing",
    depth_range="[0, 1]", common_shape_for_bb="[128, 256, 256]",
    depth="256", height="256", width="256"
)


def layout(**given_kwargs):
    given_kwargs = validate_layout_kwargs(given_kwargs, default_query_kwargs)

    return html.Div([
        html.H2(f"Animations of Cropped 3D (v{__version__})"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=targets, value=given_kwargs["uid"],
            clearable=False
        ),
        dcc.Dropdown(
            id="my-dropdown-ss-model-version",
            options=list(ss_model_dict.keys()),
            value=given_kwargs["model_version"]
        ),
        dbc.Input(id="my-input-depth-range", pattern=depth_range_regex, value=given_kwargs["depth_range"]),
        dcc.Dropdown(
            id=f"my-dropdown-preprocessing-type-{module_uid}",
            options=["normal", "voi_lut", "windowing"], value=given_kwargs["preprocessing_type"]
        ),
        dcc.Slider(
            id="my-slider-depth",
            min=0, max=512, value=int(given_kwargs["depth"]),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dcc.Slider(
            id="my-slider-height",
            min=0, max=512, value=int(given_kwargs["height"]),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dcc.Slider(
            id="my-slider-width",
            min=0, max=512, value=int(given_kwargs["width"]),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dbc.Input(
            id=f"my-input-common-shape-for-bb", pattern=common_shape_for_bb_regex,
            value=given_kwargs["common_shape_for_bb"]
        ),
        html.Div([
            *get_elements_of_status_bar(f"{module_uid}"),
            # dbc.Progress(id="my-progress-bar", value=0, striped=True, animated=True, style={"visibility": "hidden"}),
            # html.Div(id="my-progressing-status", children="", style={"visibility": "hidden"}),
            dcc.Loading(
                id=f"loading-{module_uid}",
                children=get_dcc_graph(f"{module_uid}"),
                type="circle"
            )
        ])
    ])


@callback(
    Output(component_id=f"my-graph-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-ss-model-version", component_property="value"),
     Input(component_id=f"my-dropdown-preprocessing-type-{module_uid}", component_property="value"),
     Input(component_id="my-input-depth-range", component_property="value"),
     Input(component_id=f"my-input-common-shape-for-bb", component_property="value"),
     Input(component_id="my-slider-depth", component_property="value"),
     Input(component_id="my-slider-height", component_property="value"),
     Input(component_id="my-slider-width", component_property="value")],
    **get_kwargs_to_callback_for_progress_bar(f"{module_uid}")
)
def plot(set_progress, target, ss_model_version, preprocessing_type, depth_range, common_shape_for_bb, depth, height, width):
    uid, unet_ss_model_name = validate_callback_kwargs(
        uid_dropdown_text=target, ss_model_version=ss_model_version
    )

    depth_range = list(map(float, re.match(depth_range_regex, depth_range).groups()))
    common_shape_for_bb = list(map(eval, re.match(common_shape_for_bb_regex, common_shape_for_bb).groups()))

    cfg.dataset.type_to_load = "dcm"

    progress = Progress(set_progress, 7 + 2)

    if depth == 0:
        cfg.dataset.depth = None
    else:
        cfg.dataset.depth = depth

    if depth_range == [0, 1]:
        cfg.dataset.depth_range = None
    else:
        cfg.dataset.depth_range = depth_range

    cfg.dataset.semantic_segmentation_bb_path = f"../semantic_segmentation/models/{unet_ss_model_name}/semantic_segmentation_bb/train_semantic_segmentation_bb_fold0.csv"
    cfg.dataset.common_shape_for_bb = common_shape_for_bb

    if height == 0 and width == 0:
        cfg.dataset.image_2d_shape = None
    elif height == 0 or width == 0:
        raise dash.exceptions.PreventUpdate("both/neither of height and/nor width must be None/non-None")
    else:
        cfg.dataset.image_2d_shape = [height, width]

    if preprocessing_type == "normal":
        cfg.dataset.type_to_load = "dcm"
        cfg.dataset.use_voi_lut = False
        cfg.dataset.use_windowing = False
    elif preprocessing_type == "voi_lut":
        cfg.dataset.use_voi_lut = True
        cfg.dataset.use_windowing = False
    elif preprocessing_type == "windowing":
        cfg.dataset.type_to_load = "npz"
        cfg.dataset.use_voi_lut = False
        cfg.dataset.use_windowing = True
    elif preprocessing_type is None:
        raise dash.exceptions.PreventUpdate
    else:
        raise ValueError(preprocessing_type)

    progress.step("creating dataloader...")

    datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
    datamodule.setup("predict")
    dataset : CSFD.monai.datamodule.Cropped3DDataset = datamodule.test_dataset

    indices = np.where(dataset.unique_id[:, 0] == uid)[0]
    types = dataset.unique_id[indices, 1]
    assert np.all(types == [f"C{i}" for i in np.arange(7) + 1]), types

    data_list = [dataset[idx]["data"][0].numpy() for idx in progress.tqdm(indices, "loading images...")]
    stacked_data = np.stack(data_list, axis=0)
    fig = px.imshow(
        stacked_data,
        facet_col=0, facet_col_wrap=4,
        animation_frame=1,
        color_continuous_scale="gray",
        range_color=[0, 1], template="plotly_dark"
    )
    fig.for_each_annotation(lambda a: a.update(text=f'C{int(a.text.split("=")[1]) + 1}'))

    # fig.layout.sliders[0]["active"] = 20
    # # fig._data = fig.frames[20].data
    #
    # import plotly.graph_objs as go
    # fig2 = go.Figure(frames=fig.frames, layout=fig.layout)
    # for trace in fig.frames[20].data:
    #     fig2.add_trace(trace)
    # fig2._grid_ref = fig._grid_ref

    progress.step("fetching images from the server...")
    return fig

