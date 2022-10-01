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

number_regex = r"(?:\d+(?:\.\d+)?)"
depth_range_regex = rf"\[({number_regex}),\W?({number_regex})\]"
common_shape_for_bb_regex = rf"\[({number_regex}|None),\W?({number_regex}|None),\W?({number_regex}|None)\]"


def layout(
        uid="0", preprocessing_type="normal",
        depth_range="[0.1, 0.9]", common_shape_for_bb = "[128, 256, 256]",
        depth="128", height="256", width="256",
        **kwargs
):
    if len(kwargs) > 0:
        warnings.warn(f"{kwargs} not expected")

    if depth == "original":
        depth = "0"
    if height == "original":
        height = "0"
    if width == "original":
        width = "0"

    default_target = parse_uid_query(uid)

    return html.Div([
        dcc.Store(id='my-parameters-cache', storage_type='local'),
        html.H2(f"Animations of Cropped 3D (v{__version__})"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=targets, value=default_target
        ),

        dbc.Input(id="my-input-depth-range", pattern=depth_range_regex, value=depth_range),
        dcc.Dropdown(id="my-dropdown-preprocessing-type", options=["normal", "voi_lut", "windowing"], value=preprocessing_type),
        dcc.Slider(
            id="my-slider-depth",
            min=0, max=512, value=int(depth),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dcc.Slider(
            id="my-slider-height",
            min=0, max=512, value=int(height),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dcc.Slider(
            id="my-slider-width",
            min=0, max=512, value=int(width),
            marks={0: {"label": "original"}, 128: {"label": "128"}, 256: {"label": "256"}, 512: {"label": "512"}},
            tooltip={"placement": "bottom", "always_visible": False}
        ),
        dbc.Input(id=f"my-input-common-shape-for-bb", pattern=common_shape_for_bb_regex, value=common_shape_for_bb),
        html.Div([
            dbc.Progress(id="my-progress-bar", value=0, striped=True, animated=True, style={"visibility": "hidden"}),
            html.Div(id="my-progressing-status", children="", style={"visibility": "hidden"}),
            dcc.Loading(
                id=f"loading-{module_uid}",
                children=get_dcc_graph(module_uid),
                type="circle"
            )
        ])
    ])



# @callback(
#     [Output(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
#      Output(component_id="my-dropdown-preprocessing-type", component_property="value"),
#      Output(component_id="my-slider-height", component_property="value"),
#      Output(component_id="my-slider-width", component_property="value")],
#     Input("my-parameters-cache", 'modified_timestamp'),
#     State("my-parameters-cache", 'data')
# )
# def on_data(ts, data):
#     if ts is None or data is None:
#         raise dash.exceptions.PreventUpdate
#     return data


@callback(
    Output(component_id=f"my-graph-{module_uid}", component_property="figure"),
    [Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
     Input(component_id="my-dropdown-preprocessing-type", component_property="value"),
     Input(component_id="my-input-depth-range", component_property="value"),
     Input(component_id=f"my-input-common-shape-for-bb", component_property="value"),
     Input(component_id="my-slider-depth", component_property="value"),
     Input(component_id="my-slider-height", component_property="value"),
     Input(component_id="my-slider-width", component_property="value")],
    progress=[Output("my-progress-bar", "value"), Output("my-progressing-status", "children")],
    running=[
        (
                Output("my-progress-bar", "style"),
                {"visibility": "visible"},
                {"visibility": "hidden"},
        ),
        (
                Output("my-progressing-status", "style"),
                {"visibility": "visible"},
                {"visibility": "hidden"},
        ),
        # (
        #     Output(f"my-graph-{module_uid}", "style"),
        #     {"visibility": "hidden"},
        #     {"visibility": "visible"},
        # )
    ],
    background=True
)
def plot(set_progress, target, preprocessing_type, depth_range, common_shape_for_bb, depth, height, width):
    if target is None:
        raise dash.exceptions.PreventUpdate
    uid, _ = target.split()

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

    cfg.dataset.common_shape_for_bb = common_shape_for_bb

    if height == 0 and width == 0:
        cfg.dataset.image_2d_shape = None
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

