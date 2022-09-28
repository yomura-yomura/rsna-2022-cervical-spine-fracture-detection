import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback


dash.register_page(__name__)
uid = __name__.replace(".", "_")


cfg = CSFD.data.io.load_yaml_config("../monai_with_semantic_segmentation/SEResNext50.yaml")
df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
targets = [f"{uid} (#{i + 1})" for i, uid in enumerate(df["StudyInstanceUID"])]

layout = html.Div([
    html.H2("Animations of Cropped 3D (v0.0)"),
    html.Hr(),
    dcc.Dropdown(
        id=f"my-input-dropdown-{uid}",
        options=targets, value=targets[3]
    ),
    dcc.Loading(
        id=f"loading-{uid}",
        children=[
            dcc.Graph(id=f"my-graph-{uid}", style={"height": "80vh"})
        ],
        type="circle"
    )
])


@callback(
    Output(component_id=f"my-graph-{uid}", component_property="figure"),
    Input(component_id=f"my-input-dropdown-{uid}", component_property="value"),
)
def dropdown_callback(target):
    uid, _ = target.split()

    datamodule = CSFD.monai.datamodule.CSFDCropped3DDataModule(cfg, df)
    datamodule.setup("predict")
    dataset : CSFD.monai.datamodule.Cropped3DDataset = datamodule.test_dataset

    indices = np.where(dataset.unique_id[:, 0] == uid)[0]
    types = dataset.unique_id[indices, 1]
    assert np.all(types == [f"C{i}" for i in np.arange(7) + 1]), types

    data_list = [dataset[idx]["data"][0].numpy() for idx in indices]

    fig = px.imshow(
        np.stack(data_list, axis=0),
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

    return fig