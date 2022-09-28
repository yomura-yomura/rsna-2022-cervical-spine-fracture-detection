import warnings

import CSFD.data.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import numpy as np
import plotly.express as px
import dash
from dash import dcc, html, Input, Output, callback


dash.register_page(__name__)
uid = __name__.replace(".", "_")


cfg = CSFD.data.load_yaml_config("../monai_with_semantic_segmentation/SEResNext50.yaml")
df = CSFD.data.three_dimensions.get_df(cfg.dataset)
targets = [f"{uid} (#{i + 1})" for i, uid in enumerate(df["StudyInstanceUID"])]

layout = html.Div([
    html.H2("Image Preprocessing (v0.0)"),
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
    preprocessing_types = ["voi_lut"]

    labels = ["normal"] + preprocessing_types

    datamodule = CSFD.monai.datamodule.CSFDDataModule(cfg, df)
    datamodule.setup("predict")
    dataset : CSFD.monai.datamodule.CacheDataset = datamodule.test_dataset

    indices = [i for i, record in enumerate(dataset.data) if record["StudyInstanceUID"] == uid]
    assert len(indices) == 1
    idx = indices[0]

    data_list = [dataset[idx]["data"][0].numpy()]

    for p_type in preprocessing_types:
        copied_cfg = cfg.copy()
        if p_type == "voi_lut":
            copied_cfg.dataset.type_to_load = "dcm"
            copied_cfg.dataset.use_voi_lut = True
        else:
            warnings.warn(f"{p_type} not expected")
            continue

        datamodule = CSFD.monai.datamodule.CSFDDataModule(copied_cfg, df)
        datamodule.setup("predict")
        dataset: CSFD.monai.datamodule.CacheDataset = datamodule.test_dataset
        data_list.append(dataset[idx]["data"][0].numpy())

    fig = px.imshow(
        np.stack(data_list, axis=0),
        color_continuous_scale="gray", range_color=[0, 1],
        animation_frame=1, facet_col=0,
        template = "plotly_dark"
    )
    fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[1])]))

    return fig
