import warnings
import CSFD.data.io.three_dimensions
import CSFD.bounding_box
import CSFD.monai
import numpy as np
import dash
from dash import dcc, html, Input, Output, callback
from .common import *


dash.register_page(__name__)
module_uid = __name__.replace(".", "_")


# cfg = CSFD.data.io.load_yaml_config("../monai_with_ss_3d/SEResNext50.yaml")
# cfg.dataset.type_to_load = "both"
# df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
# targets = [f"{uid} (#{i + 1})" for i, uid in enumerate(df["StudyInstanceUID"])]

def layout(uid="0", **kwargs):
    if len(kwargs) > 0:
        warnings.warn(f"{kwargs} not expected")

    default_target = parse_uid_query(uid)
    return html.Div([
        html.H2("Image Preprocessing (v0.0)"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=targets, value=default_target
        ),
        html.Hr(),
        dcc.Loading(
            id=f"loading-{module_uid}",
            children=get_dcc_graph(module_uid),
            type="circle"
        )
    ])


@callback(
    Output(component_id=f"my-graph-{module_uid}", component_property="figure"),
    Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value"),
)
def dropdown_callback(target):
    if target is None:
        raise dash.exceptions.PreventUpdate
    uid, _ = target.split()
    preprocessing_types = ["voi_lut", "windowing"]

    labels = ["normal"] + preprocessing_types

    cfg.dataset.type_to_load = "npz"
    datamodule = CSFD.monai.datamodule.CSFDDataModule(cfg, df)
    datamodule.setup("predict")
    dataset : CSFD.monai.datamodule.CacheDataset = datamodule.test_dataset

    indices = [i for i, record in enumerate(dataset.data) if record["StudyInstanceUID"] == uid]
    assert len(indices) == 1
    idx = indices[0]

    data_list = [dataset[idx]["data"][0].numpy()]

    for p_type in preprocessing_types:
        copied_cfg = cfg.copy()
        copied_cfg.dataset.type_to_load = "dcm"
        if p_type == "voi_lut":
            copied_cfg.dataset.use_voi_lut = True
        elif p_type == "windowing":
            copied_cfg.dataset.use_windowing = True
        else:
            warnings.warn(f"{p_type} not expected")
            continue

        datamodule = CSFD.monai.datamodule.CSFDDataModule(copied_cfg, df)
        datamodule.setup("predict")
        dataset: CSFD.monai.datamodule.CacheDataset = datamodule.test_dataset
        data_list.append(dataset[idx]["data"][0].numpy())

    print("plot")
    fig = px.imshow(
        np.stack(data_list, axis=0),
        color_continuous_scale="gray", range_color=[0, 1],
        animation_frame=1, facet_col=0,
        template = "plotly_dark"
    )
    fig.for_each_annotation(lambda a: a.update(text=labels[int(a.text.split("=")[1])]))

    return fig
