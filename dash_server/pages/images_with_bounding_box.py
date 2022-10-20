import os.path
import pathlib

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


__version__ = "0.0"


dash.register_page(__name__)
module_uid = __name__.replace(".", "_")

default_query_kwargs = dict(
    uid=None
)


def layout(**given_kwargs):
    given_kwargs = validate_layout_kwargs(given_kwargs, default_query_kwargs)
    available_uid_list = [p.parent.name for p in stored_path["pictures_images_with_bb"].glob("*/00.svg")]
    available_targets = [target for target in targets if target.split()[0] in available_uid_list]
    if given_kwargs["uid"] not in available_targets:
        warnings.warn(f"given uid '{given_kwargs['uid']}' is not available", UserWarning)
        given_kwargs["uid"] = available_targets[0]

    return html.Div([
        html.H2(f"Images with Bounding Box (v{__version__})"),
        dcc.Dropdown(
            id=f"my-input-uid-dropdown-{module_uid}",
            options=available_targets,
            value=given_kwargs["uid"],
            clearable=False
        ),
        html.Div([
            dbc.Carousel(id="my-carousel", items=[], controls=True, indicators=True)
        ])
    ])


@callback(
    Output(component_id="my-carousel", component_property="items"),
    Input(component_id=f"my-input-uid-dropdown-{module_uid}", component_property="value")
)
def plot(target):
    uid = validate_callback_kwargs(
        uid_dropdown_text=target
    )

    return [
        {"key": f"{i}", "src": os.path.join("/stored/pictures_images_with_bb", p.relative_to(stored_path["pictures_images_with_bb"]))}
        for i, p in enumerate(
            sorted((stored_path["pictures_images_with_bb"] / uid).glob("*.svg"), key=lambda p: int(p.name[:-4]))
        )
    ]