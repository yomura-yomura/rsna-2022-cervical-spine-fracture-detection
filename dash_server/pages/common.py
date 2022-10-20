import pathlib

import CSFD.data.io_with_cfg
from dash import dcc
import dash
import plotly.express as px
import warnings


cfg = CSFD.data.io.load_yaml_config("config.yaml", show_warning=False)
cfg.dataset.type_to_load = "npz"
df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
print(f"len(df) == {len(df):,}")
targets = [f"{uid} (#{i + 1})" for i, uid in enumerate(df["StudyInstanceUID"])]

# cfg.dataset.cropped_2d_labels_path = "../data/cropped_2d_labels/old/new_cropped_2d_labels.csv"
cfg.dataset.cropped_2d_labels_path = "../data/cropped_2d_labels/cropped_2d_labels.csv"

number_regex = r"(?:\d+(?:\.\d+)?)"
depth_range_regex = rf"\[({number_regex}),\W?({number_regex})\]"
common_shape_for_bb_regex = rf"\[({number_regex}|None),\W?({number_regex}|None),\W?({number_regex}|None)\]"

ss_model_dict = {
    "v0 (128x256x256, depth_range=[0.1, 0.9])": "UNet_128x256x256",
    "v1 (256x256x256)": "UNet_256x256x256"
}

stored_path = {
    "pictures_images_with_bb": pathlib.Path("../data/pictures_images_with_bb")
}
for k, v in stored_path.items():
    assert v.exists(), k


def get_dcc_graph(id_suffix):
    return dcc.Graph(
        id=f"my-graph-{id_suffix}", style={"height": "80vh"}, figure=px.scatter(template="plotly_dark")
    )


def query_to_dict(query):
    return dict(
        tuple(element.split("="))
        for element in query.split("#")
        if element != ""
    )


def dict_to_query(query_dict: dict):
    return "".join([
        f"#{k}={v}"
        for k, v in query_dict.items()
    ])


def parse_uid_query(uid):
    if uid is None:
        return None
    elif uid.isdigit():
        return targets[int(uid) - 1]
    else:
        matched = [target for target in targets if target.split(maxsplit=1)[0] == uid]
        if len(matched) == 0:
            raise ValueError(f"{uid} not found")
        elif len(matched) == 1:
            pass
        else:
            raise NotImplementedError
        return matched[0]


class Progress:
    def __init__(self, set_progress, total):
        self.set_progress = set_progress
        self.total = total - 1
        self.cnt = 0

    def step(self, *args):
        if self.cnt > self.total:
            raise StopIteration
        self.set_progress((self.cnt / self.total * 100, *args))
        self.cnt += 1

    def tqdm(self, iterator, *step_args):
        iterator = iter(iterator)
        for step in iterator:
            self.step(*step_args)
            yield step


import dash_bootstrap_components as dbc
from dash import html
from dash import Output


def get_elements_of_status_bar(suffix):
    return [
        dbc.Progress(id=f"my-progress-bar-{suffix}", value=0, striped=True, animated=True, style={"visibility": "hidden"}),
        html.Div(id=f"my-progressing-status-{suffix}", children="", style={"visibility": "hidden"})
    ]


def get_kwargs_to_callback_for_progress_bar(suffix):
    return dict(
        progress=[Output(f"my-progress-bar-{suffix}", "value"), Output(f"my-progressing-status-{suffix}", "children")],
        running=[
            (
                    Output(f"my-progress-bar-{suffix}", "style"),
                    {"visibility": "visible"},
                    {"visibility": "hidden"},
            ),
            (
                    Output(f"my-progressing-status-{suffix}", "style"),
                    {"visibility": "visible"},
                    {"visibility": "hidden"},
            )
        ],
        background=True
    )


def validate_layout_kwargs(given_kwargs: dict, default_query_kwargs: dict):
    unexpected_kwargs = set(given_kwargs.keys()) - set(default_query_kwargs.keys())
    if len(unexpected_kwargs) > 0:
        for key in unexpected_kwargs:
            warnings.warn(f"{key} not expected (value: {given_kwargs.pop(key)})", UserWarning)
    
    for key, default_value in default_query_kwargs.items():
        if key in given_kwargs.keys():
            given_value = given_kwargs[key]
            if key == "depth" and given_value == "original":
                value_to_replace = "0"
            elif key == "height" and given_value == "original":
                value_to_replace = "0"
            elif key == "width" and given_value == "original":
                value_to_replace = "0"
            else:
                value_to_replace = given_value
            given_kwargs[key] = value_to_replace
        else:
            given_kwargs[key] = default_value

        if key == "uid":
            given_kwargs[key] = parse_uid_query(given_kwargs[key])
    return given_kwargs


class NotSelected:
    def __str__(self):
        return "NotSelected"

    def __repr__(self):
        return self.__str__()

not_selected = NotSelected()


def validate_callback_kwargs(uid_dropdown_text: str = not_selected, ss_model_version: str = not_selected):
    ret = []
    if uid_dropdown_text is not not_selected:
        if uid_dropdown_text is None or uid_dropdown_text == "":
            raise dash.exceptions.PreventUpdate
        else:
            uid, _ = uid_dropdown_text.split()
            ret.append(uid)

    if ss_model_version is not not_selected:
        unet_ss_model_name = ss_model_dict[ss_model_version]
        ret.append(unet_ss_model_name)

    if len(ret) == 1:
        return ret[0]
    else:
        return ret


