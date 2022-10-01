import CSFD.data.io
import CSFD.data.io_with_cfg
from dash import dcc
import plotly.express as px


print("* common.py loaded")

cfg = CSFD.data.io.load_yaml_config("config.yaml")
cfg.dataset.type_to_load = "both"
# cfg.dataset.type_to_load = "dcm"
df = CSFD.data.io_with_cfg.three_dimensions.get_df(cfg.dataset)
targets = [f"{uid} (#{i + 1})" for i, uid in enumerate(df["StudyInstanceUID"])]


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
    if uid.isdigit():
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

    def tqdm(self, iterator, *args):
        iterator = iter(iterator)
        for step in iterator:
            self.step(*args)
            yield step
