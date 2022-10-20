import pathlib

import dash
from dash import html, dcc, dash_table
from dash import callback, Output, Input
import pandas as pd
import os
import datetime as dt
import dateutil.tz


dash.register_page(__name__)

columns = ["ref", "filename", "description", "submittedBy", "past_time"]

layout = html.Div([
    dcc.Interval(id="my-interval", interval=10_000),

    html.H3("Current Running Processes:"),
    html.Code(id="current-running-processes", children=""),
    html.Div(id="last-updated", children=""),

    html.Hr(),

    html.H3("Recorded Processes So Far:"),
    dash_table.DataTable(
        id="my-tbl", columns=[{"name": i, "id": i} for i in columns],
        sort_action="native", sort_mode="multi", sort_by=[{"column_id": "ref", "direction": "desc"}],
        filter_action="native",
        style_header={
            'backgroundColor': 'rgb(30, 30, 30)',
            'color': 'white'
        },
        style_filter={
            'backgroundColor': 'rgb(70, 70, 70)',
            'color': 'white'
        },
        style_data={
            'backgroundColor': 'rgb(50, 50, 50)',
            'color': 'white'
        }
        # css=[
        #     {"selector": ".dash-table-container", "rule": "--text-color: white !important"}
        # ]
    )
])

@callback(
    [Output('my-tbl', 'data'), Output("current-running-processes", "children"), Output("last-updated", "children")],
    Input('my-interval', 'n_intervals')
)
def update_graphs(_):
    df = pd.read_csv("../meta/results.tsv", sep="\t", names=columns)

    latest_status_fn = pathlib.Path("../meta/latest_status.txt")
    if latest_status_fn.exists():
        with open(latest_status_fn, "r") as f:
            latest_status = f.read()
            last_updated_at = dt.datetime.fromtimestamp(
                os.stat(f.name).st_mtime, dateutil.tz.gettz('Asia/Tokyo')
            ).isoformat(" ", "seconds")
            last_updated = f"last updated at {last_updated_at}"
    else:
        latest_status = ""
        last_updated = "No processes found"
    return df.to_dict("records"), latest_status, last_updated
