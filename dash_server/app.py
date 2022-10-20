import flask
from dash import Dash, html, dcc
import dash
import dash_bootstrap_components as dbc
from dash import DiskcacheManager, CeleryManager
import os
import warnings
from pages.common import *


with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import dash_auth


if 'REDIS_URL' in os.environ:
    # Use Redis & Celery if REDIS_URL set as an env variable
    from celery import Celery
    celery_app = Celery(__name__, broker=os.environ['REDIS_URL'], backend=os.environ['REDIS_URL'])
    background_callback_manager = CeleryManager(celery_app)
else:
    # Diskcache for non-production apps when developing locally
    import diskcache
    cache = diskcache.Cache("./cache")
    background_callback_manager = DiskcacheManager(cache)


app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    use_pages=True,
    background_callback_manager=background_callback_manager,
    external_stylesheets=[
        # dbc.themes.GRID,
        dbc.themes.BOOTSTRAP,
        "https://cdn.jsdelivr.net/npm/bootstrap-dark-5@1.1.3/dist/css/bootstrap-night.min.css"
    ]
)
dash_auth.BasicAuth(app, {"inochi_wo_moyase": "X!6aXRe%Wug"})

app.title = "CSFD Visualization"

app.layout = html.Div([
	html.H1(app.title),

    html.Div(
        [
            html.Div(
                dcc.Link(
                    f"{page['name']}", href=page["relative_path"]
                )
            )
            for page in dash.page_registry.values()
        ]
    ),
    html.Hr(),

    dash.page_container
])

@app.server.route("/stored/pictures_images_with_bb/<uid>/<image_path>")
def serve_pictures_of_images_with_bb(uid, image_path):
    return flask.send_from_directory(stored_path["pictures_images_with_bb"] / uid, image_path)



if __name__ == '__main__':
    app.run(
        debug=True, dev_tools_hot_reload=False,
        port=8888, host="0.0.0.0"
    )
