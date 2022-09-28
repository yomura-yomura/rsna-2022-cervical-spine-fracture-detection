from dash import Dash, html, dcc
import dash
import dash_auth


app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"}
    ],
    use_pages=True
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

	dash.page_container
])




if __name__ == '__main__':
    app.run(
        debug=True, dev_tools_hot_reload=False,
        port=8888, host="0.0.0.0"
    )
