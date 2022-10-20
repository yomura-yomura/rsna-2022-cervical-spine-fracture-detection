import plotly.express as px
import numpy as np
import plotly.graph_objs as go
import itertools


def get_fig_of_image_with_segmentation(image, segmentation):
    assert np.ndim(image) == 2
    assert np.ndim(segmentation) == 2

    fig = px.imshow(image, color_continuous_scale="gray")
    fig.add_trace(
        px.imshow(
            np.ma.masked_equal(segmentation, 0).astype("f8").filled(np.nan)
        ).update_traces(opacity=0.3, coloraxis="coloraxis2").data[0]
    )
    fig.update_layout(
        coloraxis2=dict(
            colorscale=list(
                map(list,
                    zip(
                        np.linspace(0, 1, 7),
                        map(lambda hex_str: f"rgb{px.colors.hex_to_rgb(hex_str)}",
                            px.colors.qualitative.Plotly))
                    )
            ),
            showscale=True,
            colorbar=dict(x=0.9),
            cmin=1, cmax=7
        )
    )
    return fig


def _get_xyz_for_contour_plot(x_range, y_range, image_shape):
    x0, x1 = map(int, x_range)
    y0, y1 = map(int, y_range)

    x = np.arange(min(x0, 0), max(image_shape[1], x1 + 1))
    y = np.arange(min(y0, 0), max(image_shape[0], y1 + 1))
    z = np.zeros((len(x), len(y)))
    z[tuple(zip(*itertools.product(range(y0, y1 + 1), range(x0, x1 + 1))))] = 1
    return x, y, z


def get_bb_trace(x_range=None, y_range=None, image_shape=(256, 256), line_color=None, **kwargs):
    if x_range is None and y_range is None:
        x = y = z = None
    else:
        x, y, z = _get_xyz_for_contour_plot(x_range, y_range, image_shape)

    if line_color is None:
        line_color = px.colors.qualitative.Plotly[0]

    return go.Contour(
        # name=vertebrae_type,
        x=x, y=y, z=z,
        showscale=False,
        contours=dict(start=0, end=1, size=2, coloring='lines'),
        line=dict(width=1),
        colorscale=[
            [0, line_color],
            [1, line_color]
        ],
        # visible='legendonly',
        hoverinfo="skip",
        **kwargs
    )