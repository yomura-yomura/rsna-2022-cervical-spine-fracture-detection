import plotly.express as px
import numpy as np


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
                        map(lambda hex: f"rgb{px.colors.hex_to_rgb(hex)}",
                            px.colors.qualitative.Plotly))
                    )
            ),
            showscale=True,
            colorbar=dict(x=0.9),
            cmin=1, cmax=7
        )
    )
    return fig
