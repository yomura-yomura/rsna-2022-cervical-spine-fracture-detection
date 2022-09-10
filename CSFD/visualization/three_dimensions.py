import plotly.graph_objs as go
import skimage.measure
import numpy as np


def get_fig_of_surface_mesh(data: np.ndarray, voxel_spacing=(1, 1, 1), layout=None, labels_to_be_excluded=None):
    if labels_to_be_excluded is None:
        labels_to_be_excluded = []
    target_labels = np.unique(data)
    assert len(target_labels) < 20  # prevent heavy figure

    fig = go.Figure(
        data=[
            get_surface_mesh_trace_from_voxel(data == target_label, name=f"{target_label}")
            for target_label in target_labels
            if target_label not in labels_to_be_excluded
        ],
        layout=layout
    )

    voxel_spacing = np.asarray(voxel_spacing)
    fig.update_layout(
        template="plotly_dark",
        scene=dict(
            aspectmode="manual",
            aspectratio=dict(zip(["x", "y", "z"], data.shape * voxel_spacing / max(data.shape * voxel_spacing))),
            xaxis=dict(title="", range=(0, data.shape[0])),
            yaxis=dict(title="", range=(0, data.shape[1])),
            zaxis=dict(title="", range=(0, data.shape[2]))
        ),
        legend_title="label"
    )
    fig.update_traces(showlegend=True, opacity=0.5)
    return fig


def get_surface_mesh_trace_from_voxel(data, offsets=(0, 0, 0), level=0, **kwargs):
    if np.unique(data).size == 1:
        return go.Mesh3d(**kwargs)
    verts, faces, *_ = skimage.measure.marching_cubes(data, level=level)

    return go.Mesh3d(
        x=verts[:, 0] + offsets[0], y=verts[:, 1] + offsets[1], z=verts[:, 2] + offsets[2],
        i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
        **kwargs
    )

