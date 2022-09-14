import numpy as np


def get_3d_bounding_box(masks: np.ndarray):
    assert masks.ndim == 3
    indices = np.argwhere(masks)
    bb = np.stack([indices.min(axis=0), indices.max(axis=0)], axis=1)
    return bb
