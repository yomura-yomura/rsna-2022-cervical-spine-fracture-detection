import numpy as np
from . import util as _util_module
import torch.nn as nn


def loss_fn(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


competition_weights = {
    k: np.array(v)
    for k, v in _util_module.competition_weights.items()
}


def competition_loss(y_hat: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    y_hat.shape = (batch_size, num_classes)
    y.shape = (batch_size, num_classes)
    """
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+']
    weights += (1 - y) * competition_weights['-']
    # return (loss * weights).sum(axis=-1).mean() / weights.sum()
    # return (loss * weights / np.sum(weights, axis=1, keepdims=True)).mean()

    loss *= weights
    loss /= np.sum(weights, axis=1, keepdims=True)
    return np.mean(np.sum(loss, axis=1))
