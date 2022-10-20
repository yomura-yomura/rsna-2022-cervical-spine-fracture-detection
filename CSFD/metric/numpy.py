import numpy as np
from . import util as _util_module


def cross_entropy_loss(y_hat, y, eps=1e-15):
    y_hat = np.clip(y_hat, eps, 1 - eps)
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


competition_weights = {
    k: np.array(v, dtype="f8")
    for k, v in _util_module.competition_weights.items()
}


def competition_loss(y_hat: np.ndarray, y: np.ndarray, reduction="mean") -> np.ndarray:
    """
    y_hat.shape = (batch_size, num_classes)
    y.shape = (batch_size, num_classes)
    """
    loss_matrix = cross_entropy_loss(y_hat, y)
    weights = y * competition_weights['+']
    weights += (1 - y) * competition_weights['-']
    weights /= weights.sum(axis=1, keepdims=True)
    # return (loss * weights).sum(axis=-1).mean() / weights.sum()
    # return (loss * weights / np.sum(weights, axis=1, keepdims=True)).mean()

    loss_matrix *= weights
    # loss /= np.sum(weights, axis=1, keepdims=True)
    row_wise_losses = np.sum(loss_matrix, axis=1)
    if reduction == "mean":
        loss = np.mean(row_wise_losses)
    elif reduction is None:
        loss = row_wise_losses
    else:
        raise ValueError(reduction)
    return loss
    # return np.mean(loss)
    # return (loss * weights).mean()
