import numpy as np
import torch
import torch.nn as nn

_competition_weights = {
    '-': [7, 1, 1, 1, 1, 1, 1, 1],
    '+': [14, 2, 2, 2, 2, 2, 2, 2]
}


def loss_fn(y_hat, y):
    return -(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))


competition_weights = {
    k: np.array(v)
    for k, v in _competition_weights.items()
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

def weighted_loss(y_pred_logit, y, reduction='mean', verbose=False,DEVICE ='cuda'):
    """
    Weighted loss
    We reuse torch.nn.functional.binary_cross_entropy_with_logits here. pos_weight and weights combined give us necessary coefficients described in https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392

    See also this explanation: https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda/notebook
    """

    neg_weights = (torch.tensor([7., 1, 1, 1, 1, 1, 1, 1]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1])).to(DEVICE)
    pos_weights = (torch.tensor([14., 2, 2, 2, 2, 2, 2, 2]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1]) * 2.).to(DEVICE)

    loss = torch.nn.functional.binary_cross_entropy_with_logits(
        y_pred_logit,
        y,
        reduction='none',
        pos_weight=torch.as_tensor(2)
    )

    if verbose:
        print('loss', loss)

    pos_weights = y * pos_weights.unsqueeze(0)
    neg_weights = (1 - y) * neg_weights.unsqueeze(0)
    all_weights = pos_weights + neg_weights

    if verbose:
        print('all weights', all_weights)

    loss *= all_weights
    if verbose:
        print('weighted loss', loss)

    norm = torch.sum(all_weights, dim=1).unsqueeze(1)
    if verbose:
        print('normalization factors', norm)

    loss /= norm
    if verbose:
        print('normalized loss', loss)

    loss = torch.sum(loss, dim=1)
    if verbose:
        print('summed up over patient_overall-C1-C7 loss', loss)

    if reduction == 'mean':
        return torch.mean(loss)
    return loss