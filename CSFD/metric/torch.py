"""
https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/341854#1884562
"""
import torch
import torch.nn as nn
from . import util as _util_module


# change it to nn.BCELoss(reduction='none') if you have sigmoid activation in last layer
loss_fn = nn.BCEWithLogitsLoss(reduction='none')
# loss_fn = nn.BCELoss(reduction='none')

competition_weights = {
    k: torch.tensor(v, dtype=torch.float, device=torch.device("cuda"))
    for k, v in _util_module.competition_weights.items()
}


def competition_loss(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    y_hat.shape = (batch_size, num_classes)
    y.shape = (batch_size, num_classes)
    """
    loss = loss_fn(y_hat, y)
    weights = y * competition_weights['+'] + (1 - y) * competition_weights['-']
    return torch.mean(
        torch.sum(loss * weights / torch.sum(weights, dim=1, keepdim=True), dim=1)
    )


# def competition_loss(y_pred_logit, y, reduction='mean', verbose=False):
#     """
#     Weighted loss
#     We reuse torch.nn.functional.binary_cross_entropy_with_logits here. pos_weight and weights combined give us necessary coefficients described in https://www.kaggle.com/competitions/rsna-2022-cervical-spine-fracture-detection/discussion/340392
#
#     See also this explanation: https://www.kaggle.com/code/samuelcortinhas/rsna-fracture-detection-in-depth-eda/notebook
#     """
#
#     neg_weights = (torch.tensor([7., 1, 1, 1, 1, 1, 1, 1]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1])).cuda()
#     pos_weights = (torch.tensor([14., 2, 2, 2, 2, 2, 2, 2]) if y_pred_logit.shape[-1] == 8 else torch.ones(y_pred_logit.shape[-1]) * 2.).cuda()
#
#     loss = torch.nn.functional.binary_cross_entropy_with_logits(
#         y_pred_logit,
#         y,
#         reduction='none',
#     )
#
#     if verbose:
#         print('loss', loss)
#
#     pos_weights = y * pos_weights.unsqueeze(0)
#     neg_weights = (1 - y) * neg_weights.unsqueeze(0)
#     all_weights = pos_weights + neg_weights
#
#     if verbose:
#         print('all weights', all_weights)
#
#     loss *= all_weights
#     if verbose:
#         print('weighted loss', loss)
#
#     norm = torch.sum(all_weights, dim=1).unsqueeze(1)
#     if verbose:
#         print('normalization factors', norm)
#
#     loss /= norm
#     if verbose:
#         print('normalized loss', loss)
#
#     loss = torch.sum(loss, dim=1)
#     if verbose:
#         print('summed up over patient_overall-C1-C7 loss', loss)
#
#     if reduction == 'mean':
#         return torch.mean(loss)
#     return loss
