"""
The objective function of FactorGCL: the multi-period mean squared error of Eq. 12 and the
temporal residual contrastive loss of Eq. 11, combined as ``L = L_mse + gamma * L_CL``
(Eq. 13).

The contrastive loss is the InfoNCE loss of Oord, Li and Vinyals (2018) applied *across
time at the stock node level*: the past and the future alpha embedding of the same stock
form a positive pair, the embeddings of different stocks the negative pairs. The intuition
the article gives is that once every factor has been removed, what is left of a stock is
idiosyncratic to it and so should be recognisable in the following period.
"""

import torch
import torch.nn.functional as F

# defaults of the supplementary material ("Implementation Details")
DEFAULT_TEMPERATURE = 0.1   # tau
DEFAULT_GAMMA = 0.1         # the weight of the contrastive loss in Eq. 13


def info_nce_loss(alpha_past, alpha_future, projection=None, temperature=DEFAULT_TEMPERATURE):
    """The temporal residual contrastive loss of Eq. 11.

    ``alpha_past`` and ``alpha_future`` are the ``(N, H)`` alpha embeddings of the same
    cross-section of stocks over the historical and the future window, in the same order:
    row ``i`` of both is stock ``i``, which is what makes the diagonal the positive pairs.
    ``projection`` is the 2-layer MLP ``p`` of Eq. 11 (skipped when ``None``).

    The loss is the cross-entropy of the row-wise softmax over cosine similarities divided
    by ``temperature``, averaged over the ``N`` stocks.
    """
    if alpha_past.shape != alpha_future.shape:
        raise ValueError("the past and future alpha embeddings must have the same shape")
    if alpha_past.dim() != 2:
        raise ValueError("the alpha embeddings must be (N, H)")
    if alpha_past.shape[0] < 2:
        # a single stock has no negative pair, so the loss is identically zero
        return alpha_past.new_zeros(())
    if projection is not None:
        alpha_past = projection(alpha_past)
        alpha_future = projection(alpha_future)
    past = F.normalize(alpha_past, dim=1)
    future = F.normalize(alpha_future, dim=1)
    logits = past @ future.transpose(0, 1) / temperature       # sim(p(e_i), p(e'_j)) / tau
    labels = torch.arange(logits.shape[0], device=logits.device)
    return F.cross_entropy(logits, labels)


def multi_period_mse(y_hat, y, mask=None):
    """The mean squared error of Eq. 12, averaged over the ``N`` stocks and the ``L`` forward
    prediction periods.

    ``y_hat`` and ``y`` are ``(N, L)``. ``mask`` is an optional ``(N, L)`` boolean tensor
    marking the entries with a usable label -- a stock delisted within the horizon has no
    20-day return but still has a 1-day one, and the article's multi-label prediction should
    not be fed a placeholder for the missing horizon.
    """
    if y_hat.shape != y.shape:
        raise ValueError("predictions and labels must have the same shape")
    squared_error = (y_hat - y) ** 2
    if mask is None:
        return squared_error.mean()
    mask = mask.to(squared_error.dtype)
    total = mask.sum()
    if total == 0:
        return squared_error.new_zeros(())
    return (squared_error * mask).sum() / total


def factorgcl_loss(y_hat, y, alpha_past=None, alpha_future=None, projection=None,
                   gamma=DEFAULT_GAMMA, temperature=DEFAULT_TEMPERATURE, mask=None):
    """The overall objective of Eq. 13.

    Returns ``(loss, parts)`` with ``parts`` holding the two terms separately, for logging.
    The contrastive term is dropped when ``gamma`` is zero or no future embedding is given,
    which is both the "-wo CL" ablation and what happens at inference time.
    """
    mse = multi_period_mse(y_hat, y, mask)
    if gamma == 0 or alpha_past is None or alpha_future is None:
        contrastive = mse.new_zeros(())
    else:
        contrastive = info_nce_loss(alpha_past, alpha_future, projection, temperature)
    return mse + gamma * contrastive, {"mse": mse, "contrastive": contrastive}
