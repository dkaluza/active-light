import enum

import torch
from torch import FloatTensor
from torch.nn.functional import kl_div


def jensen_shannon_divergence_cdist(
    input1: FloatTensor, input2: FloatTensor
) -> FloatTensor:
    """Compute Jensen Shannon divergence between each pair of tensors from
    `input1` and `input2`.

    Parameters
    ----------
    input1 : FloatTensor
        Tensor of shape (b1, ...)
    input2 : FloatTensor
        Tensor of shape (b2, ...)
        Other dimensions should be consistent with `input1`

    Returns
    -------
    FloatTensor
        Tensor of shape (b1, b2) with Jensen Shannon divergence
        between each pair of tensors from inputs.
    """
    input1 = input1.unsqueeze(1)
    input2 = input2.unsqueeze(0)
    return jensen_shannon_divergence(input1, input2)


def jensen_shannon_divergence(input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
    """Compute Jensen Shanon divergence between consecutive tensors from
    `input1` and `input2`.

    Parameters
    ----------
    input1 : FloatTensor
        Tensor of shape (batch, ...)
    input2 : FloatTensor
        Tensor of shape (batch, ...)
        Shape of the tensor should be consistent with `input1`

    Returns
    -------
    FloatTensor
        Tensor of shape (batch,) with Jensen Shannon divergence
        between consecutive samples from inputs.
    """
    mixture_dist = (input1 + input2) / 2
    mixture_dist = mixture_dist.log()  # kl_div requires log-probas
    # kl_div has reveresed args order from math notation, therefore we are using
    # misture as first arg
    divergance1 = kl_div(mixture_dist, input1, reduction="none").sum(dim=-1)
    divergance2 = kl_div(mixture_dist, input2, reduction="none").sum(dim=-1)

    return divergance1 / 2 + divergance2 / 2


def l2_cdist(input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
    return torch.cdist(input1, input2, p=2)


def l2(input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
    return torch.nn.functional.pairwise_distance(input1, input2, p=2)
