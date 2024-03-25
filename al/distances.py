from typing import Protocol

import torch
from torch import FloatTensor

from al.base import CLASSES_DIM

# TODO: Refactor distances to metrics proto


class Distance(Protocol):
    def pairwise(
        self, input1: torch.FloatTensor, input2: torch.FloatTensor
    ) -> torch.FloatTensor: ...

    def cdist(
        self, input1: torch.FloatTensor, input2: torch.FloatTensor
    ) -> torch.FloatTensor:
        input1 = input1.unsqueeze(1)
        input2 = input2.unsqueeze(0)
        return self.pairwise(input1, input2)

    @property
    def name(self):
        return self.__class__.__name__


class JensenShannonDistance(Distance):
    def pairwise(self, input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
        return jensen_shannon_divergence(input1, input2) ** 0.5


class L2Distance(Distance):
    def pairwise(self, input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
        return torch.nn.functional.pairwise_distance(input1, input2, p=2)

    def cdist(self, input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
        return torch.cdist(input1, input2, p=2)


class L1Distance(Distance):
    def pairwise(self, input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
        return torch.nn.functional.pairwise_distance(input1, input2, p=1)

    def cdist(self, input1: FloatTensor, input2: FloatTensor) -> FloatTensor:
        return torch.cdist(input1, input2, p=1)


def jensen_shannon_divergence(
    *inputs: FloatTensor, weights: FloatTensor | None = None
) -> FloatTensor:
    """Compute Jensen Shanon divergence between consecutive tensors from
    inputs.

    Parameters
    ----------
    *inputs : FloatTensor
        Tensors of shape (batch, ...)

    wegiths: FloatTensor | None
        Weights that will be assigned to each input tensor.
        By default all inputs have the same weight.

    Returns
    -------
    FloatTensor
        Tensor of shape (batch,) with Jensen Shannon divergence
        between consecutive samples from inputs.
    """
    inputs: torch.FloatTensor = torch.stack(torch.broadcast_tensors(*inputs), dim=0)

    n_inputs = inputs.shape[0]
    if weights is None:
        weights = torch.full((n_inputs,), fill_value=1 / n_inputs)

    weights = torch.nn.functional.normalize(weights, p=1, dim=0)

    weighted_mixture_dist = (
        inputs * unsqueeze_to_len(weights, desired_shape_len=len(inputs.shape))
    ).sum(dim=0)

    inputs_entropy = entropy(inputs, dim=CLASSES_DIM)
    weighted_entropies_sum = (
        unsqueeze_to_len(weights, desired_shape_len=len(inputs_entropy.shape))
        * inputs_entropy
    ).sum(dim=0)

    divergence = (
        entropy(weighted_mixture_dist, dim=CLASSES_DIM) - weighted_entropies_sum
    )
    return divergence


def unsqueeze_to_len(tensor: torch.Tensor, desired_shape_len: int, dim=-1):
    original_len = len(tensor.shape)
    for _ in range(desired_shape_len - original_len):
        tensor = tensor.unsqueeze(dim=dim)
    return tensor


def entropy(probas: torch.FloatTensor, dim):
    values = torch.where(probas == 0, 0, probas * torch.log2(probas))
    return -torch.sum(values, dim=dim)
