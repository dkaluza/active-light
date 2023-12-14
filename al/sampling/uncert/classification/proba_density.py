from typing import Callable

import torch
from torch import FloatTensor
from torch.nn.functional import kl_div

from al.sampling.kernels import KernelProto, get_bandwidth_by_dist_quantile
from al.sampling.uncert.classification.base import UncertClassificationBase


def jensen_shanon_divergence(probas: FloatTensor) -> FloatTensor:
    dist1 = probas.unsqueeze(0)
    dist2 = probas.unsqueeze(-1)
    mixture_dist = (dist1 + dist2) / 2
    divergance1 = kl_div(dist1, mixture_dist, reduction=None)
    divergance2 = kl_div(
        dist2, mixture_dist, reduction=None
    )  # note: might be optimizes by transpose?
    return divergance1 / 2 + divergance2 / 2


def l2_distance(probas: FloatTensor) -> FloatTensor:
    return torch.cdist(probas, probas, p=2)


class ProbaDensity(UncertClassificationBase):
    def __init__(
        self,
        uncert: UncertClassificationBase,
        kernel: KernelProto,
        distance_fun: Callable[[FloatTensor], FloatTensor] = jensen_shanon_divergence,
    ) -> None:
        super().__init__()
        self.uncert = uncert
        self.kernel = kernel
        self.distance_fun = distance_fun

    def _call(self, probas: FloatTensor) -> FloatTensor:
        n_samples = probas.shape[0]
        uncerts = self.uncert._call(probas)
        bandwidth = self.get_bandwidth(probas)
        kernel_values = self.kernel(
            distances=self.get_distances_from_probas(probas),
            bandwidth=bandwidth,
        )
        densities = kernel_values / bandwidth / n_samples
        return densities * uncerts

    def get_distances_from_probas(self, probas: FloatTensor) -> FloatTensor:
        return self.distance_fun(probas)

    def get_bandwidth(self, probas):
        return get_bandwidth_by_dist_quantile(self.get_distances_from_probas(probas))

    @property
    def __name__(self):
        return "ProbaDensity" + self.uncert + self.kernel + self.distance_fun.__name__
