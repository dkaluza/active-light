from typing import Protocol

import torch
from torch import FloatTensor

# TODO move to main folder


class KernelProto(Protocol):
    support: float

    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        ...


class UniformKernel(KernelProto):
    support = 1

    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        return 0.5 * (distances <= bandwidth)


class GaussianKernel(KernelProto):
    support = torch.inf

    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        gauss_std = distances / bandwidth
        return torch.exp(-0.5 * gauss_std**2) / (torch.pi * 2) ** 0.5


def get_bandwidth_by_dist_quantile(
    distances: FloatTensor, quantile: float = 0.05
) -> float:
    distances = distances.flatten()
    bandwidth = torch.quantile(distances, q=quantile)
    return bandwidth.item()
