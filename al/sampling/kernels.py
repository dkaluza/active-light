from typing import Protocol

import torch
from torch import FloatTensor


class KernelProto(Protocol):
    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        ...


class UniformKernel(KernelProto):
    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        distances_scaled = distances / bandwidth
        return 0.5 * torch.sum(distances_scaled <= 1, dim=-1, dtype=torch.double)


def get_bandwidth_by_dist_quantile(
    distances: FloatTensor, quantile: float = 0.05
) -> float:
    distances = distances.flatten()
    bandwidth = torch.quantile(distances, q=quantile)
    return bandwidth.item()
