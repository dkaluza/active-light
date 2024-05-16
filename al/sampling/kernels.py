from typing import Protocol

import torch
from torch import FloatTensor

# TODO move to main folder


class KernelProto(Protocol):
    support: float

    def __call__(
        self, distances: FloatTensor, bandwidth: FloatTensor
    ) -> FloatTensor: ...


class UniformKernel(KernelProto):
    support = 1

    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        return 0.5 * (distances <= bandwidth)


class GaussianKernel(KernelProto):
    support = torch.inf

    def __call__(self, distances: FloatTensor, bandwidth: FloatTensor) -> FloatTensor:
        gauss_std = distances / bandwidth
        return torch.exp(-0.5 * gauss_std**2) / (torch.pi * 2) ** 0.5


class BandwidthTactic(Protocol):
    def __call__(self, distances: FloatTensor) -> float: ...


class ConstantBandwidth(BandwidthTactic):
    def __init__(self, constant: float) -> None:
        super().__init__()
        self.constant = constant

    def __call__(self, distances: FloatTensor) -> float:
        return self.constant


class SilvermanTactic(BandwidthTactic):
    """
    A Silverman's rule of the thumb estimation.

    For more details see:
    https://en.wikipedia.org/wiki/Kernel_density_estimation#A_rule-of-thumb_bandwidth_estimator
    """

    def __init__(self, multiplicative_constat: float = 1) -> None:
        super().__init__()
        self.multiplicative_constat = multiplicative_constat

    def __call__(self, distances: FloatTensor) -> float:
        normalization_constant = 1.349
        IQR = torch.quantile(distances, q=0.75) - torch.quantile(distances, q=0.25)
        IQR = IQR / normalization_constant
        std = torch.std(distances, dim=0)

        bandwidth = (
            self.multiplicative_constat
            * 0.9
            * torch.minimum(std, IQR)
            * distances.shape[0] ** (-0.2)
        )

        return bandwidth.item()


class ScottTactic(BandwidthTactic):
    """
    A Scott's rule of the thumb estimation.

    For more details see:
    https://en.wikipedia.org/wiki/Histogram#Scott's_normal_reference_rule
    """

    def __init__(self, multiplicative_constat: float = 1) -> None:
        super().__init__()
        self.multiplicative_constat = multiplicative_constat

    def __call__(self, distances: FloatTensor) -> float:
        std = torch.std(distances, dim=0)

        bandwidth = (
            self.multiplicative_constat * 3.49 * std / distances.shape[0] ** (1 / 3)
        )

        return bandwidth.item()


def get_bandwidth_by_dist_quantile(
    distances: FloatTensor, quantile: float = 0.05
) -> float:
    distances = distances.flatten()
    bandwidth = torch.quantile(distances, q=quantile)
    return bandwidth.item()
