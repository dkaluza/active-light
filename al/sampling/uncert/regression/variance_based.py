from typing import Any, Protocol

import torch
from torch import FloatTensor, Tensor

from al.base import RegressionModelProto
from al.sampling.kernels import KernelProto, UniformKernel

from .base import UncertRegressionBase


class Variance(UncertRegressionBase):
    def _call(
        self, distribution_params: Tensor, model: RegressionModelProto
    ) -> FloatTensor:
        return model.get_variance(distribution_params=distribution_params)


class EVEAL(UncertRegressionBase):
    """Uncertainty estimation using  model variance weighted by
    probability density estimation in posterior space.

    Probability density in posterior space can be used as effective
    upper bound of probability density in the domain space.

    Based on:
    D. Kałuża, A. Janusz and D. Ślęzak, "EVEAL - Expected Variance Estimation for Active Learning,"
    2022 IEEE International Conference on Big Data (Big Data), Osaka, Japan, 2022, pp. 6222-6231,
    doi: 10.1109/BigData55660.2022.10020950.


    """

    def __init__(
        self,
        epsilon: float = 1.0,
        use_std_bandwith: bool = False,
        kernel: KernelProto = None,
    ) -> None:
        super().__init__()
        self.epsilon = epsilon
        if kernel is None:
            kernel = UniformKernel()

        self.kernel = kernel
        self.use_std_bandwith = use_std_bandwith

    def _call(
        self, distribution_params: Tensor, model: RegressionModelProto
    ) -> FloatTensor:
        density = self.estimate_density(
            distribution_params=distribution_params, model=model
        )
        return density * model.get_variance(distribution_params=distribution_params)

    def estimate_density(
        self, distribution_params: Tensor, model: RegressionModelProto
    ) -> FloatTensor:
        n_samples = distribution_params.shape[0]
        expected_values = model.get_expected_value(
            distribution_params=distribution_params
        )

        bandwidth = self.epsilon
        if self.use_std_bandwith:
            bandwidth *= torch.sqrt(
                model.get_variance(distribution_params=distribution_params)
            )

        kernel_values = self.kernel(
            distances=self.get_distance_from_expected_values(expected_values),
            bandwidth=bandwidth,
        )
        return kernel_values / bandwidth / n_samples

    def get_distance_from_expected_values(
        self, expected_values: FloatTensor
    ) -> FloatTensor:
        # TODO: use batch based implementation as in classification?
        # mabe we can use common implementation parts?
        distances = expected_values.unsqueeze(1) - expected_values.unsqueeze(0)
        return distances.abs()


variance = Variance()

eveal = EVEAL()
