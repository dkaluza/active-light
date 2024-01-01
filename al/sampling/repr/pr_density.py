import functools
import math
from typing import Callable

import torch
from torch import FloatTensor
from torch.utils.data import TensorDataset

from al.base import ActiveState, RegressionModelProto
from al.distances import Distance, JensenShannonDistance, L1Distance
from al.kde import KernelDensityEstimator, NaiveKDE
from al.sampling.base import InformativenessProto
from al.sampling.kernels import KernelProto, get_bandwidth_by_dist_quantile


class PrDensity(InformativenessProto):
    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance = JensenShannonDistance(),
        kde_class: Callable[
            [KernelProto, Distance, float], KernelDensityEstimator
        ] = functools.partial(NaiveKDE),
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.distance = distance
        self.kde_class = kde_class

    def _call(self, probas: FloatTensor) -> FloatTensor:
        n_samples = probas.shape[0]
        # we are processing samples in batch manner to not allocate n_samples^2
        # memory at once as it is infeasible for most datasets
        bandwidth = self.get_bandwidth(probas)

        densities = self.get_kde_from_probas(probas=probas, bandwidth=bandwidth)

        return densities

    def get_kde_from_probas(self, probas: FloatTensor, bandwidth: float) -> FloatTensor:
        model = self.kde_class(
            kernel=self.kernel, distance=self.distance, bandwidth=bandwidth
        )
        probas_dataset = TensorDataset(probas)
        model.fit(probas_dataset)
        return model.predict(probas_dataset)

    def get_bandwidth(self, probas) -> float:
        n_samples = probas.shape[0]
        estimation_refinement_factor = math.ceil(math.log2(n_samples))
        samples_idx = torch.randint(
            high=n_samples, size=(2, n_samples * estimation_refinement_factor)
        )
        samples1 = probas[samples_idx[0]]
        samples2 = probas[samples_idx[1]]
        dists_from_chosen_samples = self.distance.pairwise(samples1, samples2)
        return get_bandwidth_by_dist_quantile(dists_from_chosen_samples)

    def __call__(self, state: ActiveState) -> FloatTensor:
        return self._call(state.get_probas())

    @property
    def __name__(self):
        return "ProbaDensity" + self.kernel.__class__.__name__ + self.distance.__name__


class PrDensityRegr(PrDensity):
    def __init__(self, kernel: KernelProto, batch_size: int = 32) -> None:
        super().__init__(kernel, distance=L1Distance(), batch_size=batch_size)

    def __call__(self, state: ActiveState) -> FloatTensor:
        model = state.get_model()
        assert isinstance(model, RegressionModelProto)
        preds = model.predict(state.get_pool())
        return self._call(preds)


# TODO: tests as in repr
