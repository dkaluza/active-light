import functools
import math
from typing import Callable

import torch
from torch import FloatTensor
from torch.utils.data import TensorDataset

from al.base import ActiveState, RegressionModelProto
from al.distances import Distance, L2Distance
from al.kde import KernelDensityEstimator, NaiveKDE
from al.sampling.base import InformativenessProto
from al.sampling.kernels import BandwidthTactic, KernelProto, SilvermanTactic


class PrDensity(InformativenessProto):
    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance = L2Distance(),
        kde_class: Callable[
            [KernelProto, Distance, float], KernelDensityEstimator
        ] = functools.partial(NaiveKDE),
        bandwidth_tactic: BandwidthTactic = SilvermanTactic(),
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.distance = distance
        self.kde_class = kde_class
        self.bandwidth_tactic = bandwidth_tactic

    def _call(self, values: FloatTensor) -> FloatTensor:
        bandwidth = self.get_bandwidth(values)
        densities = self.get_kde_from_probas(values=values, bandwidth=bandwidth)
        return densities

    def get_kde_from_probas(self, values: FloatTensor, bandwidth: float) -> FloatTensor:
        model = self.kde_class(
            kernel=self.kernel, distance=self.distance, bandwidth=bandwidth
        )
        values = TensorDataset(values)
        model.fit(values)
        return model.predict(values)

    def get_bandwidth(self, values) -> float:
        n_samples = values.shape[0]
        estimation_refinement_factor = math.ceil(math.log2(n_samples))
        samples_idx = torch.randint(
            high=n_samples, size=(2, n_samples * estimation_refinement_factor)
        )
        samples1 = values[samples_idx[0]]
        samples2 = values[samples_idx[1]]

        dists_from_chosen_samples = self.distance.pairwise(samples1, samples2)
        return self.bandwidth_tactic(dists_from_chosen_samples)

    def __call__(self, state: ActiveState) -> FloatTensor:
        pool = state.get_pool()
        model = state.get_model()
        # TODO: refactor logits to be also available through state
        logits = model.predict_logits(pool)

        return self._call(values=logits)

    @property
    def name(self):
        return (
            "ProbaDensity"
            + self.kernel.__class__.__name__
            + self.distance.name
            + self.bandwidth_tactic.__class__.__name__
        )


class PrDensityRegr(PrDensity):
    def __init__(self, kernel: KernelProto, batch_size: int = 32) -> None:
        super().__init__(kernel, distance=L2Distance(), batch_size=batch_size)

    def __call__(self, state: ActiveState) -> FloatTensor:
        model = state.get_model()
        assert isinstance(model, RegressionModelProto)
        preds = model.predict(state.get_pool())
        return self._call(preds)
