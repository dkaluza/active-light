import functools
import math
from typing import Any, Callable

import torch
from torch import FloatTensor
from torch.utils.data import TensorDataset

from al.base import CLASSES_DIM, ActiveState, RegressionModelProto
from al.distances import Distance, L2Distance
from al.kde import KernelDensityEstimator, NaiveKDE
from al.prior import get_prior_from_model_perspective
from al.sampling.base import InformativenessProto
from al.sampling.kernels import BandwidthTactic, KernelProto, QuantileBandwidth


class PrDensity(InformativenessProto):
    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance = L2Distance(),
        kde_class: Callable[
            [KernelProto, Distance, float], KernelDensityEstimator
        ] = functools.partial(NaiveKDE),
        bandwidth_tactic: BandwidthTactic = QuantileBandwidth(quantile=0.05),
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


# TODO: tests as in repr


class PrDensityWeightedEntropy(InformativenessProto):
    def __init__(
        self,
        kernel: KernelProto,
        kde_class: Callable[
            [KernelProto, Distance, float], KernelDensityEstimator
        ] = functools.partial(NaiveKDE),
        bandwidth_tactic: BandwidthTactic = QuantileBandwidth(quantile=0.05),
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.kde_class = kde_class
        self.distance = L2Distance()
        self.bandwidth_tactic = bandwidth_tactic

    def _call(self, logits: FloatTensor, probas: FloatTensor) -> FloatTensor:
        logits_per_class = torch.split(
            logits, split_size_or_sections=1, dim=CLASSES_DIM
        )

        bandwidths = [
            self.get_bandwidth(class_logits) for class_logits in logits_per_class
        ]

        densities = [
            self.get_kde_from_probas(probas=class_logits, bandwidth=bandwidth)
            for class_logits, bandwidth in zip(logits_per_class, bandwidths)
        ]
        priors = get_prior_from_model_perspective(probas=probas, keepdim=True)
        print(priors)
        densities = torch.stack(densities, dim=CLASSES_DIM)
        densities = densities / densities.sum(dim=0, keepdim=True)
        densities_weighted_entropy = torch.where(
            probas == 0, 0, -torch.log2(probas) * probas * densities * probas / priors
        )
        return densities_weighted_entropy.sum(dim=CLASSES_DIM)

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
        return self.bandwidth_tactic(dists_from_chosen_samples)

    def __call__(self, state: ActiveState) -> FloatTensor:
        model = state.get_model()
        pool = state.get_pool()
        logits = model.predict_logits(pool)
        logits = logits / logits.std(dim=0, keepdim=True)
        probas = state.get_probas()
        return self._call(logits=logits, probas=probas)

    @property
    def name(self):
        return "PrDensityWeightedEntropy" + self.kernel.__class__.__name__
