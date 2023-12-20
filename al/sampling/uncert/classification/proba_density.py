from typing import Callable

import torch
from torch import FloatTensor
from torch.utils.data import DataLoader, TensorDataset

from al.distances import jensen_shannon_divergence
from al.sampling.kernels import KernelProto, get_bandwidth_by_dist_quantile
from al.sampling.uncert.classification.base import UncertClassificationBase


class ProbaDensity(UncertClassificationBase):
    def __init__(
        self,
        kernel: KernelProto,
        distance_fun: Callable[
            [FloatTensor, FloatTensor], FloatTensor
        ] = jensen_shannon_divergence,
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.distance_fun = distance_fun
        self.batch_size = batch_size

    def _call(self, probas: FloatTensor) -> FloatTensor:
        n_samples = probas.shape[0]
        # we are processing samples in batch manner to not allocate n_samples^2
        # memory at once as it is infeasible for most datasets
        bandwidth = self.get_bandwidth(probas)

        kernel_values = self.get_kernel_values_from_probas(
            probas=probas, bandwidth=bandwidth
        )
        densities = kernel_values / bandwidth / n_samples
        return densities

    def get_kernel_values_from_probas(
        self, probas: FloatTensor, bandwidth: float
    ) -> FloatTensor:
        kernel_values = []
        probas_dataset = TensorDataset(probas)
        loader = DataLoader(
            dataset=probas_dataset, batch_size=self.batch_size, shuffle=False
        )
        for probas_batch in loader:
            # we have created the dataset in a way that batch is always a one element tuple
            # therefore we can just unpack it
            (probas_batch,) = probas_batch
            # to avoid passing 2 functions we are using simulating cdist with pairwise
            # distance function, probably can be improved in the future
            # we are assuming distance fun is broadcasting opperations
            distances_for_batch = self.distance_fun(
                probas_batch.unsqueeze(1), probas.unsqueeze(0)
            )
            # we are utilizing the fact that for the considered batCH we have all of the
            # distances
            kernel_values_for_batch = self.kernel(
                distances=distances_for_batch,
                bandwidth=bandwidth,
            )

            kernel_values.append(kernel_values_for_batch)
        return torch.concat(kernel_values)

    def get_bandwidth(self, probas) -> float:
        n_samples = probas.shape[0]
        samples_idx = torch.randint(high=n_samples, size=(2, n_samples))
        samples1 = probas[samples_idx[0]]
        samples2 = probas[samples_idx[1]]
        dists_from_chosen_samples = self.distance_fun(samples1, samples2)

        return get_bandwidth_by_dist_quantile(dists_from_chosen_samples)

    @property
    def __name__(self):
        return (
            "ProbaDensity" + self.kernel.__class__.__name__ + self.distance_fun.__name__
        )
