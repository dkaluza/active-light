import logging
import math
from typing import Callable

import faiss
import faiss.contrib.torch_utils  # import needed to add faiss torch interoperability
import torch
from torch import FloatTensor

from al.base import get_default_torch_device
from al.distances import jensen_shannon_divergence, l2
from al.sampling.kernels import KernelProto, get_bandwidth_by_dist_quantile
from al.sampling.uncert.classification.base import UncertClassificationBase

logger = logging.getLogger(__name__)


class ProbaDensity(UncertClassificationBase):
    def __init__(
        self,
        kernel: KernelProto,
        distance_fun: Callable[
            [FloatTensor, FloatTensor], FloatTensor
        ] = jensen_shannon_divergence,
        batch_size: int = 32,
        index_device: None | torch.device = None,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.distance_fun = distance_fun  # note: consier switching to faiss distances?
        metrics_for_distances = {
            jensen_shannon_divergence: faiss.METRIC_JensenShannon,
            l2: faiss.METRIC_L2,
        }
        index_metric = metrics_for_distances.get(distance_fun)
        if index_metric is None:
            logger.warn("No metric registered for distance_fun, using default L2")
            index_metric = metrics_for_distances[l2]

        self.index_metric = index_metric
        self.batch_size = batch_size
        self.k = 1000
        self.index_device = index_device

    def _call(self, probas: FloatTensor) -> FloatTensor:
        n_samples = probas.shape[0]

        if self.index_device is None:
            self.index_device = get_default_torch_device()
        # we are processing samples in batch manner to not allocate n_samples^2
        # memory at once as it is infeasible for most datasets
        bandwidth = self.get_bandwidth(probas)
        index = self.build_index(probas)
        kernel_values = self.get_kernel_values_from_probas(
            probas=probas, bandwidth=bandwidth, index=index
        )
        densities = kernel_values / bandwidth / n_samples
        return densities

    def build_index(self, probas) -> faiss.Index:
        n_classes = probas.shape[1]
        n_samples = probas.shape[0]
        nlist = math.sqrt(n_samples) if n_samples > 5_000 else math.log2(n_samples)
        nlist = math.ceil(nlist)

        index = (
            self._get_gpu_index(n_classes=n_classes, nlist=nlist)
            if self.index_device.type == "cuda"
            else self._get_cpu_index(n_classes=n_classes, nlist=nlist)
        )

        index.nprobe = 3
        probas = probas.to(self.index_device)

        index.train(probas)
        index.add(probas)

        return index

    def _get_cpu_index(self, n_classes, nlist):
        quantizer = faiss.IndexFlat(n_classes, self.index_metric)
        # IndexIVFFlat unfortunettely does not support most metrics, so we can't pass self.index_metric
        index = faiss.IndexIVFFlat(quantizer, n_classes, nlist)
        return index

    def _get_gpu_index(self, n_classes, nlist):
        res = faiss.StandardGpuResources()
        index = self._get_cpu_index(n_classes=n_classes, nlist=nlist)
        index = faiss.index_cpu_to_gpu(res, device=self.index_device.index, index=index)

        return index

    def get_kernel_values_from_probas(
        self, probas: FloatTensor, bandwidth: float, index: faiss.Index
    ) -> FloatTensor:
        n_samples = probas.shape[0]
        k = self.k + 1 if self.k < n_samples else n_samples // 10
        distances_for_batch = self.get_distances_from_index(probas, index, k=k)
        kernel_values = self.kernel(
            distances=distances_for_batch,
            bandwidth=bandwidth,
        )

        return kernel_values

    def get_distances_from_index(
        self, probas: torch.FloatTensor, index: faiss.Index, k: int
    ):
        distances, _indices = index.search(probas.to(self.index_device), self.k + 1)

        return distances[:, 1:]

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
