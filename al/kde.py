import logging
import math
from typing import Protocol

import faiss
import faiss.contrib.torch_utils  # import needed to add faiss torch interoperability
import sklearn.neighbors
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
from torchaudio.functional import convolve

from al.base import get_default_torch_device
from al.binning import approximate_point_densities_from_binned_density, linear_binning
from al.distances import Distance, JensenShannonDistance, L2Distance
from al.loops.base import ALDatasetWithoutTargets
from al.sampling.kernels import KernelProto

# TODO: tests?

logger = logging.getLogger(__name__)


class KernelDensityEstimator(Protocol):
    kernel: KernelProto
    bandwidth: float
    distance: Distance

    def __init__(
        self, kernel: KernelProto, distance: Distance, bandwidth: float
    ) -> None:
        super().__init__()

        self.kernel = kernel
        self.bandwidth = bandwidth
        self.distance = distance

    def fit(self, train: Dataset): ...

    def predict(self, data: Dataset) -> torch.FloatTensor: ...


class NaiveKDE(KernelDensityEstimator):
    saved_data: torch.FloatTensor | None
    batch_size: int

    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance,
        bandwidth: float,
        batch_size: int = 32,
    ) -> None:
        super().__init__(kernel, distance, bandwidth)
        self.batch_size = batch_size

    def fit(self, train: Dataset):
        self.saved_data = ALDatasetWithoutTargets(train).features

    def predict(self, data: Dataset) -> torch.FloatTensor:
        kernel_values = []
        loader = DataLoader(dataset=data, batch_size=self.batch_size, shuffle=False)
        for data_batch in loader:
            # we have created the dataset in a way that batch is always a one element tuple
            # therefore we can just unpack it
            (data_batch,) = data_batch
            distances_for_batch = self.distance.cdist(data_batch, self.saved_data)
            # we are utilizing the fact that for the considered batch we have all of the
            # distances
            kernel_values_for_batch = self.kernel(
                distances=distances_for_batch,
                bandwidth=self.bandwidth,
            )

            kernel_values.append(kernel_values_for_batch.sum(-1))
        logger.debug(
            "NaiveKDE. Bandwidth: %f, n_samples: %d.\nKernel values %s",
            self.bandwidth,
            self.saved_data.shape[0],
            kernel_values,
        )
        return torch.concat(kernel_values) / self.bandwidth / self.saved_data.shape[0]


class LinearBinningKDE(KernelDensityEstimator):
    ditance: Distance

    grid_densities: None | torch.FloatTensor
    min_coords: None | torch.FloatTensor
    step_sizes: None | torch.FloatTensor

    def __init__(
        self, kernel: KernelProto, bandwidth: float, distance: Distance
    ) -> None:
        super().__init__(kernel=kernel, bandwidth=bandwidth)
        self.ditance = distance
        self.grid_densities = None
        self.min_coords = None
        self.step_sizes = None

    def fit(self, train: Dataset):
        raise Exception("Fikx me, this implementation is bugged right now")
        probas = ALDatasetWithoutTargets(train).features
        binning_results = linear_binning(points=probas, distance=self.distance)
        grid_weights = binning_results.grid_weights
        step_sizes = binning_results.step_sizes

        kernel_support_distance = self.kernel.support * self.bandwidth

        # note: this approximation might be inaccurate for distances like Jensen-Shannon
        n_grid_sizes_in_kernel_support_distance = torch.floor(
            kernel_support_distance / step_sizes.squeeze()
        )
        n_grid_sizes_in_kernel_support_distance.nan_to_num_(torch.inf)

        n_grid_sizes_to_check = torch.minimum(
            n_grid_sizes_in_kernel_support_distance,
            torch.tensor(grid_weights.shape, dtype=torch.int),
        ).int()

        grid_coords_in_kernel_sup_distance = torch.cartesian_prod(
            *[
                torch.linspace(
                    -step_size * num_of_steps,
                    step_size * num_of_steps,
                    steps=2 * num_of_steps.item() + 1,
                )  # 0 should be included as a result of steps equal to 2*num_of_steps+1
                for num_of_steps, step_size in zip(
                    n_grid_sizes_to_check, step_sizes.squeeze()
                )
            ]
        )

        # note: this might cause large errors for Jensen-Shannon distance
        distances_to_grid_coords = self.distance.pairwise(
            torch.zeros_like(
                grid_coords_in_kernel_sup_distance,
            ),
            grid_coords_in_kernel_sup_distance,
        )
        print(distances_to_grid_coords.shape)
        kernel_values_for_distances = self.kernel(
            distances=distances_to_grid_coords, bandwidth=self.bandwidth
        )

        print(kernel_values_for_distances.shape)
        self.grid_densities = convolve(
            grid_weights, kernel_values_for_distances, mode="same"
        )
        self.step_sizes = step_sizes
        self.min_coords = binning_results.min_coords

    def predict(self, data: Dataset) -> torch.FloatTensor:
        if self.grid_densities is None:
            raise Exception(
                "Fit should be run before predict"
            )  # Note introduce own exception class?

        probas = ALDatasetWithoutTargets(data).features
        points_densities = approximate_point_densities_from_binned_density(
            grid_densities=self.grid_densities,
            points=probas,
            distance=self.distance,
            min_coords=self.min_coords,
            step_sizes=self.step_sizes,
        )

        return points_densities


class TreeKDE(KernelDensityEstimator):
    "Warning this implementation looks slower than NaiveKDE for data with 35k samples"
    sklearn_model: sklearn.neighbors.KernelDensity

    def __init__(
        self,
        kernel: KernelProto,
        bandwidth: float,
        distance: Distance,
        algorithm: str,
        atol: float = 1e-4,
    ) -> None:
        super().__init__(kernel, distance, bandwidth)
        if isinstance(self.distance, L2Distance):
            metric = "euclidean"
        else:
            raise NotImplementedError(self.distance)

        raise NotImplementedError("Kernel mapping to sklearn not implemented yet")

        self.sklearn_model = sklearn.neighbors.KernelDensity(
            bandwidth=bandwidth,
            algorithm=algorithm,
            kernel=kernel,
            metric=metric,
            atol=atol,
        )

    def fit(self, train: Dataset):
        data: torch.FloatTensor = ALDatasetWithoutTargets(train).features
        data = data.numpy()
        self.sklearn_model.fit(data)

    def predict(self, data: Dataset) -> torch.FloatTensor:
        data: torch.FloatTensor = ALDatasetWithoutTargets(data).features
        data = data.numpy()
        log_probas = self.sklearn_model.score_samples(data)
        probas = torch.from_numpy(log_probas).exp()
        return probas


class ClusteringKDE(KernelDensityEstimator):
    index: faiss.Index | None
    n_index_samples: int | None
    batch_size: int

    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance,
        bandwidth: float,
        index_device: None | torch.device = None,
        batch_size=1024,
    ) -> None:
        super().__init__(kernel, distance, bandwidth)
        metrics_for_distances = {
            JensenShannonDistance: faiss.METRIC_JensenShannon,
            L2Distance: faiss.METRIC_L2,
        }
        index_metric = metrics_for_distances.get(distance.__class__)
        if index_metric is None:
            logger.warn("No metric registered for distance_fun, using default L2")
            index_metric = metrics_for_distances[L2Distance]

        self.index_metric = index_metric
        self.index_device = index_device
        if self.index_device is None:
            self.index_device = get_default_torch_device()

        self.index = None
        self.n_index_samples = None
        self.batch_size = batch_size

    def fit(self, train: Dataset):
        data: torch.FloatTensor = ALDatasetWithoutTargets(train).features
        n_features = data.shape[1]
        n_samples = data.shape[0]
        nlist = math.sqrt(n_samples) if n_samples > 5_000 else math.log2(n_samples)
        nlist = math.ceil(nlist)

        index = (
            self._get_gpu_index(n_features=n_features, nlist=nlist)
            if self.index_device.type == "cuda"
            else self._get_cpu_index(n_features=n_features, nlist=nlist)
        )

        index.nprobe = 10
        data = data.to(self.index_device)

        index.train(data)
        index.add(data)

        self.index = index
        self.n_index_samples = n_samples

    def _get_cpu_index(self, n_features, nlist):
        quantizer = faiss.IndexFlat(n_features, self.index_metric)
        # IndexIVFFlat unfortunettely does not support most metrics, so we can't pass self.index_metric
        index = faiss.IndexIVFFlat(quantizer, n_features, nlist)
        return index

    def _get_gpu_index(self, n_features, nlist):
        res = faiss.StandardGpuResources()
        index = self._get_cpu_index(n_features=n_features, nlist=nlist)
        index = faiss.index_cpu_to_gpu(res, device=self.index_device.index, index=index)

        return index

    def predict(self, data: Dataset) -> torch.FloatTensor:
        if self.index is None:
            raise Exception(
                "Fit should be run before predict"
            )  # Note introduce own exception class?

        data: torch.FloatTensor = ALDatasetWithoutTargets(data).features
        kernel_values = []
        for data_batch in torch.split(data, self.batch_size):
            # faiss is using sq dist so we have to adjust threshold
            bandwidth_in_squared_dist = self.bandwidth**2
            # we are using 3 bw as for normal distribution it covers
            # most of the values
            lims, distances, _indices = self.index.range_search(
                data_batch.to(self.index_device),
                bandwidth_in_squared_dist
                * 9,  # we wish to retrieve 3 * bw but since it is squared we have to retrive bw**2 * 9
            )
            # TODO: test if appropriate distances are taken

            sequence_sizes = lims[1:] - lims[:-1]
            splitted_distances = torch.split(
                distances, split_size_or_sections=sequence_sizes.tolist()
            )

            data_batch_distances_padded = pad_sequence(
                splitted_distances, batch_first=True, padding_value=torch.inf
            )
            # faiss returns square of the L2 distance therefore sqrt is needed
            values_for_batch = self.kernel(
                distances=data_batch_distances_padded**0.5,
                bandwidth=self.bandwidth,
            ).sum(-1)
            kernel_values.append(values_for_batch)

        return (
            torch.concat(kernel_values).to(get_default_torch_device())
            / self.bandwidth
            / self.n_index_samples
        )
