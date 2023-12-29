import math

import torch
from torch import FloatTensor
from torch.utils.data import DataLoader, TensorDataset
from torchaudio.functional import convolve

from al.base import RegressionModelProto
from al.binning import approximate_point_densities_from_binned_density, linear_binning
from al.distances import Distance, JensenShannonDistance, L1Distance
from al.sampling.base import ActiveState, InformativenessProto
from al.sampling.kernels import KernelProto, get_bandwidth_by_dist_quantile


class PrDensity(InformativenessProto):
    def __init__(
        self,
        kernel: KernelProto,
        distance: Distance = JensenShannonDistance(),
        batch_size: int = 32,
    ) -> None:
        super().__init__()
        self.kernel = kernel
        self.distance = distance
        self.batch_size = batch_size

    def _call(self, probas: FloatTensor) -> FloatTensor:
        n_samples = probas.shape[0]
        # we are processing samples in batch manner to not allocate n_samples^2
        # memory at once as it is infeasible for most datasets
        bandwidth = self.get_bandwidth(probas)

        kernel_values = self.get_kernel_values_from_probas(
            probas=probas, bandwidth=bandwidth
        ).sum(dim=-1)
        print(kernel_values.max(), n_samples, bandwidth)
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
            distances_for_batch = self.distance.cdist(probas_batch, probas)
            # we are utilizing the fact that for the considered batch we have all of the
            # distances
            kernel_values_for_batch = self.kernel(
                distances=distances_for_batch,
                bandwidth=bandwidth,
            )

            kernel_values.append(kernel_values_for_batch)
        return torch.concat(kernel_values)

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


class PrDensityApprox(PrDensity):
    def get_kernel_values_from_probas(
        self, probas: FloatTensor, bandwidth: float
    ) -> FloatTensor:
        binning_results = linear_binning(points=probas, distance=self.distance)
        grid_weights = binning_results.grid_weights
        step_sizes = binning_results.step_sizes

        kernel_support_distance = self.kernel.support * bandwidth

        # note: this approximation might be inaccurate for distances like Jensen-Shannon
        n_grid_sizes_in_kernel_support_distance = torch.floor(
            kernel_support_distance / step_sizes.squeeze()
        )
        n_grid_sizes_to_check = torch.minimum(
            n_grid_sizes_in_kernel_support_distance, grid_weights.shape
        )

        grid_coords_in_kernel_sup_distance = torch.cartesian_prod(
            *[
                torch.linspace(
                    -step_size * num_of_steps,
                    step_size * num_of_steps,
                    steps=2 * num_of_steps + 1,
                )  # 0 should be included as a result of steps equal to 2*num_of_steps+1
                for num_of_steps, step_size in zip(n_grid_sizes_to_check, step_sizes)
            ]
        )

        # note: this might cause large errors for Jensen-Shannon distance
        distances_to_grid_coords = self.distance.pairwise(
            torch.zeros_like(
                grid_coords_in_kernel_sup_distance,
            ),
            grid_coords_in_kernel_sup_distance,
        )
        kernel_values_for_distances = self.kernel(
            distances=distances_to_grid_coords, bandwidth=bandwidth
        )

        grid_densities = convolve(
            grid_weights, kernel_values_for_distances, padding="same"
        )

        points_densities = approximate_point_densities_from_binned_density(
            grid_densities=grid_densities,
            points=probas,
            distance=self.distance,
            min_coords=binning_results.min_coords,
            step_sizes=step_sizes,
        )

        return points_densities


class PrDensityRegr(PrDensity):
    def __init__(self, kernel: KernelProto, batch_size: int = 32) -> None:
        super().__init__(kernel, distance=L1Distance(), batch_size=batch_size)

    def __call__(self, state: ActiveState) -> FloatTensor:
        model = state.get_model()
        assert isinstance(model, RegressionModelProto)
        preds = model.predict(state.get_pool())
        return self._call(preds)


# TODO: tests as in repr
