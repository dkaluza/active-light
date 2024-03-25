import pytest
import torch
from torch.utils.data import Dataset, TensorDataset

from al.distances import L2Distance
from al.kde import ClusteringKDE, NaiveKDE
from al.sampling.kernels import GaussianKernel, KernelProto, UniformKernel


@pytest.mark.parametrize(
    "data",
    [
        TensorDataset(torch.rand(500, 10)),
        TensorDataset(torch.rand(1000, 3)),
        TensorDataset(torch.rand(1000, 1)),
    ],
)
@pytest.mark.parametrize("bandwidth", [0.3, 0.5, 1.0])
@pytest.mark.parametrize("kernel", [GaussianKernel(), UniformKernel()])
def test_clustering_kde_returns_close_values_to_naive(
    data: Dataset, bandwidth: float, kernel: KernelProto
):
    naive = NaiveKDE(kernel=kernel, distance=L2Distance(), bandwidth=bandwidth)
    clustering = ClusteringKDE(
        kernel=kernel,
        distance=L2Distance(),
        bandwidth=bandwidth,
    )

    naive.fit(data)
    clustering.fit(data)

    assert torch.allclose(naive.predict(data), clustering.predict(data), atol=5e-3)
