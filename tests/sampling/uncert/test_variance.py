import pytest
import torch
from scipy.stats import norm

from al.sampling.kernels import GaussianKernel
from al.sampling.uncert import eveal, variance
from al.sampling.uncert.regression.variance_based import UniformKernel


@pytest.mark.parametrize(
    "distribution, dist_kwargs",
    [
        (norm, dict(loc=1, scale=2)),
        (norm, {}),
    ],
)
@pytest.mark.parametrize("kernel", [UniformKernel(), GaussianKernel()])
@pytest.mark.parametrize("bandwidth", [0.1, 0.3, 0.5])
def test_kernel_estimation_converges_to_expected_density(
    distribution, dist_kwargs, kernel, bandwidth
):
    samples = distribution.rvs(**dist_kwargs, size=1_000)
    samples = torch.from_numpy(samples)
    distances = (samples.unsqueeze(0) - samples.unsqueeze(-1)).abs()
    kernel_values = kernel(distances, bandwidth=torch.tensor(bandwidth))
    estimated_densities = kernel_values.sum(dim=-1) / len(samples) / bandwidth
    expected_prob_density = distribution.pdf(x=samples, **dist_kwargs)
    expected_prob_density = torch.from_numpy(expected_prob_density)
    assert estimated_densities.shape == expected_prob_density.shape
    assert torch.nn.functional.l1_loss(estimated_densities, expected_prob_density) < 0.1
