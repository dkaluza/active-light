import pytest
import torch
from sklearn.neighbors import KernelDensity

from al.distances import JensenShannonDistance, L2Distance
from al.distributions import uniform_mesh
from al.sampling.kernels import GaussianKernel, UniformKernel
from al.sampling.repr.pr_density import PrDensity
from tests.helpers import probas_state

N_SAMPLES = 1_000
N_CLASSES = 5


@pytest.mark.parametrize(
    "dist, proba_density",
    [
        (
            torch.distributions.HalfNormal(scale=1),
            PrDensity(kernel=UniformKernel(), distance=JensenShannonDistance()),
        ),
        (
            torch.distributions.HalfNormal(scale=1),
            PrDensity(kernel=GaussianKernel(), distance=JensenShannonDistance()),
        ),
        (
            torch.distributions.HalfNormal(
                scale=1
            ),  # we are repeating dists as some are not suitable for JensenShannon
            PrDensity(kernel=UniformKernel(), distance=L2Distance()),
        ),
        (
            torch.distributions.HalfNormal(scale=1),
            PrDensity(kernel=GaussianKernel(), distance=L2Distance()),
        ),
    ],
)
def test_kernel_estimation_is_close_to_pdf_for_proba_dist(dist, proba_density):
    sampled_values: torch.Tensor = dist.sample(sample_shape=(N_SAMPLES, N_CLASSES))
    probas = torch.nn.functional.normalize(sampled_values, p=1)

    state = probas_state(probas=probas)

    values = proba_density(state)
    gt_densities = torch.exp(dist.log_prob(sampled_values).sum(dim=-1))
    assert values.shape == (N_SAMPLES,)

    assert torch.nn.functional.l1_loss(values.float(), gt_densities.float()) < 0.15


@pytest.mark.parametrize(
    "dist, distance",
    [
        (torch.distributions.Normal(loc=0, scale=1), L2Distance()),
    ],
)
def test_kernel_estimation_is_not_worse_than_sklearn_kde(dist, distance):
    sampled_values: torch.Tensor = dist.sample(sample_shape=(N_SAMPLES, N_CLASSES))

    sklearn_kde = KernelDensity(kernel="tophat")
    sklearn_kde.fit(sampled_values)
    sklearn_densities = torch.exp(
        torch.from_numpy(sklearn_kde.score_samples(sampled_values))
    )

    state = probas_state(probas=sampled_values)
    proba_density = PrDensity(kernel=UniformKernel(), distance=distance)
    values = proba_density(state)

    gt_densities = torch.exp(dist.log_prob(sampled_values).sum(dim=-1))
    values_loss = torch.nn.functional.l1_loss(values.float(), gt_densities.float())
    sklearn_loss = torch.nn.functional.l1_loss(
        sklearn_densities.float(), gt_densities.float()
    )
    # TODO: sklearn is probably better because it is handling aggregated probas in numberical space
    assert values_loss < sklearn_loss


PROBA_UNIFORM_MESH_STEP = 0.01


@pytest.mark.parametrize(
    "distance, probas",
    [
        (
            JensenShannonDistance(),
            uniform_mesh(n_classes=2, step=PROBA_UNIFORM_MESH_STEP),
        ),
        (L2Distance(), uniform_mesh(n_classes=2, step=PROBA_UNIFORM_MESH_STEP)),
        (
            L2Distance(),
            torch.distributions.Normal(loc=0, scale=1).sample(
                sample_shape=(N_SAMPLES, N_CLASSES)
            ),
        ),
    ],
)
def test_bandwidth_is_close_to_real_quantile(distance, probas):
    gt_distances = distance.cdist(probas, probas)
    proba_density = PrDensity(kernel=UniformKernel(), distance=distance)
    bandwidth = proba_density.get_bandwidth(probas.double())

    gt_bandwidth = torch.quantile(gt_distances, q=0.05)
    assert (gt_bandwidth - bandwidth).abs() < gt_bandwidth / 2
