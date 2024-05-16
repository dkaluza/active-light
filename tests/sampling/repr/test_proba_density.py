import pytest
import torch
from sklearn.neighbors import KernelDensity

from al.distances import JensenShannonDistance, L2Distance
from al.sampling.kernels import GaussianKernel, UniformKernel
from al.sampling.repr.pr_density import PrDensity
from tests.helpers import probas_logit_state

N_SAMPLES = 1_000
N_CLASSES = 1


@pytest.mark.parametrize(
    "dist, proba_density",
    [
        (
            torch.distributions.HalfNormal(scale=1),
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

    state = probas_logit_state(probas=sampled_values, logits=sampled_values)

    values = proba_density(state)
    gt_densities = torch.exp(dist.log_prob(sampled_values).sum(dim=-1))
    assert values.shape == (N_SAMPLES,)
    print(values.float(), gt_densities.float())
    assert torch.nn.functional.l1_loss(values.float(), gt_densities.float()) < 0.1


@pytest.mark.parametrize(
    "dist, distance",
    [
        (torch.distributions.Normal(loc=0, scale=1), L2Distance()),
    ],
)
def test_kernel_estimation_is_not_far_worse_than_sklearn_kde(dist, distance):
    sampled_values: torch.Tensor = dist.sample(sample_shape=(N_SAMPLES, N_CLASSES))

    sklearn_kde = KernelDensity(kernel="tophat", bandwidth="silverman")
    sklearn_kde.fit(sampled_values)
    sklearn_densities = torch.exp(
        torch.from_numpy(sklearn_kde.score_samples(sampled_values))
    )

    state = probas_logit_state(probas=sampled_values, logits=sampled_values)
    proba_density = PrDensity(kernel=UniformKernel(), distance=distance)
    values = proba_density(state)

    gt_densities = torch.exp(dist.log_prob(sampled_values).sum(dim=-1))
    values_loss = torch.nn.functional.l1_loss(values.float(), gt_densities.float())
    sklearn_loss = torch.nn.functional.l1_loss(
        sklearn_densities.float(), gt_densities.float()
    )
    # TODO: sklearn might be more computationally stable
    # as it is making computation in logarithmic space
    # it should be investigated further if loss KDE is good enough
    assert values_loss < sklearn_loss * 2
