import pytest
import torch

from al.distances import l2
from al.sampling.kernels import UniformKernel
from al.sampling.uncert.classification.proba_density import ProbaDensity
from tests.helpers import probas_state

N_SAMPLES = 1_000
N_CLASSES = 10


@pytest.mark.parametrize(
    "dist",
    [
        torch.distributions.normal.Normal(0, 1),
    ],
)
def test_kernel_estimation_in_normal_mean(dist):
    probas = dist.sample(sample_shape=(N_SAMPLES, N_CLASSES))
    state = probas_state(torch.concat([torch.zeros(1, N_CLASSES), probas]))
    proba_density = ProbaDensity(kernel=UniformKernel(), distance_fun=l2)

    values = proba_density(state)
    assert values.shape == (N_SAMPLES + 1,)
    print(values[0], torch.exp(dist.log_prob(dist.mean)))
    # TODO fix
    assert torch.allclose(values[0], torch.exp(dist.log_prob(dist.mean)).double())


def test_bandwidth_is_close_to_real_quantile():
    # TODO
    assert False
