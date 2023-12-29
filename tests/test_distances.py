import pytest
import torch
from scipy.spatial.distance import jensenshannon

from al.distances import JensenShannonDistance, L2Distance, jensen_shannon_divergence
from tests.helpers import random_proba

DISTANCES = [JensenShannonDistance(), L2Distance()]


@pytest.mark.parametrize(
    "probas1, probas2",
    [
        (torch.eye(10), random_proba((10, 10))),
        (random_proba((100, 5)), random_proba((100, 5))),
    ],
)
@pytest.mark.parametrize("distance", DISTANCES)
def test_pairwise_distance_returns_appropriate_shape(probas1, probas2, distance):
    assert probas1.shape == probas2.shape

    n_samples = probas1.shape[0]
    div_values = distance.pairwise(probas1, probas2)
    assert div_values.shape == (n_samples,)


def test_js_divergance_is_in_range_0_1_for_2_inputs():
    probas1, probas2 = random_proba((1000, 5)), random_proba((1000, 5))

    div_values = jensen_shannon_divergence(probas1, probas2)
    assert torch.all(div_values >= 0)
    assert torch.all(div_values <= 1)


@pytest.mark.parametrize(
    "probas1, probas2",
    [
        (torch.eye(10), random_proba((45, 10))),
        (random_proba((100, 5)), random_proba((12, 5))),
    ],
)
@pytest.mark.parametrize("distance", DISTANCES)
def test_cdist_returns_appropriate_shape(probas1, probas2, distance):
    n_samples1 = probas1.shape[0]
    n_samples2 = probas2.shape[0]

    div_values = distance.cdist(probas1, probas2)
    assert div_values.shape == (n_samples1, n_samples2)


@pytest.mark.parametrize(
    "probas1, probas2",
    [
        (random_proba((100, 5)), random_proba((100, 5))),
    ],
)
def test_jensen_shannon_dist_for_2_vectors_matches_scipy_jensenshannon_squared(
    probas1, probas2
):
    # distance in scipy is squared root of divergance
    js_scipy = jensenshannon(probas1, probas2, axis=1, base=2) ** 2

    assert torch.allclose(
        jensen_shannon_divergence(probas1, probas2),
        torch.from_numpy(js_scipy),
        atol=0.0001,
    )
