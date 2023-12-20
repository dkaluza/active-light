import pytest
import torch

from al.distances import jensen_shannon_divergence, jensen_shannon_divergence_cdist
from tests.helpers import random_proba


@pytest.mark.parametrize(
    "probas1, probas2",
    [
        (torch.eye(10), random_proba((10, 10))),
        (random_proba((100, 5)), random_proba((100, 5))),
    ],
)
def test_jensen_shannon_returns_appropriate_shape(probas1, probas2):
    assert probas1.shape == probas2.shape

    n_samples = probas1.shape[0]
    div_values = jensen_shannon_divergence(probas1, probas2)
    assert div_values.shape == (n_samples,)


def test_js_divergance_is_in_range_0_1():
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
def test_jensen_shannon_cdist_returns_appropriate_shape(probas1, probas2):
    n_samples1 = probas1.shape[0]
    n_samples2 = probas2.shape[0]

    div_values = jensen_shannon_divergence_cdist(probas1, probas2)
    assert div_values.shape == (n_samples1, n_samples2)
