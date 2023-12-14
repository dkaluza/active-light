import pytest
import torch

from al.sampling.uncert import entropy, least_confidence, margin
from al.sampling.uncert.classification.metrics import (
    _difference_quotient,
    _get_nearest_vertex_position,
    prior_descent_ratio,
    uncert_maximum_descent_ratio,
)
from tests.helpers import random_proba


@pytest.mark.parametrize(
    "distribution, expected_vertex",
    [
        (torch.eye(3, dtype=torch.float), torch.eye(3, dtype=torch.float)),
        (
            torch.tensor([[0, 0.3, 0.5, 0, 0.2], [0.8, 0.1, 0, 0.1, 0]]),
            torch.tensor([[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float),
        ),
    ],
)
def test_get_nearest_vertex_position(distribution, expected_vertex):
    vertex = _get_nearest_vertex_position(distribution)
    assert torch.allclose(vertex, expected_vertex)


def test_get_nearest_vertex_position_for_larger_shape():
    distribution = torch.tensor([[0, 0.3, 0.5, 0, 0.2], [0.8, 0.1, 0, 0.1, 0]]).repeat(
        2, 1, 1
    )
    expected_vertex = torch.tensor(
        [[0, 0, 1, 0, 0], [1, 0, 0, 0, 0]], dtype=torch.float
    ).repeat(2, 1, 1)

    # checks about input data
    assert distribution.shape == (2, 2, 5)
    assert expected_vertex.shape == (2, 2, 5)

    vertex = _get_nearest_vertex_position(distribution)

    assert torch.allclose(vertex, expected_vertex)


def test_compute_quotient_singeloton_tensors():
    dist = torch.tensor([[0], [1], [2]], dtype=torch.float)
    func = lambda x: 2 * x.squeeze(dim=-1)
    quotient_result = _difference_quotient(
        func, dist, func(dist), torch.tensor([0.1]).reshape((1, 1, 1))
    )
    assert torch.allclose(quotient_result, torch.full_like(quotient_result, 2))


def test_compute_quotient_2d_tensors():
    dist = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4]], dtype=torch.float)
    dist = dist.unsqueeze(1)
    func = lambda x: 2 * x.sum(dim=-1)
    quotient_result = _difference_quotient(
        func,
        dist,
        func(dist),
        (torch.eye(3, dtype=torch.float) * 0.1).unsqueeze(0),
    )
    assert torch.allclose(quotient_result, torch.full_like(quotient_result, 2))


def test_compute_quotient_0_samples_tensors():
    dist = torch.empty(0, 3, dtype=torch.float)
    dist = dist.unsqueeze(1)
    func = lambda x: 2 * x.sum(dim=-1)
    quotient_result = _difference_quotient(
        func,
        dist,
        func(dist),
        (torch.eye(3, dtype=torch.float) * 0.1).unsqueeze(0),
    )
    assert quotient_result.shape == (0, 1)


# TODO: Test for metrics,
# especially uncert_maximum_descent_ratio
@pytest.mark.parametrize(
    "uncert_func",
    [
        entropy,
        least_confidence,
        margin,
    ],
)
@pytest.mark.parametrize(
    "distibution",
    [
        torch.eye(10) * 0.9
        + 0.01,  # we are avoding "pure" points as they are not differenetiable
        torch.concatenate(
            [torch.full((8, 4), 1.05 / 8), torch.full((8, 4), 0.95 / 8)], dim=1
        ),
        random_proba((8, 8)),
    ],
)
def test_uncert_maximum_descent_ratio_0_is_prior_descent_ratio(
    uncert_func, distibution
):
    prior_descent_values = prior_descent_ratio(
        uncert_func._call, distribution=distibution
    )
    uncdert_max_values = uncert_maximum_descent_ratio(
        uncert_func._call, distribution=distibution
    )

    assert torch.allclose(prior_descent_values, uncdert_max_values[..., 0], rtol=1e-4)
