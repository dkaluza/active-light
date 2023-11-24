import pytest
import torch

from al.sampling.uncert.metrics import (
    _difference_quotient,
    _get_nearest_vertex_position,
)


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
