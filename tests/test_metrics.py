import pytest
import torch

from al.sampling.uncert.metrics import _get_nearest_vertex_position


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
