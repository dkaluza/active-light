import pytest
import torch
from torch.utils.data import TensorDataset

from al.sampling.base import ActiveInMemoryState


@pytest.fixture(
    params=[torch.rand(7, 3), torch.zeros((6, 2)), torch.eye(3), torch.ones((0, 3))]
)
def state_with_pool(request: pytest.FixtureRequest):
    pool = request.param
    return ActiveInMemoryState(pool=TensorDataset(pool))


@pytest.fixture(params=[torch.rand(7, 3), torch.zeros((6, 2)), torch.eye(3)])
def state_with_non_empty_pool(request: pytest.FixtureRequest):
    pool = request.param
    return ActiveInMemoryState(pool=TensorDataset(pool))


@pytest.fixture(scope="function")
def state_with_random_pool():
    n_samples_from_normal_dist = 200
    pool = torch.normal(
        torch.zeros(n_samples_from_normal_dist, 4),
        torch.ones(n_samples_from_normal_dist, 4),
    )
    return ActiveInMemoryState(pool=TensorDataset(pool))
