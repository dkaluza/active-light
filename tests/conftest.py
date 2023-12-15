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
