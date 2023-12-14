import pytest
import torch
from torch.utils.data import TensorDataset

from al.sampling.base import ActiveInMemoryState
from al.sampling.baseline import random

ActiveInMemoryState


@pytest.mark.parametrize(
    "state, expected_shape",
    [
        (ActiveInMemoryState(pool=TensorDataset(torch.zeros(1, 3))), 1),
        (ActiveInMemoryState(pool=TensorDataset(torch.zeros(7, 10))), 7),
        (ActiveInMemoryState(pool=TensorDataset(torch.zeros(0, 5))), 0),
        (ActiveInMemoryState(probas=torch.eye(3)), 3),
    ],
)
def test_random_returns_correct_shape_for_dataset(state, expected_shape):
    infos = random(state=state)
    assert infos.shape == (expected_shape,)
