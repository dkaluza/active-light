import pytest
import torch
from torch.utils.data import TensorDataset

from al.sampling.baseline import random


@pytest.mark.parametrize(
    "dataset, expected_shape",
    [
        (TensorDataset(torch.zeros(1, 3)), 1),
        (TensorDataset(torch.zeros(7, 10)), 7),
        (TensorDataset(torch.zeros(0, 5)), 0),
    ],
)
def test_random_returns_correct_shape_for_dataset(dataset, expected_shape):
    infos = random(dataset=dataset)
    assert infos.shape == (expected_shape,)
