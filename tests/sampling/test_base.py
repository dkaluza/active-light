import pytest
import torch
from torch.utils.data import TensorDataset

from al.loops.base import ALDatasetWithoutTargets
from al.sampling.base import remove_indices_from_dataset

TENSOR_DATASET_RANGE_4 = TensorDataset(torch.arange(4, dtype=torch.float))


@pytest.mark.parametrize(
    "dataset, indices, expected_dataset",
    [
        (
            TENSOR_DATASET_RANGE_4,
            [2],
            TensorDataset(torch.tensor([0, 1, 3], dtype=torch.float)),
        ),
        (
            TENSOR_DATASET_RANGE_4,
            [1, 2],
            TensorDataset(torch.tensor([0, 3], dtype=torch.float)),
        ),
        (
            TENSOR_DATASET_RANGE_4,
            [],
            TENSOR_DATASET_RANGE_4,
        ),
        (
            TENSOR_DATASET_RANGE_4,
            [0, 1, 2, 3],
            TensorDataset(torch.tensor([], dtype=torch.float)),
        ),
    ],
)
def test_remove_indices_from_dataset_removes_selected_indices(
    dataset, indices, expected_dataset
):
    result = remove_indices_from_dataset(dataset=dataset, indices=indices)

    assert ALDatasetWithoutTargets(expected_dataset) == ALDatasetWithoutTargets(result)
