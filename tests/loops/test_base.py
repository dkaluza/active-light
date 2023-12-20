import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import ConcatDataset, Dataset, Subset, TensorDataset

from al.loops.base import ALDatasetWithoutTargets

N_SAMPLES = 100
N_FEATURES = 10
RAND_TENSOR = torch.rand(N_SAMPLES, N_FEATURES)
RAND_TENSOR_DATASET = TensorDataset(RAND_TENSOR)


@pytest.mark.parametrize(
    "dataset, expected_tensor",
    [
        (RAND_TENSOR_DATASET, RAND_TENSOR),
        (
            ConcatDataset([RAND_TENSOR_DATASET, RAND_TENSOR_DATASET]),
            torch.concat([RAND_TENSOR, RAND_TENSOR]),
        ),
        (Subset(RAND_TENSOR_DATASET, indices=[0, 1, 2, 3]), RAND_TENSOR[:4]),
        (
            ConcatDataset(
                [
                    Subset(RAND_TENSOR_DATASET, indices=[0, 1]),
                    Subset(RAND_TENSOR_DATASET, indices=[2, 3]),
                ]
            ),
            RAND_TENSOR[:4],
        ),
    ],
)
def test_ALDatasetWithoutTargets_optimized_retrival_leads_to_appropriate_tensor(
    dataset: Dataset, expected_tensor: torch.Tensor, mocker: MockerFixture
):
    al_dataset = ALDatasetWithoutTargets(dataset)

    mock_iteration = mocker.spy(al_dataset, "_initialize_features_by_iteration")
    features = al_dataset.features
    assert torch.all(features == expected_tensor)
    mock_iteration.assert_not_called()

    assert torch.all(features == al_dataset._retrieve_by_iteration(index_to_retrieve=0))
