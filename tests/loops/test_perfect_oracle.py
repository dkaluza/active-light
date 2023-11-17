from unittest.mock import Mock

import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import Dataset, Subset, TensorDataset

from al.base import ModelProto
from al.loops.base import ALDataset, ALDatasetWithoutTargets, LoopConfig
from al.loops.perfect_oracle import active_learning_loop, remove_indices_from_dataset
from al.sampling.base import InformativenessProto


class MockedInfo(InformativenessProto):
    """Informativeness that returns 1 if there is a feature with value 7, 0.8 if there is feature with value 9
    for the sample. Otherwise it always returns 0."""

    def __call__(
        self,
        probas: torch.Tensor = None,
        model: ModelProto | None = None,
        dataset: Dataset | None = None,
    ) -> torch.FloatTensor:
        dataset = ALDataset(dataset)
        info = torch.zeros(len(dataset), dtype=torch.float)
        info[(dataset.features == 7).any(dim=-1)] = 1.0
        info[(dataset.features == 9).any(dim=-1)] = 0.8

        return info


N_SAMPLES = 20
POOL_DATASET = ALDataset(
    TensorDataset(
        torch.arange(N_SAMPLES, dtype=torch.float).reshape(-1, 1),
        torch.zeros(N_SAMPLES),
    )
)
INITIAL_DATASET = ALDataset(TensorDataset(torch.empty(0, 1)))


def test_active_learning_loop_chooses_most_informative_sample():
    model: ModelProto = Mock(spec=ModelProto)
    test = POOL_DATASET
    active_learning_loop(
        initial_train=INITIAL_DATASET,
        pool=POOL_DATASET,
        test=test,
        info_func=MockedInfo(),
        budget=2,
        config=LoopConfig(),
        model=model,
    )

    assert model.fit.call_count == 3
    call_args_kwargs = model.fit.call_args_list
    assert call_args_kwargs[0].kwargs == dict(train=INITIAL_DATASET)
    assert call_args_kwargs[1].kwargs == dict(
        train=ALDataset(Subset(POOL_DATASET, indices=[7]))
    )
    assert call_args_kwargs[2].kwargs == dict(
        train=ALDataset(Subset(POOL_DATASET, indices=[7, 9]))
    )


def test_active_learning_loop_removes_from_pool(mocker: MockerFixture):
    mocked_info = MockedInfo()
    spied_call = mocker.spy(MockedInfo, "__call__")  # __call__ bypases instance lookup
    # according to https://github.com/pytest-dev/pytest-mock/issues/325 this in such case
    # spy should be done on class
    model: ModelProto = Mock(spec=ModelProto)

    active_learning_loop(
        initial_train=INITIAL_DATASET,
        pool=POOL_DATASET,
        test=POOL_DATASET,
        info_func=mocked_info,
        budget=2,
        config=LoopConfig(),
        model=model,
    )
    assert spied_call.call_count == 2
    call_args_kwargs = spied_call.call_args_list
    assert call_args_kwargs[0].kwargs == dict(dataset=POOL_DATASET, model=model)
    assert call_args_kwargs[1].kwargs == dict(
        dataset=ALDataset(remove_indices_from_dataset(POOL_DATASET, indices=[7])),
        model=model,
    )


@pytest.mark.parametrize("batch_size", [2, 5])
def test_active_learning_loop_chooses_most_informative_sample_for_larger_batch(
    batch_size,
):
    model: ModelProto = Mock(spec=ModelProto)
    active_learning_loop(
        initial_train=INITIAL_DATASET,
        pool=POOL_DATASET,
        test=POOL_DATASET,
        info_func=MockedInfo(),
        budget=batch_size,
        config=LoopConfig(batch_size=batch_size),
        model=model,
    )

    assert model.fit.call_count == 2
    call_args_kwargs = model.fit.call_args_list
    assert call_args_kwargs[0].kwargs == dict(train=INITIAL_DATASET)

    # assumption,
    assert call_args_kwargs[1].kwargs == dict(
        train=ALDataset(
            Subset(POOL_DATASET, indices=[7, 9] + list(range(batch_size - 2)))
        )
    )


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
