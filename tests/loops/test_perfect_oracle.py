from unittest.mock import Mock

import pytest
import torch
from pytest_mock import MockerFixture
from torch.utils.data import Dataset, Subset, TensorDataset

from al.base import ModelProto, remove_indices_from_dataset
from al.loops.base import ALDataset, LoopConfig
from al.loops.perfect_oracle import active_learning_loop
from al.sampling.base import InformativenessProto


class MockedInfo(InformativenessProto):
    """Informativeness that returns 1 if there is a feature with value 7, 0.8 if there is feature with value 9
    for the sample. Otherwise it always returns 0."""

    def __call__(self, state) -> torch.FloatTensor:
        dataset = ALDataset(state.get_pool())
        return self._call(dataset=dataset)

    def _call(self, dataset: ALDataset) -> torch.FloatTensor:
        info = torch.zeros(len(dataset), dtype=torch.float)
        info[(dataset.features == 7).any(dim=-1)] = 1.0
        info[(dataset.features == 9).any(dim=-1)] = 0.8

        return info


def check_if_call_was_with_only_expected_keys_converting_datasets(
    call_kwargs, call_keys, expected_values
):
    assert list(call_kwargs.keys()) == call_keys
    for key, value in call_kwargs.items():
        if isinstance(value, Dataset):
            assert ALDataset(value) == ALDataset(expected_values[key])
        else:
            assert value == expected_values[key]


N_SAMPLES = 20
POOL_DATASET = ALDataset(
    TensorDataset(
        torch.arange(N_SAMPLES, dtype=torch.float).reshape(-1, 1),
        torch.zeros(N_SAMPLES),
    )
)
INITIAL_DATASET = ALDataset(
    TensorDataset(
        torch.empty(0, 1),
        torch.zeros(0),
    )
)


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

    expected_call_kwargs = dict(
        train=Subset(POOL_DATASET, indices=[7]),
    )
    check_if_call_was_with_only_expected_keys_converting_datasets(
        call_args_kwargs[1].kwargs,
        list(expected_call_kwargs.keys()),
        expected_call_kwargs,
    )

    expected_call_kwargs = dict(
        train=Subset(POOL_DATASET, indices=[7, 9]),
    )
    check_if_call_was_with_only_expected_keys_converting_datasets(
        call_args_kwargs[2].kwargs,
        list(expected_call_kwargs.keys()),
        expected_call_kwargs,
    )


def test_active_learning_loop_removes_from_pool(mocker: MockerFixture):
    mocked_info = MockedInfo()
    spied_call = mocker.spy(mocked_info, "_call")
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

    assert call_args_kwargs[0].kwargs == dict(dataset=POOL_DATASET)

    assert call_args_kwargs[1].kwargs == dict(
        dataset=ALDataset(remove_indices_from_dataset(POOL_DATASET, indices=[7])),
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

    # assumption that loop is selecting values in order if they have the same
    # informativeness score

    expected_call_kwargs = dict(
        train=Subset(POOL_DATASET, indices=[7, 9] + list(range(batch_size - 2))),
    )
    check_if_call_was_with_only_expected_keys_converting_datasets(
        call_args_kwargs[1].kwargs,
        list(expected_call_kwargs.keys()),
        expected_call_kwargs,
    )
