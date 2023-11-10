from unittest.mock import Mock

import torch
from torch.utils.data import Dataset, TensorDataset

from al.base import ModelProto
from al.loops import run_experiment
from al.loops.base import LoopConfig, LoopMetric, LoopResults
from al.loops.perfect_oracle import active_learning_loop
from al.sampling.uncert.classical import entropy


def test_run_experiments_runs_for_selected_number_of_seeds():
    def naive_loop(*args, **kwargs) -> LoopResults:
        return LoopResults()

    def primitive_info1():
        ...

    def primitive_info2():
        ...

    n_repeats = 7
    results = run_experiment(
        naive_loop,
        data=TensorDataset(torch.zeros(1000, 10), torch.ones(1000)),
        infos=[primitive_info1, primitive_info2],
        n_repeats=n_repeats,
    )
    for _, res in results.items():
        assert len(res) == n_repeats


def test_run_experiments_uses_same_datasets_for_different_infos():
    mocked_loop = Mock()

    def primitive_info1():
        ...

    def primitive_info2():
        ...

    results = run_experiment(
        mocked_loop,
        data=TensorDataset(torch.arange(1000).unsqueeze(1), torch.ones(1000)),
        infos=[primitive_info1, primitive_info2],
    )
    all_calls_args = mocked_loop.call_args_list
    assert len(all_calls_args) == 20  # 2 infos * 10 default repeats
    info1_calls_datasets = _retrieve_datasets_for_info(all_calls_args, primitive_info1)
    info2_calls_datasets = _retrieve_datasets_for_info(all_calls_args, primitive_info2)
    assert info1_calls_datasets == info2_calls_datasets


def _retrieve_datasets_for_info(all_calls_args, info):
    return [
        (
            call_args.kwargs["initial_train"],
            call_args.kwargs["pool"],
            call_args.kwargs["test"],
        )
        for call_args in all_calls_args
        if call_args.kwargs["info_func"] == info
    ]


def test_run_experiments_uses_generates_different_intial_datasets():
    mocked_loop = Mock()

    def primitive_info1():
        ...

    results = run_experiment(
        mocked_loop,
        data=TensorDataset(torch.arange(1000).unsqueeze(1), torch.ones(1000)),
        infos=[primitive_info1],
    )

    all_calls_args = mocked_loop.call_args_list
    assert len(all_calls_args) == 10
    info1_calls_datasets = _retrieve_datasets_for_info(all_calls_args, primitive_info1)
    assert len(set(info1_calls_datasets)) == 10


def test_al_loop_experiment_generates_scores_for_metrics():
    class DumbModel(ModelProto):
        def fit(self, train: Dataset):
            return self

        def predict_proba(self, data: Dataset) -> torch.FloatTensor:
            return torch.full((len(data), 2), fill_value=0.5, dtype=torch.float)

    target = torch.ones(1000)
    results = run_experiment(
        active_learning_loop,
        data=TensorDataset(torch.arange(1000).unsqueeze(1), target),
        infos=[entropy],
        config=LoopConfig(metrics=[LoopMetric.BAC], n_classes=2),
        budget=100,
        model=DumbModel(),
    )

    assert len(results) == 1

    entropy_results = results[entropy.__name__]

    for res in entropy_results:
        assert all([res == 0 for res in res.metrics[LoopMetric.BAC]])
