import pathlib
import tempfile
from typing import Callable
from unittest.mock import Mock

import pytest
import torch
from torch import tensor
from torch.utils.data import Dataset, TensorDataset
from xgboost import XGBRFClassifier

from al.base import ModelProto
from al.loops import run_experiment
from al.loops.base import ALDataset, LoopConfig, LoopMetric, LoopResults
from al.loops.experiments import (
    ExperimentResults,
    NClassesGuaranteeWrapper,
    XGBWrapper,
    add_uncert_metric_for_probas,
    load_results,
    save_results,
)
from al.loops.perfect_oracle import active_learning_loop
from al.sampling.uncert.classical import entropy
from al.sampling.uncert.metrics import (
    monotonicity_from_vertex,
    prior_descent_ratio,
    simplex_vertex_repel_ratio,
    uncert_maximum_descent_ratio,
)


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
        assert all([res == 0 for res in res.metrics[LoopMetric.BAC.name]])


EXAMPLARY_LOOP_RESULT = LoopResults(
    metrics={LoopMetric.BAC.name: [0.3, 0.3, 0.3]},
    pool_probas=[torch.eye(3), torch.eye(3), torch.eye(3)],
)

EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS = LoopResults(
    metrics={
        LoopMetric.BAC.name: [torch.tensor(0.3), torch.tensor(0.3), torch.tensor(0.3)]
    },
    pool_probas=[torch.eye(3), torch.eye(3), torch.eye(3)],
)


@pytest.mark.parametrize(
    "experiment_results",
    [
        {"info1": [EXAMPLARY_LOOP_RESULT]},
        {
            "info1": [
                EXAMPLARY_LOOP_RESULT,
                EXAMPLARY_LOOP_RESULT,
                EXAMPLARY_LOOP_RESULT,
            ],
            "info2": [
                EXAMPLARY_LOOP_RESULT,
                EXAMPLARY_LOOP_RESULT,
                EXAMPLARY_LOOP_RESULT,
            ],
        },
        dict(),
        {"info1": []},
        {
            "info1": [
                EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
                EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
            ]
        },
        {
            "a": [
                LoopResults.model_validate(
                    {
                        "metrics": {
                            "BAC": [
                                tensor(0.4616),
                                tensor(0.4614),
                                tensor(0.4806),
                            ]
                        },
                        "pool_probas": [
                            tensor(
                                [
                                    [
                                        0.5,
                                        0.3,
                                        0.2,
                                    ],
                                    [
                                        0.5,
                                        0.1,
                                        0.4,
                                    ],
                                ]
                            ),
                            tensor(
                                [
                                    [
                                        0.8,
                                        0.1,
                                        0.0,
                                        0.1,
                                    ],
                                    [
                                        0.05,
                                        0.9,
                                        0.03,
                                        0.02,
                                    ],
                                ]
                            ),
                        ],
                    }
                )
            ]
        },
    ],
)
def test_save_results_is_loadable(experiment_results: ExperimentResults):
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file_path = pathlib.Path(tmpdir, "results.bin")
        save_results(path=result_file_path, results=experiment_results)
        loaded_results = load_results(result_file_path)
    assert loaded_results == experiment_results


@pytest.mark.parametrize(
    "config",
    [
        LoopConfig(),
        LoopConfig(metrics=[LoopMetric.BAC]),
        LoopConfig(metrics=[LoopMetric.BAC], return_pool_probas=True),
    ],
)
def test_run_experiments_save_results_is_loaded(config: LoopConfig):
    dataset = TensorDataset(
        torch.rand((20, 5)),
        torch.tensor([0, 1] * 10, dtype=torch.long),
    )
    model = XGBWrapper(
        XGBRFClassifier(n_jobs=1, objective="multi:softprob", num_class=2)
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file_path = pathlib.Path(tmpdir, "results.bin")
        resuts = run_experiment(
            active_learning_loop,
            data=dataset,
            model=model,
            init_frac=12,
            test_frac=2,
            budget=2,
            n_repeats=1,
            save_path=result_file_path,
            infos=[entropy],
            config=config,
        )
        loaded_results = load_results(result_file_path)

    assert loaded_results == resuts


class MockedModel(ModelProto):
    def fit(self, train: Dataset):
        train = ALDataset(train)
        self.n_classes = train.n_classes
        return super().fit(train)

    def predict_proba(self, data: Dataset) -> torch.FloatTensor:
        data = ALDataset(data)
        return torch.ones((len(data), self.n_classes), dtype=torch.float)


@pytest.mark.parametrize(
    "dataset, n_classes",
    [
        (TensorDataset(torch.ones((3, 10)), torch.tensor([0, 1, 2])), 3),
        (TensorDataset(torch.ones((2, 10)), torch.tensor([0, 3])), 4),
        (TensorDataset(torch.ones((1, 10)), torch.tensor([0])), 5),
        (TensorDataset(torch.ones((1, 10)), torch.tensor([3])), 5),
    ],
)
def test_nclassesguranteewrapper_returns_n_classes(dataset, n_classes):
    model = NClassesGuaranteeWrapper(MockedModel(), n_classes=n_classes)
    model.fit(dataset)
    probas = model.predict_proba(dataset)
    assert probas.shape[1] == n_classes


@pytest.mark.parametrize(
    "dataset, n_classes",
    [
        (TensorDataset(torch.ones((4, 10)), torch.tensor([0, 1, 2, 3])), 4),
        (TensorDataset(torch.ones((2, 10)), torch.tensor([0, 3])), 4),
        (TensorDataset(torch.ones((1, 10)), torch.tensor([0])), 5),
        (TensorDataset(torch.ones((1, 10)), torch.tensor([3])), 5),
    ],
)
def test_nclassesguranteewrapper_maintains_probas(dataset, n_classes):
    dataset = ALDataset(dataset)
    model = NClassesGuaranteeWrapper(MockedModel(), n_classes=n_classes)
    model.fit(dataset)
    probas = model.predict_proba(dataset)
    missing_targets = torch.full((n_classes,), fill_value=True, dtype=torch.bool)
    missing_targets[dataset.targets] = False
    assert torch.all(probas[:, dataset.targets] == 1)
    assert torch.all(probas[:, missing_targets] == 0)


@pytest.mark.parametrize(
    "experiment_results",
    [
        {
            entropy.__name__: [
                EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
                EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
            ]
        }
    ],
)
@pytest.mark.parametrize(
    "metric_func",
    [
        monotonicity_from_vertex,
        simplex_vertex_repel_ratio,
        uncert_maximum_descent_ratio,
        prior_descent_ratio,
    ],
)
def test_add_uncert_metric_for_probas_adds_metric(
    experiment_results: ExperimentResults, metric_func: Callable
):
    add_uncert_metric_for_probas(
        experiment_results, uncerts=[entropy], metric=metric_func, name="test"
    )
    for _, val in experiment_results.items():
        for loop_result in val:
            assert "test" in loop_result.metrics
            assert len(loop_result.metrics["test"]) == len(loop_result.pool_probas)
