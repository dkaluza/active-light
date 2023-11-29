import pathlib
import tempfile
from typing import Callable, Sequence
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
    ConfigurationResults,
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
        return LoopResults(metrics={"TEST": [0.1, 0.2, 0.3]})

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
    for _, res in results.res.items():
        assert len(res.metrics["TEST"]) == n_repeats


def test_run_experiments_uses_same_datasets_for_different_infos():
    mocked_loop = Mock()
    mocked_loop.return_value = LoopResults()

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


def test_run_experiments_uses_different_intial_datasets():
    mocked_loop = Mock()
    mocked_loop.return_value = LoopResults()

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
    ).res

    assert len(results) == 1

    entropy_results = results[entropy.__name__]

    bac = entropy_results.metrics[LoopMetric.BAC.name]
    assert bac.shape[0] >= 1
    assert bac.shape[1] == 101
    assert torch.allclose(bac, torch.zeros_like(bac))


EXAMPLARY_POOL_PROBAS = [torch.eye(3), torch.eye(3), torch.eye(3)]
EXAMPLARY_POOL_PROBAS_TENSOR = torch.stack(EXAMPLARY_POOL_PROBAS)

EXAMPLARY_METRIC_VALUES = [0.3, 0.3, 0.3]

EXAMPLARY_LOOP_RESULT = LoopResults(
    metrics={LoopMetric.BAC.name: EXAMPLARY_METRIC_VALUES},
    pool_probas=EXAMPLARY_POOL_PROBAS,
)


EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS = LoopResults(
    metrics={
        LoopMetric.BAC.name: [torch.tensor(0.3), torch.tensor(0.3), torch.tensor(0.3)]
    },
    pool_probas=EXAMPLARY_POOL_PROBAS,
)

TRIPLE_EXAMPLARY_CONFIG_RESULTS = ConfigurationResults(
    metrics={
        LoopMetric.BAC.name: torch.tensor(
            [
                EXAMPLARY_METRIC_VALUES,
                EXAMPLARY_METRIC_VALUES,
                EXAMPLARY_METRIC_VALUES,
            ]
        )
    },
    pool_probas=torch.stack(
        [
            EXAMPLARY_POOL_PROBAS_TENSOR,
            EXAMPLARY_POOL_PROBAS_TENSOR,
            EXAMPLARY_POOL_PROBAS_TENSOR,
        ]
    ),
)


@pytest.fixture(
    params=[
        (
            {"info1": [EXAMPLARY_LOOP_RESULT]},
            ExperimentResults(
                res={
                    "info1": ConfigurationResults(
                        metrics={
                            LoopMetric.BAC.name: torch.tensor([EXAMPLARY_METRIC_VALUES])
                        },
                        pool_probas=EXAMPLARY_POOL_PROBAS_TENSOR.unsqueeze(0),
                    )
                }
            ),
        ),
        (
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
            ExperimentResults(
                res={
                    "info1": TRIPLE_EXAMPLARY_CONFIG_RESULTS,
                    "info2": TRIPLE_EXAMPLARY_CONFIG_RESULTS,
                }
            ),
        ),
        (dict(), ExperimentResults(res={})),
        (
            {"info1": []},
            ExperimentResults(
                res={"info1": ConfigurationResults(metrics={}, pool_probas=None)}
            ),
        ),
        (
            {
                "info1": [
                    EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
                    EXAMPLARY_LOOP_RESULT_WTIH_TENSORS_AS_METRICS,
                ]
            },
            ExperimentResults(
                res={
                    "info1": ConfigurationResults(
                        metrics={
                            LoopMetric.BAC.name: torch.tensor(
                                [EXAMPLARY_METRIC_VALUES, EXAMPLARY_METRIC_VALUES]
                            )
                        },
                        pool_probas=torch.stack(
                            [EXAMPLARY_POOL_PROBAS_TENSOR, EXAMPLARY_POOL_PROBAS_TENSOR]
                        ),
                    )
                }
            ),
        ),
        (
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
                                        [0.5, 0.3, 0.2, 0.0],
                                        [0.5, 0.1, 0.4, 0.0],
                                        [0.5, 0.1, 0.4, 0.0],
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
            ExperimentResults(
                res={
                    "a": ConfigurationResults(
                        metrics={
                            LoopMetric.BAC.name: torch.tensor(
                                [
                                    [
                                        tensor(0.4616),
                                        tensor(0.4614),
                                        tensor(0.4806),
                                    ]
                                ]
                            )
                        },
                        pool_probas=torch.stack(
                            [
                                tensor(
                                    [
                                        [0.5, 0.3, 0.2, 0.0],
                                        [0.5, 0.1, 0.4, 0.0],
                                        [0.5, 0.1, 0.4, 0.0],
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
                                        [
                                            torch.nan,
                                            torch.nan,
                                            torch.nan,
                                            torch.nan,
                                        ],
                                    ]
                                ),
                            ],
                        ).unsqueeze(0),
                    )
                }
            ),
        ),
    ]
)
def experiment_results_with_loop_results_collection(request: pytest.FixtureRequest):
    return request.param


@pytest.fixture
def experiment_results(experiment_results_with_loop_results_collection):
    return experiment_results_with_loop_results_collection[1]


@pytest.fixture
def loop_results_collection(experiment_results_with_loop_results_collection):
    return experiment_results_with_loop_results_collection[0]


def test_experiment_results_from_loop_results_conversion(
    experiment_results_with_loop_results_collection,
):
    loop_results_collection = experiment_results_with_loop_results_collection[0]
    expected_experiment_result = experiment_results_with_loop_results_collection[1]

    assert (
        ExperimentResults.from_loop_results(loop_results_collection)
        == expected_experiment_result
    )


def test_save_results_is_loadable(experiment_results: ExperimentResults):
    # TODO: swap to new implementation
    with tempfile.TemporaryDirectory() as tmpdir:
        result_file_path = pathlib.Path(tmpdir, "results.bin")
        experiment_results.save(
            path=result_file_path,
        )
        loaded_results = ExperimentResults.load(result_file_path)

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
        results = run_experiment(
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
        loaded_results = ExperimentResults.load(result_file_path)

    assert loaded_results == results


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
