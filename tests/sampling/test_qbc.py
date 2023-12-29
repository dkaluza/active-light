import pytest
import torch
from torch.utils.data import ConcatDataset, TensorDataset
from xgboost import XGBRFClassifier
from xgboost_distribution import XGBDistribution

from al.loops.base import ALDatasetWithoutTargets
from al.loops.experiments import (
    NClassesGuaranteeWrapper,
    XGBDistributionRegressionWrapper,
    XGBWrapper,
)
from al.sampling.base import ActiveInMemoryState
from al.sampling.qbc import Ambiguity, BootstrapJS, BootstrapTactic, RandomSplitTactic

REGRESSION_MODEL = XGBDistributionRegressionWrapper(XGBDistribution(n_estimators=10))
REGRESSION_TRAINING_LABELS = torch.rand(10)
CLASSIFICATION_MODEL = NClassesGuaranteeWrapper(
    XGBWrapper(XGBRFClassifier(n_estimators=10)), n_classes=2
)
CLASSIFICATION_TRAINING_LABELS = torch.randint(0, 2, size=(10,))


@pytest.mark.parametrize(
    "pool",
    [
        torch.rand(2, 7),
        torch.rand(1, 3),
        torch.rand(3, 2),
    ],
)
@pytest.mark.parametrize(
    "qbc, model, training_labels",
    [
        (Ambiguity(), REGRESSION_MODEL, REGRESSION_TRAINING_LABELS),
        (BootstrapJS(), CLASSIFICATION_MODEL, CLASSIFICATION_TRAINING_LABELS),
    ],
)
def test_qbc_return_n_samples_shape(pool, qbc, model, training_labels):
    expected_shape = pool.shape[0]
    n_features = pool.shape[1]
    state = ActiveInMemoryState(
        model=model,
        pool=TensorDataset(pool),
        training_data=TensorDataset(
            torch.rand(training_labels.shape[0], n_features), training_labels
        ),
    )

    assert qbc(state).shape == (expected_shape,)


def test_bootstrap_tactic_returns_repeated_indices():
    tactic = BootstrapTactic()
    for samples in tactic(
        5, torch.Generator().manual_seed(42), TensorDataset(torch.arange(10_000))
    ):
        dataset: ALDatasetWithoutTargets = ALDatasetWithoutTargets(samples)
        assert len(torch.unique(dataset.features)) < len(dataset)


def test_bootstrap_tactic_returns_n_samples():
    tactic = BootstrapTactic()
    n_samples = 10_000
    for samples in tactic(
        5, torch.Generator().manual_seed(42), TensorDataset(torch.arange(n_samples))
    ):
        dataset: ALDatasetWithoutTargets = ALDatasetWithoutTargets(samples)

        assert len(dataset) == n_samples


def test_random_split_tactic_returns_whole_dataset_splitted_evenly():
    tactic = RandomSplitTactic()
    n_samples = 10_000
    n_splits = 5
    all_samples = []
    for samples in tactic(
        n_splits,
        torch.Generator().manual_seed(42),
        TensorDataset(torch.arange(n_samples)),
    ):
        assert len(samples) == n_samples / n_splits
        all_samples.append(samples)

    dataset: ALDatasetWithoutTargets = ALDatasetWithoutTargets(
        ConcatDataset(all_samples)
    )
    assert len(dataset) == n_samples
    assert len(torch.unique(dataset.features)) == n_samples
