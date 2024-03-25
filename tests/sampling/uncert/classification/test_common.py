import pytest
import torch
from torch.utils.data import TensorDataset
from xgboost import XGBRFClassifier

from al.base import ActiveInMemoryState
from al.loops.experiments import XGBWrapper
from al.sampling.uncert import (
    confidence_ratio,
    entropy_info,
    height_ratio_exponent_evidence,
    height_ratio_large_exponent_evidence,
    height_ratio_log_plus_evidence,
    least_confidence,
    margin,
    off_centered_entropy,
    pyramidal_exponent_evidence,
    pyramidal_large_exponent_evidence,
    pyramidal_log_plus_evidence,
)
from al.sampling.uncert.classification.base import UncertClassificationBase
from tests.helpers import probas_state

CLASSICAL_UNCERT_MEASURES = [
    entropy_info,
    least_confidence,
    margin,
    confidence_ratio,
]
EVIDENCE_BASED_UNCERT_MEASURES = [
    pyramidal_exponent_evidence,
    pyramidal_large_exponent_evidence,
    pyramidal_log_plus_evidence,
    height_ratio_exponent_evidence,
    height_ratio_large_exponent_evidence,
    height_ratio_log_plus_evidence,
]
PRIOR_BASED_UNCERT_MEASURES = [off_centered_entropy]


@pytest.fixture(
    params=CLASSICAL_UNCERT_MEASURES
    + EVIDENCE_BASED_UNCERT_MEASURES
    + PRIOR_BASED_UNCERT_MEASURES
)
def uncert_func(request: pytest.FixtureRequest) -> UncertClassificationBase:
    return request.param


@pytest.mark.parametrize(
    "probas",
    [
        torch.eye(2, dtype=torch.float),
        torch.tensor([[0, 0, 1], [1, 0, 0], [0, 1, 0], [0, 1, 0]], dtype=torch.float),
    ],
)
def test_uncert_pure_distribution_returns_zero(uncert_func, probas):
    state = probas_state(probas)
    uncert_values = uncert_func(state)
    assert torch.allclose(uncert_values, torch.zeros_like(uncert_values))


@pytest.mark.parametrize(
    "probas, expected_shape",
    [
        (torch.rand(2, 7), 2),
        (torch.rand(0, 2), 0),
        (torch.rand(1, 3), 1),
        (torch.rand(3, 2), 3),
    ],
)
def test_uncert_return_n_samples_shape(uncert_func, probas, expected_shape):
    state = probas_state(probas)
    assert uncert_func(state).shape == (expected_shape,)


@pytest.mark.parametrize(
    "probas",
    [
        torch.rand(1, 1),
        torch.rand(3, 1),
        torch.rand(2, 0),
    ],
)
def test_uncert_raises_on_wrong_input_shape(uncert_func, probas):
    state = probas_state(probas)

    with pytest.raises(ValueError):
        uncert_func(state)


@pytest.mark.parametrize(
    "dataset, expected_shape",
    [
        (TensorDataset(torch.rand(2, 7), torch.tensor([0, 1])), 2),
        (TensorDataset(torch.rand(1, 2), torch.tensor([0])), 1),
        (TensorDataset(torch.rand(3, 2), torch.tensor([0, 1, 2])), 3),
    ],
)
def test_uncert_with_model_returns_n_samples_shape(
    uncert_func, dataset, expected_shape
):
    model = XGBWrapper(XGBRFClassifier(n_jobs=1, n_estimators=10))
    model.fit(dataset)
    state = ActiveInMemoryState(model=model, pool=dataset)
    uncert_values: torch.FloatTensor = uncert_func(state)
    assert uncert_values.shape == (expected_shape,)
