import pytest
import torch
from torch.utils.data import TensorDataset

from al.base import ActiveInMemoryState, ActiveState
from al.sampling.base import InformativenessProto
from al.sampling.combination import InfoEnsemble, ProductAggregation, SumAggregation


class MockedInfo(InformativenessProto):
    """Informativeness that always returns just 2 unique values.
    One for first sample and other for rest of samples.
    """

    def __init__(self, value_for_first_sample, value_for_rest) -> None:
        super().__init__()
        self.val_first = value_for_first_sample
        self.val_rest = value_for_rest

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        n_samples = len(state.get_pool())
        values = torch.full((n_samples,), fill_value=self.val_rest)
        if n_samples >= 1:
            values[0] = self.val_first
        return values


@pytest.mark.parametrize(
    "info",
    [
        InfoEnsemble([]),
        InfoEnsemble([MockedInfo(0.25, 0.75)]),
        InfoEnsemble([MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)]),
        InfoEnsemble(
            [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)], weights=[0.5, 1.0]
        ),
        InfoEnsemble(
            [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)],
            aggregation_tactic=ProductAggregation(),
        ),
    ],
)
def test_info_ensemble_returns_n_values(
    info: InfoEnsemble, state_with_pool: ActiveState
):
    n_samples = len(state_with_pool.get_pool())
    values = info(state_with_pool)
    assert values.shape == (n_samples,)


@pytest.mark.parametrize(
    "info",
    [
        InfoEnsemble([]),
        InfoEnsemble([], aggregation_tactic=ProductAggregation()),
    ],
)
def test_ensemble_returns_constant_without_infos(
    info: InfoEnsemble, state_with_pool: ActiveState
):
    values = info(state_with_pool)

    assert len(values.unique()) == 1 or (
        len(state_with_pool.get_pool()) == 0 and values.shape[0] == 0
    )


@pytest.mark.parametrize(
    "info, expected_first_element_ge",
    [
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)],
            ),
            True,
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)],
                aggregation_tactic=ProductAggregation(),
            ),
            True,
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)], weights=[2.0, 1.0]
            ),
            False,
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)],
                weights=[2.0, 1.0],
                aggregation_tactic=ProductAggregation(),
            ),
            False,
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)], weights=[0.5, 1.0]
            ),
            True,
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.25, 0.75), MockedInfo(0.75, 0.25)],
                weights=[0.5, 1.0],
                aggregation_tactic=ProductAggregation(),
            ),
            True,
        ),
    ],
)
def test_ensemble_weighting(info: InfoEnsemble, expected_first_element_ge: bool):
    pool = torch.zeros((6, 2))
    state = ActiveInMemoryState(pool=TensorDataset(pool))
    values = info(state)
    is_first_element_ge = values[0] >= values[1]
    assert is_first_element_ge == expected_first_element_ge


@pytest.mark.parametrize(
    "info, expected_value",
    [
        (
            InfoEnsemble(
                [MockedInfo(0.2, 0.3), MockedInfo(0.3, 0.1)],
            ),
            (0.5, 0.4),
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.2, 0.3), MockedInfo(0.3, 0.1)],
                aggregation_tactic=ProductAggregation(),
            ),
            (0.06, 0.03),
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.2, 0.3), MockedInfo(0.3, 0.1)], weights=[2.0, 1.0]
            ),
            (0.7, 0.7),
        ),
        (
            InfoEnsemble(
                [MockedInfo(0.2, 0.3), MockedInfo(0.3, 0.1)],
                weights=[3.0, 1.0],
                aggregation_tactic=ProductAggregation(),
            ),
            (0.0024, 0.0027),
        ),
    ],
)
def test_aggregated_values(info, expected_value):
    pool = torch.zeros((6, 2))
    state = ActiveInMemoryState(pool=TensorDataset(pool))
    values = info(state)
    assert torch.allclose(values[:2], torch.tensor(expected_value))
