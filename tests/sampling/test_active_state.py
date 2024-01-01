from typing import NamedTuple

import pytest
import torch
from pytest import FixtureRequest
from sklearn.dummy import DummyClassifier
from torch.utils.data import TensorDataset

from al.base import ActiveInMemoryState, ActiveState
from al.loops.base import ALDataset
from al.loops.experiments import ScikitWrapper
from tests.helpers import random_proba


class StateParams(NamedTuple):
    class_: type
    constructor_params: dict


@pytest.fixture(
    params=[
        StateParams(
            class_=ActiveInMemoryState,
            constructor_params=dict(
                model=ScikitWrapper(
                    DummyClassifier().fit(
                        X=random_proba((10, 5)).numpy(),
                        y=torch.randint(high=2, size=(10,)).numpy(),
                    )
                ),
                pool=TensorDataset(random_proba((10, 5))),
                training_data=TensorDataset(torch.empty(0, 5), torch.empty(0, 1)),
            ),
        )
    ]
)
def state(request: FixtureRequest) -> ActiveState:
    state_params = request.param
    return state_params.class_(**state_params.constructor_params)


@pytest.mark.parametrize(
    "test_value",
    ["TEST_VALUE", 7, DummyClassifier()],
)
def test_cached_object_is_retrievable(state: ActiveState, test_value):
    test_key = "TEST"
    state.save_in_cache(test_key, test_value)
    retrieved_value = state.get_from_cache(test_key)
    assert retrieved_value is not None
    assert retrieved_value == test_value


def test_cache_returns_None_for_non_existing_key(state: ActiveState):
    retrieved_value = state.get_from_cache("TEST")
    assert retrieved_value is None


def test_probas_are_computed_from_model_and_pool_if_missing(state: ActiveState):
    assert not state.has_probas()
    probas = state.get_probas()
    assert probas.shape[0] > 0
    assert torch.all(probas >= 0)
    assert torch.all(probas <= 1)


def test_select_samples_removes_from_pool_when_flag_true(state: ActiveState):
    pool = state.get_pool()
    original_features = ALDataset(pool).features
    selected_idx = [7, 2]
    state.select_samples(selected_idx, remove_from_pool=True)
    selected_samples = torch.ones(original_features.shape[0], dtype=torch.bool)
    selected_samples[selected_idx] = False
    features_after_select = ALDataset(state.get_pool()).features
    assert torch.all(original_features[selected_samples] == features_after_select)


def test_select_samples_maintains_pool_when_flag_false(state: ActiveState):
    pool = state.get_pool()
    original_features = ALDataset(pool).features
    state.select_samples([7], remove_from_pool=False)
    features_after_select = ALDataset(state.get_pool()).features
    assert torch.all(original_features == features_after_select)
