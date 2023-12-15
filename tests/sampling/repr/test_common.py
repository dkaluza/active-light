import pytest
import torch
from torch.utils.data import TensorDataset

from al.sampling.base import ActiveInMemoryState, InformativenessProto
from al.sampling.repr.knn import KNearestNeighborRepr

REPR_FUNCTIONS = [KNearestNeighborRepr()]


@pytest.mark.parametrize("info", REPR_FUNCTIONS)
def test_returns_n_samples(
    info: InformativenessProto, state_with_non_empty_pool: ActiveInMemoryState
):
    n_samples = len(state_with_non_empty_pool.get_pool())
    values = info(state_with_non_empty_pool)

    assert values.shape == (n_samples,)


@pytest.mark.parametrize("info", REPR_FUNCTIONS)
def test_repr_returns_greater_value_for_cluster_center_than_outlier(
    info: InformativenessProto,
):
    n_samples_from_normal_dist = 200
    pool = torch.concat(
        [
            torch.tensor([[1e6, 1e6]]),
            torch.tensor([[0, 0]]),
            torch.normal(
                torch.zeros(n_samples_from_normal_dist, 2),
                torch.ones(n_samples_from_normal_dist, 2),
            ),
        ]
    )

    state = ActiveInMemoryState(pool=TensorDataset(pool))

    values = info(state)
    assert values[0] < values[1]
