from pytest_mock import MockerFixture

from al.sampling.base import ActiveState
from al.sampling.repr.knn import k_nearest_neighbor_repr


def test_knn_repr_uses_cache(
    state_with_random_pool: ActiveState, mocker: MockerFixture
):
    buil_index_spy = mocker.spy(k_nearest_neighbor_repr, "build_index")

    # first call should build index
    _ = k_nearest_neighbor_repr(state_with_random_pool)
    buil_index_spy.assert_called_once()

    state_spy = mocker.spy(state_with_random_pool, "get_from_cache")

    # second call shouldn't build index,
    # it should retrieve index from cache instead
    _ = k_nearest_neighbor_repr(state_with_random_pool)
    buil_index_spy.assert_called_once()
    state_spy.assert_called_once()
