import pytest
import torch
from torch.utils.data import TensorDataset
from xgboost_distribution import XGBDistribution

from al.loops.experiments import XGBDistributionRegressionWrapper
from al.sampling.base import ActiveInMemoryState
from al.sampling.qbc import AmbiguityQBC


@pytest.mark.parametrize(
    "pool",
    [
        torch.rand(2, 7),
        torch.rand(0, 2),
        torch.rand(1, 3),
        torch.rand(3, 2),
    ],
)
def test_qbc_return_n_samples_shape(pool):
    expected_shape = pool.shape[0]
    n_features = pool.shape[1]
    state = ActiveInMemoryState(
        model=XGBDistributionRegressionWrapper(XGBDistribution(n_estimators=10)),
        pool=TensorDataset(pool),
        training_data=TensorDataset(torch.rand(10, n_features), torch.rand(10)),
    )
    qbc = AmbiguityQBC()

    assert qbc(state).shape == (expected_shape,)
