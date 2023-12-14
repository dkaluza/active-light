import pytest
import torch

from al.sampling.uncert import entropy, off_centered_entropy
from al.sampling.uncert.classification.base import CLASSES_DIM
from al.sampling.uncert.classification.prior import OffCenteredEntropy
from tests.helpers import probas_state, random_proba


@pytest.mark.parametrize("probas", [torch.eye(5), torch.full((4, 4), fill_value=1 / 4)])
def test_off_centered_entropy_is_equal_to_entropy_for_balanced_data(probas):
    state = probas_state(probas)
    assert torch.allclose(entropy(state), off_centered_entropy(state))


@pytest.mark.parametrize(
    "probas", [random_proba((7, 10)), random_proba((3, 5)), random_proba((5, 2))]
)
def test_off_centered_entropy_is_equal_to_entropy_for_uniform_max_location(probas):
    n_classes = probas.shape[CLASSES_DIM]

    maximum_loc = torch.full((n_classes,), fill_value=1 / n_classes)
    off_centered_entropy_with_uniform_max = OffCenteredEntropy(maximum_loc=maximum_loc)

    state = probas_state(probas)
    assert torch.allclose(entropy(state), off_centered_entropy_with_uniform_max(state))


MAXIMUM_LOC = torch.tensor([0.2, 0.6, 0.1, 0.05, 0.05])


@pytest.mark.parametrize("uncert", [OffCenteredEntropy(maximum_loc=MAXIMUM_LOC)])
@pytest.mark.parametrize(
    "probas", [random_proba((7, 5)), random_proba((3, 5)), random_proba((5, 5))]
)
def test_prior_based_measure_obtain_max_in_prior(uncert, probas):
    state = probas_state(probas)
    max_loc_state = probas_state(MAXIMUM_LOC.unsqueeze(0))
    max_loc_uncert = uncert(max_loc_state)
    probas_uncert = uncert(state)

    assert torch.all(max_loc_uncert >= probas_uncert)
