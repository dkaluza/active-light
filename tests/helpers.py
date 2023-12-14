import torch

from al.sampling.base import ActiveInMemoryState, ActiveState
from al.sampling.uncert.classification.base import CLASSES_DIM


def random_proba(shape) -> torch.FloatTensor:
    values = torch.rand(shape)
    return torch.nn.functional.normalize(values, p=1, dim=CLASSES_DIM)


def probas_state(probas) -> ActiveState:
    return ActiveInMemoryState(probas=probas)
