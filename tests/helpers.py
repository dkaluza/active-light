from unittest.mock import Mock

import torch

from al.base import ActiveInMemoryState, ActiveState, ModelProto
from al.sampling.uncert.classification.base import CLASSES_DIM


def random_proba(shape) -> torch.FloatTensor:
    values = torch.rand(shape)
    return torch.nn.functional.normalize(values, p=1, dim=CLASSES_DIM)


def probas_state(probas) -> ActiveState:
    return ActiveInMemoryState(probas=probas)


def probas_logit_state(probas: torch.Tensor, logits: torch.Tensor) -> ActiveState:
    """Get mocked state with predefined probas and logits.

    Parameters
    ----------
    probas : torch.Tensor
        Probas that will be returned from the state.
    logits : torch.Tensor
        Logits that will be returned from the mocked model.

    Returns
    -------
    ActiveState
        State with probas available and pool and model
            that will always return provided logits.
    """
    model = Mock(spec=ModelProto)
    model.predict_logits = lambda x: logits
    return ActiveInMemoryState(probas=probas, pool=logits, model=model)
