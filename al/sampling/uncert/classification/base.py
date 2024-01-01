"""All uncertainty functions expect classes to be passed as the last dimension in probas"""
import abc

import torch

from al.base import CLASSES_DIM, ActiveState
from al.sampling.base import InformativenessProto


class UncertClassificationBase(InformativenessProto, abc.ABC):
    @abc.abstractmethod
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        ...

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        probas: torch.FloatTensor = state.get_probas()

        if not len(probas):
            return torch.empty(0)
        self._validate_probas(probas)

        return self._call(probas)

    def _validate_probas(self, probas: torch.FloatTensor):
        if len(probas) and probas.shape[CLASSES_DIM] < 2:
            raise ValueError(
                "Uncertainty functions can only be used in case of distributions with at least 2 values."
            )
