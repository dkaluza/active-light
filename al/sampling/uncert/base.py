import abc

import torch


class UncertBase(abc.ABC):
    @abc.abstractmethod
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        ...

    def __call__(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        if not len(probas):
            return torch.empty(0)
        self._validate_probas(probas)

        return self._call(probas)

    def _validate_probas(self, probas: torch.FloatTensor):
        if len(probas) and probas.shape[1] < 2:
            raise ValueError(
                "Uncertainty functions can only be used in case of distributions with at least 2 values"
            )
