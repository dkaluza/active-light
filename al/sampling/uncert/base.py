"""All uncertainty functions expect classes to be passed as the last dimension in probas"""
import abc

import torch
from torch.utils.data import Dataset

from al.base import ModelProto
from al.sampling.base import InformativenessProto

CLASSES_DIM = -1


def select_by_classes_dim(tensor: torch.Tensor, slice_to_select: slice | int):
    indexing = [slice(None, None) for _ in tensor.shape]
    indexing[CLASSES_DIM] = slice_to_select
    return tensor[*indexing]


class UncertBase(InformativenessProto, abc.ABC):
    @abc.abstractmethod
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        ...

    def __call__(
        self,
        probas: torch.Tensor = None,
        model: ModelProto | None = None,
        pool: Dataset | None = None,
    ) -> torch.FloatTensor:
        if probas is None:
            assert (
                model is not None and pool is not None
            ), "In case of no probas passed model and pool have to be defined."
            probas = model.predict_proba(pool)
        if not len(probas):
            return torch.empty(0)
        self._validate_probas(probas)

        return self._call(probas)

    def _validate_probas(self, probas: torch.FloatTensor):
        if len(probas) and probas.shape[CLASSES_DIM] < 2:
            raise ValueError(
                "Uncertainty functions can only be used in case of distributions with at least 2 values."
            )

    @property
    def __name__(self):
        return self.__class__.__name__
