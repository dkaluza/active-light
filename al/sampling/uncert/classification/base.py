"""All uncertainty functions expect classes to be passed as the last dimension in probas"""
import abc

import torch
from torch.utils.data import Dataset

from al.base import ModelProto
from al.sampling.base import InformativenessProto

CLASSES_DIM = -1


class UncertClassificationBase(InformativenessProto, abc.ABC):
    @abc.abstractmethod
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        ...

    def __call__(
        self,
        probas: torch.FloatTensor = None,
        model: ModelProto | None = None,
        dataset: Dataset | None = None,
    ) -> torch.FloatTensor:
        if probas is None:
            assert (
                model is not None and dataset is not None
            ), "In case of no probas passed model and dataset have to be defined."
            probas = model.predict_proba(dataset)

        if not len(probas):
            return torch.empty(0)
        self._validate_probas(probas)

        return self._call(probas)

    def _validate_probas(self, probas: torch.FloatTensor):
        if len(probas) and probas.shape[CLASSES_DIM] < 2:
            raise ValueError(
                "Uncertainty functions can only be used in case of distributions with at least 2 values."
            )
