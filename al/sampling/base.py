from typing import Protocol

import torch
from torch.utils.data import Dataset

from al.base import ModelProto


class InformativenessProto(Protocol):
    def __call__(
        self,
        probas: torch.Tensor = None,
        model: ModelProto | None = None,
        dataset: Dataset | None = None,
    ) -> torch.FloatTensor:
        ...

    @property
    def __name__(self) -> str:
        return self.__class__.__name__
