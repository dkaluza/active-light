from typing import Protocol

import torch

from al.base import ActiveState


class InformativenessProto(Protocol):
    def __call__(self, state: ActiveState) -> torch.FloatTensor: ...

    @property
    def name(self) -> str:
        return self.__class__.__name__
