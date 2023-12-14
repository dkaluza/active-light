import torch
from torch.utils.data import Dataset

from al.base import ModelProto

from .base import ActiveState, InformativenessProto


class Random(InformativenessProto):
    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        probas = None
        if state.has_probas():
            probas = state.get_probas()
            n_samples = probas.shape[:-1]
        else:
            pool = state.get_pool()
            n_samples = len(pool)
        # TODO what about datasets with none singleton shape?
        requires_grad = probas.requires_grad if probas is not None else False
        return torch.rand(n_samples, requires_grad=requires_grad)


random = Random()
