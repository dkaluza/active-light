import torch
from torch.utils.data import Dataset

from al.base import ModelProto

from .base import InformativenessProto


class Random(InformativenessProto):
    def __call__(
        self,
        probas: torch.Tensor = None,
        model: ModelProto | None = None,
        dataset: Dataset | None = None,
    ) -> torch.FloatTensor:
        assert probas is not None or dataset is not None
        n_samples = (
            probas.shape[:-1] if probas is not None else len(dataset)
        )  # TODO what about datasets with none singleton shape?
        requires_grad = probas.requires_grad if probas is not None else False
        return torch.rand(n_samples, requires_grad=requires_grad)


random = Random()
