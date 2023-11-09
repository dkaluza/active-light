"""
Set of uncetainty functions. 

Every function is expected to have only one required argument, i.e. probability distribution with shape `(n_samples, n_classes)`.
The functions return a tensor of shape `(n_samples,)`, where higher values indicate uncertain samples.
"""


import torch

from .base import CLASSES_DIM, UncertBase


class Entropy(UncertBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        uncert = torch.where(probas == 0, 0, probas * torch.log2(probas))
        return -torch.sum(uncert, dim=CLASSES_DIM)


class Margin(UncertBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        probas_sorted = torch.sort(probas, descending=True, dim=CLASSES_DIM).values
        return 1 - (probas_sorted[..., 0] - probas_sorted[..., 1])


class ConfidenceRatio(UncertBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        probas_sorted = torch.sort(probas, descending=True, dim=CLASSES_DIM).values
        return probas_sorted[..., 1] / probas_sorted[..., 0]


class LeastConfidence(UncertBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        max_proba = torch.max(probas, dim=CLASSES_DIM).values
        return 1 - max_proba


entropy = Entropy()
margin = Margin()
confidence_ratio = ConfidenceRatio()
least_confidence = LeastConfidence()
