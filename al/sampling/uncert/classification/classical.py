"""
Set of uncetainty functions. 

Every function is expected to have only one required argument, i.e. probability distribution with shape `(n_samples, n_classes)`.
The functions return a tensor of shape `(n_samples,)`, where higher values indicate uncertain samples.
"""


import torch

from al.distances import entropy as dist_entropy

from .base import CLASSES_DIM, UncertClassificationBase


class Entropy(UncertClassificationBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        return dist_entropy(probas=probas, dim=CLASSES_DIM)


class Margin(UncertClassificationBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        probas_sorted = torch.sort(probas, descending=True, dim=CLASSES_DIM).values
        return 1 - (probas_sorted[..., 0] - probas_sorted[..., 1])


class ConfidenceRatio(UncertClassificationBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        probas_sorted = torch.sort(probas, descending=True, dim=CLASSES_DIM).values
        return probas_sorted[..., 1] / probas_sorted[..., 0]


class LeastConfidence(UncertClassificationBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        max_proba = torch.max(probas, dim=CLASSES_DIM).values
        return 1 - max_proba


entropy_info = Entropy()
margin = Margin()
confidence_ratio = ConfidenceRatio()
least_confidence = LeastConfidence()
