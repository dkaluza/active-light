from typing import NamedTuple, Sequence

import torch

from .base import UncertBase


class ProbaLayers(NamedTuple):
    probas: torch.FloatTensor
    """ 
    Tensor of shape `(n_samples, n_layers)` containing unique proba levels, padded with 0s at the end.
    """
    layers: torch.BoolTensor
    """ Tensor of shape `(n_samples, n_layers, n_classes)`.
    Value [i, j, k] denotes whether for sample i, k-th class belongs to layered set j.
    """


def _compute_proba_layers(probas: torch.FloatTensor) -> ProbaLayers:
    """Compute layered sets and corresponding probability differences.

    Layered sets are returned in order from the largest one in terms of
    inclusion to the smallest.

    Based on "On Several New Dempster-Shafer-inspired
    Uncertainty Measures Applicable for Active Learning" D. Kaluza et al.


    Parameters
    ----------
    probas : torch.FloatTensor
        Probability distribution for which layered sets should be computed/

    Returns
    -------
    ProbaLayers
         - probas - probability differences corresponding to the layered sets
         - layers - indicators of classes belonging to layered sets
    """
    n_samples, n_classes = probas.shape
    unique_probas = torch.nn.utils.rnn.pad_sequence(
        [sample.unique() for sample in probas], batch_first=True
    )
    n_layers = unique_probas.shape[1]

    returned_probas = torch.zeros_like(unique_probas)
    layers = torch.zeros((n_samples, n_layers, n_classes), dtype=torch.bool)
    previous_proba = 0
    for i in range(n_layers):
        considered_proba = unique_probas[:, i]
        proba_difference = considered_proba - previous_proba
        returned_probas[:, i] = torch.where(
            proba_difference > 0, proba_difference, torch.zeros_like(proba_difference)
        )

        layers[:, i] = probas >= considered_proba.unsqueeze(dim=1)
        previous_proba = considered_proba

    return ProbaLayers(probas=returned_probas, layers=layers)


class PyramidalEvidence(UncertBase):
    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        return super()._call(probas)
