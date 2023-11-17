import functools
from typing import Callable, NamedTuple, Sequence

import torch

from .base import CLASSES_DIM, UncertBase


class ProbaLayers(NamedTuple):
    probas: torch.FloatTensor
    """ 
    Tensor of shape `(n_samples, ..., n_layers)` containing unique proba levels, padded with 0s at the end.
    """
    layers: torch.BoolTensor
    """ Tensor of shape `(n_samples,  ..., n_layers, n_classes)`.
    In case of no additional dimensions
    balue [i, j, k] denotes whether for sample i, k-th class belongs to layered set j.
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
        Probability distribution for which layered sets should be computed

    Returns
    -------
    ProbaLayers
         - probas - probability differences corresponding to the layered sets
         - layers - indicators of classes belonging to layered sets
    """
    n_samples, n_classes = probas.shape[0], probas.shape[CLASSES_DIM]
    original_shape = probas.shape

    # unique is not differentiable therefore in case of gradient requirement
    # we have to just stick to sorting, which will result in more zeros in output mass assignments
    # although it is more vectorized therefore it might still lead to better performance in
    # some cases
    if not probas.requires_grad:
        # in case of more dimensions than 2 add dimensions to samples before unique and then restore appropriate shape
        # heave assumption on CLASSES_DIM == -1
        unique_probas = torch.nn.utils.rnn.pad_sequence(
            [sample.unique() for sample in probas.reshape(-1, n_classes)],
            batch_first=True,
        )
        unique_probas = unique_probas.reshape(*original_shape[:-1], -1)
    else:
        unique_probas = probas.sort(dim=-1).values

    zeros_to_prepend_to_difference = torch.zeros((*unique_probas.shape[:-1], 1))
    proba_difference = torch.diff(
        unique_probas, dim=CLASSES_DIM, prepend=zeros_to_prepend_to_difference
    )
    returned_probas = torch.where(
        proba_difference > 0, proba_difference, torch.zeros_like(proba_difference)
    )

    # we add dim 1, i.e. layers dimension to probas, and classes dimension to unique_probas
    layers = probas.unsqueeze(-2) >= unique_probas.unsqueeze(CLASSES_DIM)

    return ProbaLayers(probas=returned_probas, layers=layers)


class PyramidalEvidence(UncertBase):
    def __init__(
        self,
        aggregation_function: Callable[
            [torch.FloatTensor, torch.BoolTensor], torch.FloatTensor
        ],
        *,
        name: str
    ) -> None:
        super().__init__()
        self.aggregation_function = aggregation_function
        self._name = name

    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        proba_layers = _compute_proba_layers(probas)
        layer_sizes = proba_layers.layers.sum(dim=CLASSES_DIM)
        return self.aggregation_function(
            proba_layers.probas * layer_sizes, proba_layers.layers
        )

    @property
    def __name__(self):
        return self._name


class HeightRatioEvidence(UncertBase):
    def __init__(
        self,
        aggregation_function: Callable[
            [torch.FloatTensor, torch.BoolTensor], torch.FloatTensor
        ],
        *,
        name: str
    ) -> None:
        super().__init__()
        self.aggregation_function = aggregation_function
        self._name = name

    def _call(self, probas: torch.FloatTensor) -> torch.FloatTensor:
        proba_layers = _compute_proba_layers(probas)

        return self.aggregation_function(
            proba_layers.probas / probas.max(dim=CLASSES_DIM, keepdim=True).values,
            proba_layers.layers,
        )

    @property
    def __name__(self):
        return self._name


def exponent_evidence_aggregation(
    proba_masses: torch.FloatTensor, layers: torch.BoolTensor, exponent_base: int = 2
) -> torch.FloatTensor:
    layered_set_size = layers.sum(dim=CLASSES_DIM)
    exponent_base = torch.full_like(
        layered_set_size, fill_value=exponent_base, dtype=torch.float
    )
    return 1 - torch.sum(
        proba_masses * exponent_base ** (-layered_set_size + 1), dim=CLASSES_DIM
    )


def log_plus_evidence_aggregation(
    proba_masses: torch.FloatTensor, layers: torch.BoolTensor
) -> torch.FloatTensor:
    layered_set_size = layers.sum(dim=CLASSES_DIM)
    # we substract one to get 0 value for pure probability
    # as in othe uncertainty functions
    return (
        torch.sum(proba_masses * torch.log2(layered_set_size + 1), dim=CLASSES_DIM) - 1
    )


pyramidal_exponent_evidence = PyramidalEvidence(
    exponent_evidence_aggregation, name="pyramidal_exponent_evidence"
)

pyramidal_large_exponent_evidence = PyramidalEvidence(
    functools.partial(exponent_evidence_aggregation, exponent_base=4),
    name="pyramidal_large_exponent_evidence",
)

pyramidal_log_plus_evidence = PyramidalEvidence(
    log_plus_evidence_aggregation, name="pyramidal_log_plus_evidence"
)

height_ratio_exponent_evidence = HeightRatioEvidence(
    exponent_evidence_aggregation, name="height_ratio_exponent_evidence"
)

height_ratio_large_exponent_evidence = HeightRatioEvidence(
    functools.partial(exponent_evidence_aggregation, exponent_base=4),
    name="height_ratio_large_exponent_evidence",
)

height_ratio_log_plus_evidence = HeightRatioEvidence(
    log_plus_evidence_aggregation, name="height_ratio_log_plus_evidence"
)
