import torch
from torch import FloatTensor

from .base import CLASSES_DIM, UncertClassificationBase
from .classical import entropy


def get_prior_from_model_perspective(probas: FloatTensor) -> FloatTensor:
    return probas.mean(dim=0)


class OffCenteredEntropy(UncertClassificationBase):
    """Entropy tranformed to obtain maximum in point specified by user, e.g. a priori distribution,
    or computed from data.

    Based on P. Lenca, S. Lallich, & B. Vaillant (2010).
    "Construction of an Off-Centered Entropy for the Supervised Learning of Imbalanced Classes: Some First Results."
    Communications in Statistics - Theory and Methods, 39(3), 493-507.

    If `None` is passed(default) as `maximum_loc` the location of maximum value will be computed
    using `get_prior_from_model_perspective` function and indicate mean prediction of the model
    for the available samples.
    """

    def __init__(self, maximum_loc: FloatTensor | None = None) -> None:
        super().__init__()
        self.maximum_loc = maximum_loc

    def _call(self, probas: FloatTensor) -> FloatTensor:
        if self.maximum_loc is None:
            maximum_loc = get_prior_from_model_perspective(probas=probas)
        else:
            maximum_loc = self.maximum_loc

        relocated_distributions = self.relocate_distribuition(
            probas=probas, maximum_loc=maximum_loc
        )
        return entropy._call(relocated_distributions)

    def relocate_distribuition(
        self, probas: FloatTensor, maximum_loc: FloatTensor
    ) -> FloatTensor:
        n_classes = probas.shape[CLASSES_DIM]
        maximum_loc = maximum_loc.unsqueeze(0)
        assert len(probas.shape) == len(maximum_loc.shape)

        values_for_probas_less_than_max = probas / (n_classes * maximum_loc)
        values_for_probas_ge_than_max = (
            n_classes * (probas - maximum_loc) + 1 - probas
        ) / (n_classes * (1 - maximum_loc))

        relocated_values = torch.where(
            probas >= maximum_loc,
            values_for_probas_ge_than_max,
            values_for_probas_less_than_max,
        )

        return torch.nn.functional.normalize(relocated_values, p=1, dim=CLASSES_DIM)


off_centered_entropy = OffCenteredEntropy()
