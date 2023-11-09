from typing import Callable

import torch
from torch.nn.functional import cosine_similarity

from al.sampling.uncert.base import CLASSES_DIM


def numerical_gradient(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    distribution: torch.FloatTensor,
    gradient_step: float = 1e-4,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    """Compute gradients numericaly.

    For given function and probability distribution the method computes gradient approximation
    using differntial quotient.

    Gradients are only returned for points in the distribution for which `distribution +- gradient_step`
    is in range [0, 1].

    Parameters
    ----------
    func : Callable[[torch.FloatTensor], torch.FloatTensor]
        function for which approximation of gradients should be found
    distribution : torch.FloatTensor
        samples from the distribution of shape (n_samples, ..., n_classes)
    gradient_step : float
        step that will be used in gradient approximation computation

    Returns
    -------
    Tuple with:
        - distribution points for which gradients have been computed of shape `(n_samples', ..., n_classes)`
        - gradients of shape `(n_samples', ..., n_classes)`
    """
    n_probas = distribution.shape[CLASSES_DIM]
    d_ = torch.eye(n_probas) * (gradient_step) - torch.full(
        (n_probas, n_probas), fill_value=gradient_step / n_probas
    )
    d_ = d_.unsqueeze(0)

    well_defined_positions = (
        (distribution + gradient_step <= 1).bool()
        & (distribution - gradient_step >= 0).bool()
    ).all(dim=CLASSES_DIM)

    distribution = distribution.unsqueeze(1)
    uncert_values = func(distribution)

    distribution = distribution[well_defined_positions]
    uncert_values = uncert_values[well_defined_positions]

    return distribution.squeeze(dim=1), _difference_quotient(
        func=func, distribution=distribution, func_values_for_dist=uncert_values, d_=d_
    )


def _difference_quotient(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    distribution: torch.Tensor,
    func_values_for_dist: torch.Tensor,
    d_: torch.Tensor,
):
    return (func(distribution + d_) - func_values_for_dist) / (d_**2).sum() ** (0.5)


# gradients cannot be well computed for distributions near simplex border, therefore
# some of the values might be not defined
def torch_gradient(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    distribution: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    distribution_with_grad = distribution.clone().detach().requires_grad_(True)
    uncert_result = func(distribution_with_grad)
    uncert_result.mean().backward()
    return distribution, distribution_with_grad.grad


def prior_descent_ratio(func, distribution, prior=None):
    n_classes = distribution.shape[CLASSES_DIM]
    if prior is None:
        prior = torch.full((1, n_classes), fill_value=1 / n_classes)

    # in case of user passed prior is transposed
    prior = prior.reshape(1, n_classes)

    distribution, gradient = numerical_gradient(func, distribution=distribution)

    return cosine_similarity(gradient, prior - distribution, dim=CLASSES_DIM)


def simplex_vertex_repel_ratio(func, distribution):
    distribution, gradient = numerical_gradient(func, distribution=distribution)
    vertex_position = _get_nearest_vertex_position(distribution)

    return cosine_similarity(-gradient, vertex_position - distribution, dim=CLASSES_DIM)


def _get_nearest_vertex_position(distribution: torch.FloatTensor) -> torch.FloatTensor:
    vertex_position = torch.zeros_like(distribution)
    vertex_position[
        torch.arange(distribution.shape[0]), torch.argmax(distribution, dim=CLASSES_DIM)
    ] = 1
    return vertex_position
