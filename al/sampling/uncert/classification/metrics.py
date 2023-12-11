from typing import Callable

import torch
from torch.nn.functional import cosine_similarity

from al.sampling.uncert.classification.base import CLASSES_DIM

# TODO refactor gradients to new file


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

    # dist shape (n_samples, ..., n_classes)
    distribution = distribution.unsqueeze(1)
    # dist shape (n_samples, 1, ..., n_classes)
    uncert_values = func(distribution)
    # uncert_values shape  (n_samples, 1, ...)
    # the additional 1 size is needed to make difeerence quotient over all directions

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
) -> torch.FloatTensor:
    """Difference quotient before functiion values and function applied to
    ditribution moved by a small step.

    Parameters
    ----------
    func : Callable[[torch.FloatTensor], torch.FloatTensor]
        Function to apply to moved distribution.
    distribution : torch.Tensor
        Distribution to move, it should be of shape `(n_samples, 1, ..., n_classes)`.
    func_values_for_dist : torch.Tensor
        Precomputed values of function fo the distribution with shape `(n_samples, 1, ...)`
    d_ : torch.Tensor
        Small step directions in which distribution should be moved before applying the
        function. Usually they should have shape `(1, n_directions, ..., n_classes)`, although they can be also
        of shape `(n_samples, n_directions, ..., n_classes)` to have different directions applied for each sample.

    Returns
    -------
    torch.FloatTensor
        Difference quotient values obtained in each direction.
        The shape of the tensor will be `(n_samples, n_directions, ...)`
    """
    if distribution.shape[0] == 0:
        return torch.empty_like(func_values_for_dist)
    return (func(distribution + d_) - func_values_for_dist) / torch.linalg.vector_norm(
        d_, dim=CLASSES_DIM
    )


# gradients cannot be well computed for distributions near simplex border, therefore
# some of the values might be not defined
def torch_gradient(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    distribution: torch.FloatTensor,
) -> tuple[torch.FloatTensor, torch.FloatTensor]:
    distribution_with_grad = distribution.clone().detach_().requires_grad_(True)
    uncert_result = func(distribution_with_grad)
    uncert_result.sum().backward()
    grad = distribution_with_grad.grad

    if (
        grad is None
    ):  # may occur if uncert is independent from distribution, e.g. random
        grad = torch.zeros_like(distribution_with_grad)
    return distribution_with_grad.detach_(), grad


def prior_descent_ratio(func, distribution, prior=None):
    n_classes = distribution.shape[CLASSES_DIM]
    if prior is None:
        prior = torch.full((1, n_classes), fill_value=1 / n_classes)

    # in case of user passed prior is transposed
    prior = prior.reshape(1, n_classes)

    distribution, gradient = torch_gradient(func, distribution=distribution)

    return cosine_similarity(gradient, prior - distribution, dim=CLASSES_DIM)


def uncert_maximum_descent_ratio(func, distribution):
    n_classes = distribution.shape[CLASSES_DIM]

    original_shape = distribution.shape
    distribution = distribution.reshape(-1, n_classes)

    distribution, gradient = torch_gradient(func, distribution=distribution)
    uncertainty_max_points_shape = (distribution.shape[0], n_classes - 1, n_classes)
    top_k_classes = torch.arange(n_classes, 1, -1)
    uncertainty_dist_values = 1 / top_k_classes
    uncertainty_dist_values = (
        uncertainty_dist_values.unsqueeze(0)
        .unsqueeze(-1)
        .expand(uncertainty_max_points_shape)
    )
    # we take upper triangular part to have probability divided to decreasing number
    # of classes in every row
    # the i-th row should have probability divided to n-classes - i
    uncertainty_dist_values = torch.triu(uncertainty_dist_values)

    # descending sort will match the upper triangular with values
    dist_sort = torch.sort(distribution, dim=CLASSES_DIM, descending=False)
    order = dist_sort.indices
    ordered_dist = dist_sort.values

    # top k values become an average of top k values in the distribution
    uncertainty_top_values = ordered_dist.unsqueeze(1) * uncertainty_dist_values
    uncertainty_top_values = torch.sum(
        uncertainty_top_values, dim=CLASSES_DIM, keepdim=True
    )
    uncertainty_top_values = torch.triu(
        uncertainty_top_values.expand(uncertainty_max_points_shape)
    )

    # we fill the rest values(not the top k) with uniform distribution of remaining proba
    rest_values = (
        1 - torch.sum(uncertainty_top_values, dim=CLASSES_DIM, keepdim=True)
    ) / (n_classes - top_k_classes).unsqueeze(0).unsqueeze(-1)

    rest_values = torch.tril(
        rest_values.expand(uncertainty_max_points_shape), diagonal=-1
    )
    uncertainty_max_points_sorted = rest_values + uncertainty_top_values

    # as our uncertainty_max_points_sorted are sorted and gradient and distribution are not
    # we have to apply the same ordering to them
    reuslts = cosine_similarity(
        torch.take_along_dim(gradient, order, dim=CLASSES_DIM).unsqueeze(1),
        uncertainty_max_points_sorted
        - torch.take_along_dim(distribution, order, dim=CLASSES_DIM).unsqueeze(1),
        dim=CLASSES_DIM,
    )

    return reuslts.reshape([*original_shape[:-1], n_classes - 1])


def max_uncert_maximum_descent_ratio(func, distribution):
    results = uncert_maximum_descent_ratio(func, distribution)
    return results.max(dim=-1).values


def max_uncert(func, distribution):
    results = func(distribution)
    return results


def sum_uncert_maximum_descent_ratio(func, distribution):
    results = uncert_maximum_descent_ratio(func, distribution)
    return results.sum(dim=-1)


def simplex_vertex_repel_ratio(func, distribution):
    distribution, gradient = torch_gradient(func, distribution=distribution)
    vertex_position = _get_nearest_vertex_position(distribution)
    return cosine_similarity(-gradient, vertex_position - distribution, dim=CLASSES_DIM)


def monotonicity_from_vertex(func, distribution, derivative_step: float = 1e-4):
    vertex_position = _get_nearest_vertex_position(distribution)
    func_values_for_dist = func(distribution)
    direction_to_nearest_vertex = vertex_position - distribution
    direction_to_nearest_vertex = (
        direction_to_nearest_vertex
        / torch.linalg.vector_norm(
            direction_to_nearest_vertex, dim=CLASSES_DIM, keepdim=True
        )
    )
    derivative_approx = _difference_quotient(
        func=func,
        distribution=distribution,
        func_values_for_dist=func_values_for_dist,
        d_=direction_to_nearest_vertex * derivative_step,
    )  # approximation of change of func when we are getting nearer the vertex

    return -derivative_approx


def _get_nearest_vertex_position(distribution: torch.FloatTensor) -> torch.FloatTensor:
    vertex_position = torch.zeros_like(distribution)
    vertex_position.scatter_(
        dim=CLASSES_DIM,
        index=torch.argmax(distribution, dim=CLASSES_DIM, keepdim=True),
        src=torch.ones_like(vertex_position),
    )
    return vertex_position


def get_uncert_quntile_order(
    func, distribution: torch.FloatTensor
) -> torch.FloatTensor:
    n_classes = distribution.shape[CLASSES_DIM]
    n_values = 1_000_000

    probas = torch.rand((n_values, n_classes))
    probas = torch.nn.functional.normalize(probas, p=1, dim=1)

    searched_uncerts = func(distribution)
    uncert_values = func(probas)
    sorted_uncerts = torch.sort(uncert_values).values

    indices = torch.searchsorted(sorted_uncerts, searched_uncerts)
    qunatiles_order = indices / n_values
    return qunatiles_order
