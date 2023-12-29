import functools
from itertools import chain, combinations
from logging import warning
from typing import Callable, NamedTuple

import torch

from al.distances import Distance


class BinningResult(NamedTuple):
    grid_weights: torch.FloatTensor
    min_coords: torch.FloatTensor
    max_coords: torch.FloatTensor
    step_sizes: torch.FloatTensor


def linear_binning(
    points: torch.FloatTensor,
    distance: Distance,
    n_grid_points=1024,
    min_coords: torch.FloatTensor | None = None,
    max_coords: torch.FloatTensor | None = None,
) -> BinningResult:
    """
    TODO: Better docstring
    This algorithm has O(n*2^d) time complexity, where n is the number of samples and d is the number of features.
    Therefore is suitable only for data sets with small features dimension.

    points : torch.FloatTensor
        Data points of shape (n_samples, n_features) that should be splitted
        into equidistant bins.
    distance : Distance
        Distance function to use in the computation.
    n_grid_points : int, optional
        Number of grid points to use in the computation, by default 1024.
        The grid will be a mesh of n_grid_points**(1/n_features) equidistant points in each direction.
        Therefore more features will usually require a greater number of points and
        will lead to larger memory and time consuption.
    min_coords : torch.FloatTensor | None, optional
        Coordinates of the smallest point in the grid, by default None.
        In case of None min of passed points will be taken.
    max_coords : torch.FloatTensor | None, optional
        Coordinates of the largest point in the grid, by default None.
        In case of None max of passed points will be taken.

    """
    n_dims = points.shape[-1]
    n_points_per_dim = int(n_grid_points ** (1 / n_grid_points))
    if n_points_per_dim < 10:
        warning("Low number of points for each dimension %d", n_points_per_dim)

    if min_coords is None:
        min_coords = points.min(dim=0)
    if max_coords is None:
        max_coords = points.max(dim=0)

    # we divide by n_points_per_dim - 1 because both start and end should be included in the n_points
    step_sizes = (max_coords - min_coords) / (n_points_per_dim - 1)

    grid_shape = [n_points_per_dim] * n_dims
    grid_weights = torch.zeros(grid_shape)

    # mesh grid cannot be equidistant in arbitrary metric, therefore it is equidistant in l1,
    # the same as a result of linspace, but point weights will be computed with defined
    # metric to all of the neighboring points

    step_sizes = step_sizes.unsqueeze(0)
    add_to_weights = functools.partial(
        _add_weights_indexing_by_bounds, grid_weights=grid_weights
    )
    _apply_function_for_weights_for_bounds(
        points=points,
        distance=distance,
        min_coords=min_coords,
        step_sizes=step_sizes,
        func=add_to_weights,
    )
    print(grid_weights)
    return BinningResult(
        grid_weights=grid_weights,
        min_coords=min_coords,
        max_coords=max_coords,
        step_sizes=step_sizes,
    )


def _add_weights_indexing_by_bounds(grid_weights, shifted_bounds, weights_for_bounds):
    grid_weights[
        *torch.split(shifted_bounds, split_size_or_sections=1, dim=1)
    ] += weights_for_bounds


def _apply_function_for_weights_for_bounds(
    points: torch.FloatTensor,
    distance: Distance,
    min_coords: torch.FloatTensor,
    step_sizes: torch.FloatTensor,
    func: Callable[[torch.Tensor, torch.Tensor]],
):
    n_dims = points.shape[-1]
    lower_bounds = torch.div(points - min_coords, step_sizes, rounding_mode="floor")

    lower_bounds_coords = lower_bounds * step_sizes + min_coords

    # powerset of possible different coords shifts, as in itertools docs
    possible_dim_idx = list(range(n_dims))
    powerset_of_dim_idx_shifts = chain.from_iterable(
        combinations(possible_dim_idx, r) for r in range(len(possible_dim_idx) + 1)
    )
    for idxs_to_shift in powerset_of_dim_idx_shifts:
        bound_shift = torch.zeros_like(lower_bounds)
        bound_shift[:, idxs_to_shift] = 1

        coords_shift = torch.zeros_like(lower_bounds_coords)
        coords_shift[:, idxs_to_shift] = step_sizes

        shifted_bounds = lower_bounds + bound_shift
        shifted_coords = lower_bounds_coords + coords_shift
        weights_for_bounds = distance.pairwise(
            points, shifted_coords
        ) / distance.pairwise(
            lower_bounds_coords, shifted_coords
        )  # Note: rethink the normalization
        func(shifted_bounds=shifted_bounds, weights_for_bounds=weights_for_bounds)


def approximate_point_densities_from_binned_density(
    grid_density: torch.FloatTensor,
    points: torch.FloatTensor,
    distance: Distance,
    min_coords: torch.FloatTensor,
    step_sizes: torch.FloatTensor,
):
    points_densities = torch.zeros(points.shape[0])
    func = functools.partial(
        _approximate_point_densities_by_weighted_average,
        points_densities=points_densities,
        grid_density=grid_density,
    )
    _apply_function_for_weights_for_bounds(
        points=points,
        distance=distance,
        min_coords=min_coords,
        step_sizes=step_sizes,
        func=func,
    )

    return points_densities


def _approximate_point_densities_by_weighted_average(
    points_densities, grid_densities, shifted_bounds, weights_for_bounds
):
    grid_indexer = torch.split(shifted_bounds, split_size_or_sections=1, dim=1)
    points_densities[:] += (
        grid_densities[*grid_indexer] * weights_for_bounds[*grid_indexer]
    )
