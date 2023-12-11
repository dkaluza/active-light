from typing import Callable

import matplotlib.pyplot as plt
import mpltern  # pylint: disable=unused-import
import torch

from al.distributions import uniform_mesh
from al.sampling.uncert.classification.metrics import numerical_gradient, torch_gradient


def plot_uncertainty_function(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    levels: int = 10,
    step: float = 0.01,
    colorbar: bool = True,
):
    """Plot uncertainty function as a heatmap for
    3 class distribution on a ternary plot.

    Parameters
    ----------
    func : Callable[[torch.FloatTensor], torch.FloatTensor]
        Uncertainty function to plot. It should expect exactly one required
        parameter, i.e. probability distribution with shape (n_samples, n_classes).

    levels : int
        Number of levels to include in plot.
    """
    distribution = uniform_mesh(3, step=step)

    uncert_values = func(distribution)

    ax = plt.subplot(projection="ternary")
    ax.set_tlabel("Class 1")
    ax.set_llabel("Class 2")
    ax.set_rlabel("Class 3")

    contour_set = ax.tricontourf(
        distribution[:, 0],
        distribution[:, 1],
        distribution[:, 2],
        uncert_values,
        levels=levels,
    )
    if colorbar:
        plt.colorbar(mappable=contour_set, shrink=0.5)

    # based on
    # https://mpltern.readthedocs.io/en/latest/gallery/axis_and_tick/99.tick_labels_inside_triangle.html#sphx-glr-gallery-axis-and-tick-99-tick-labels-inside-triangle-py
    ax.tick_params(tick1On=False, tick2On=False)

    ax.taxis.set_tick_params(labelrotation=("manual", 0.0))
    ax.laxis.set_tick_params(labelrotation=("manual", -60.0))
    ax.raxis.set_tick_params(labelrotation=("manual", 60.0))

    kwargs = {"y": 0.5, "ha": "center", "va": "center", "rotation_mode": "anchor"}
    tkwargs = {"transform": ax.get_taxis_transform()}
    lkwargs = {"transform": ax.get_laxis_transform()}
    rkwargs = {"transform": ax.get_raxis_transform()}
    tkwargs.update(kwargs)
    lkwargs.update(kwargs)
    rkwargs.update(kwargs)
    [text.update(tkwargs) for text in ax.taxis.get_ticklabels()]
    [text.update(lkwargs) for text in ax.laxis.get_ticklabels()]
    [text.update(rkwargs) for text in ax.raxis.get_ticklabels()]

    return ax


def plot_uncertainty_function_with_gradients(
    func: Callable[[torch.FloatTensor], torch.FloatTensor],
    levels: int = 10,
    step: float = 0.05,
    colorbar: bool = True,
    gradient_colorbar: bool = False,
    gradient_color_norm=None,
):
    ax = plot_uncertainty_function(func=func, levels=levels, colorbar=colorbar)
    distribution = uniform_mesh(3, step=step)

    distribution, gradients = numerical_gradient(func=func, distribution=distribution)
    gradient_dt, gradient_dl, gradient_dr = torch.unbind(gradients, dim=1)

    length = torch.sqrt(gradient_dt**2 + gradient_dl**2 + gradient_dr**2)

    t, l, r = distribution[:, 0], distribution[:, 1], distribution[:, 2]

    gradient_sum = gradient_dt + gradient_dl + gradient_dr
    # if gradients do not add up to 0 then the function is in this place undifferentiable
    # therefore we remove those gradient arrows and add appropriate x marks
    undifferentiable = gradient_sum.abs() > 1e-1

    # we add white x to points that are undifferentiable
    ax.scatter(
        t[undifferentiable],
        l[undifferentiable],
        r[undifferentiable],
        c="white",
        marker="x",
    )

    contour_set = ax.quiver(
        t[~undifferentiable],
        l[~undifferentiable],
        r[~undifferentiable],
        gradient_dt[~undifferentiable],
        gradient_dl[~undifferentiable],
        gradient_dr[~undifferentiable],
        length[~undifferentiable],
        cmap="inferno",
        norm=gradient_color_norm,
    )
    if gradient_colorbar:
        plt.colorbar(mappable=contour_set, location="bottom")

    return ax
