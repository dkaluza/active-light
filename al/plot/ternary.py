from typing import Callable

import matplotlib.pyplot as plt
import mpltern  # pylint: disable=unused-import
import torch

from al.distributions import uniform_mesh


def plot_uncertainty_function(
    func: Callable[[torch.FloatTensor], torch.FloatTensor], levels: int = 10
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
    step = 0.01
    distribution = uniform_mesh(3, step=step)

    uncert_values = func(distribution)

    ax = plt.subplot(projection="ternary")
    ax.set_tlabel("Class 1")
    ax.set_llabel("Class 2")
    ax.set_rlabel("Class 3")

    ax.tricontourf(
        distribution[:, 0],
        distribution[:, 1],
        distribution[:, 2],
        uncert_values,
        levels=levels,
    )
    plt.colorbar()
