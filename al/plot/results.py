from typing import Sequence

import matplotlib.pyplot as plt
import torch

from al.loops.base import LoopMetricName, LoopResults
from al.loops.experiments import ExperimentResults


def plot_metric(
    experiment_results: ExperimentResults,
    metric_name: LoopMetricName,
    add_mean_to_legened: bool = False,
    metric_slice: slice = None,
):
    for config_name, config_results in experiment_results.res.items():
        metric_values = config_results.metrics[metric_name]
        if metric_slice is not None:
            metric_values = metric_values[metric_slice]
        metric_avg_over_seeds = metric_values.mean(dim=0).cpu()
        iterations = torch.arange(
            metric_avg_over_seeds.shape[0], device=torch.device("cpu")
        )

        label = config_name
        if add_mean_to_legened:
            mean_accross_iters = metric_values.mean(dim=-1)
            label += f" ({mean_accross_iters.mean() : .3f} $\pm$ {torch.std(mean_accross_iters) : .4f})"
        plt.plot(iterations, metric_avg_over_seeds, label=label)

    plt.legend()
