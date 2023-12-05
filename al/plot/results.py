from typing import Sequence

import matplotlib.pyplot as plt
import torch

from al.loops.base import LoopMetricName, LoopResults
from al.loops.experiments import ExperimentResults


def plot_metric(
    experiment_results: ExperimentResults,
    metric_name: LoopMetricName,
    metric_slice: slice = None,
):
    for config_name, config_results in experiment_results.res.items():
        metric_values = config_results.metrics[metric_name]
        if metric_slice is not None:
            metric_values = metric_values[metric_slice]
        metric_avg_over_seeds = metric_values.mean(dim=0)
        iterations = torch.arange(metric_avg_over_seeds.shape[0])

        plt.plot(iterations, metric_avg_over_seeds, label=config_name)

    plt.legend()
