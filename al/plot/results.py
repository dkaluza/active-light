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
    for info_name, loop_result_for_seed in experiment_results.items():
        metric_values = [
            loop_result.metrics[metric_name] for loop_result in loop_result_for_seed
        ]
        metric_values = torch.tensor(metric_values)
        if metric_slice is not None:
            metric_values = metric_values[metric_slice]
        metric_avg_over_seeds = metric_values.mean(dim=0)
        iterations = torch.arange(metric_avg_over_seeds.shape[0])

        plt.plot(iterations, metric_avg_over_seeds, label=info_name)

    plt.legend()
    plt.show()
