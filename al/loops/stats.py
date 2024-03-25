from collections import defaultdict
from typing import Sequence

import torch
from scipy.stats import friedmanchisquare

from al.loops.base import LoopMetricName
from al.loops.experiments import ExperimentResults

# TODO: check dependencies if they are in pyproject toml


def friedman_difference_test(
    results: Sequence[ExperimentResults], metric_name: LoopMetricName
) -> float:
    """Computes Friedman test to detect statstical diferences between
    results for indicated metric.

    The test is performed taking into account all configurations available in the results.
    Results may come from experiments spanning conducted on multiple datasets.
    In such case each repetition of experiment is taken into account
    as a separate block in Friedman test.
    Therefore, each experiment should have the same number of repetitions
    if the same weight for each dataset is desired.

    This function assumes all of the experiments were performed with exactly
    the same set of configurations.

    If metric with mutliple measurements is passed, e.g. BAC for each iteration,
    then test is performed on average across all of the iterations.

    Parameters
    ----------
    results : list[ExperimentResults]
        Experiment results for which test should be computed.
        Each experiment should contain exactly the same configurations.
    Returns
    -------
    float
        P-value of the one sided Friedman test.
    """

    results_for_configs = defaultdict(list)

    for result in results:
        for config_name, config_values in result.res.items():
            metric_values = config_values.metrics[metric_name]

            # average accross multiple measurements, e.g. iterations
            # if metric is defined for each iteration
            metric_values = metric_values.reshape((metric_values.shape[0], -1))
            metric_values = metric_values.mean(1)

            results_for_configs[config_name].append(metric_values)

    results_for_configs = {
        config_name: torch.concat(config_value_list)
        for config_name, config_value_list in results_for_configs.items()
    }

    test_results = friedmanchisquare(*results_for_configs.values())
    return test_results.pvalue


def 