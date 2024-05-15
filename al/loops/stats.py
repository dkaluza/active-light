from collections import defaultdict
from typing import Iterable, NamedTuple, Sequence

import scipy.stats
import torch
from scipy.stats import false_discovery_control, friedmanchisquare, wilcoxon

from al.loops.base import LoopMetricName
from al.loops.experiments import ExperimentResults

# TODO: check dependencies if they are in pyproject toml


def friedman_difference_test(
    results: Sequence[ExperimentResults], metric_name: LoopMetricName
) -> float:
    """Computes non-parametric Friedman ranks test to detect statstical diferences between
    results for indicated metric.

    For more information about the test see:
     `Friedman test on Wikipedia<https://en.wikipedia.org/wiki/Friedman_test>`_.

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
        P-value of the Friedman test indicating probability of insignificant
        difference beetwen tested methods.
    """
    results_for_configs = _get_results_for_configs_across_exp_reps(
        results=results, metric_name=metric_name
    )
    test_results = friedmanchisquare(*results_for_configs.values())
    return test_results.pvalue


def _get_results_for_configs_across_exp_reps(
    results: Sequence[ExperimentResults], metric_name: LoopMetricName
) -> dict[str, torch.FloatTensor]:
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
    return results_for_configs


class PairwiseTestResults(NamedTuple):
    pvalues: torch.FloatTensor
    """Tensor with p-values obtained from pairwise testing each 2 configurations.
    Only upper triangular part of the matrix just above the diagonal have to be filled,
    as other values will result in multiple times testing the same combination of configurations
    or testing method against itself. i-th row j-th column indicate the p-value on the test
    with i-th configuration used as a first argument and j-th configuration used as the second
    argument of the appropriate test.
    """
    config_names: list[str]
    """Configuration names correponding to rows and columns of the pvalues tensor."""


def pairwise_wilcoxon_test(
    results: Sequence[ExperimentResults],
    metric_name: LoopMetricName,
    configs_to_test: str | None = None,
    alternative: str = "two-sided",
    pvalue_adjustment_method: str = "bh",
    include_self_comparison: bool = True,
) -> PairwiseTestResults:
    # TODO: docstring, todo test if proper only values are filled
    # TODO: test that both triu and selected configs work
    results_for_configs = _get_results_for_configs_across_exp_reps(
        results=results, metric_name=metric_name
    )
    config_names = list(results_for_configs.keys())
    config_values = list(results_for_configs.values())

    n_configs = len(config_names)
    pvalues = torch.full((n_configs, n_configs), fill_value=torch.nan)

    if configs_to_test is None:
        indices = torch.triu_indices(n_configs, n_configs, offset=1)
    else:
        configs_indices = [
            i for i, name in enumerate(config_names) if name in configs_to_test
        ]

        if include_self_comparison:
            indices = [
                [i, j] for i in configs_indices for j in range(n_configs) if i != j
            ]
        else:
            indices = [
                [i, j]
                for i in configs_indices
                for j in range(n_configs)
                if j not in configs_indices
            ]

        indices = torch.tensor(indices).transpose(0, 1)

    for i, j in zip(*indices):
        pvalues[i, j] = wilcoxon(
            config_values[i], config_values[j], alternative=alternative
        ).pvalue

    adjusted_pvalues = false_discovery_control(
        pvalues[*indices], method=pvalue_adjustment_method
    )
    pvalues[*indices] = torch.from_numpy(adjusted_pvalues)
    return PairwiseTestResults(pvalues=pvalues, config_names=config_names)


def get_ranks_for_experiment(
    result: ExperimentResults, metric_name: LoopMetricName, desc: bool = True
) -> torch.FloatTensor:
    # TODO: docs & test
    config_results = []
    for config_name, config_values in result.res.items():
        metric_values = config_values.metrics[metric_name]
        if desc:
            metric_values = -metric_values

        # average accross both multiple measurements and n repetitions of the experiment
        metric_values = metric_values.reshape((metric_values.shape[0], -1))
        average_config_result = metric_values.mean(1).mean(0)

        config_results.append(average_config_result.item())

    config_results = scipy.stats.rankdata(config_results)
    return torch.from_numpy(config_results)
