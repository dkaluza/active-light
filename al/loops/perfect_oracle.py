import functools
import time

import torch
from tqdm.auto import tqdm

from al.base import ActiveInMemoryState, ActiveState, ModelProto
from al.loops.base import ALDataset, LoopConfig, LoopMetric, LoopResults
from al.sampling.base import InformativenessProto


def active_learning_loop(
    initial_train: ALDataset,
    pool: ALDataset,
    test: ALDataset,
    info_func: InformativenessProto,
    budget: float | int,
    model: ModelProto,
    config: LoopConfig,
) -> LoopResults:
    # TODO: refactor to make it reusable with other loops
    results = LoopResults()
    results.initialize_from_config(config=config)
    used_budget = 0
    state = ActiveInMemoryState(model=model, pool=pool, training_data=initial_train)

    if config.n_classes is None:
        config.n_classes = test.n_classes

    if isinstance(budget, float):
        budget = int(budget * len(pool))

    state.refit_model()
    with torch.no_grad():
        add_next_metrics_evaluation(
            results=results, state=state, test=test, config=config
        )
        add_pool_probas(results=results, state=state, config=config)

    with tqdm(total=budget, leave=None) as progress_bar:
        while used_budget < budget:
            batch_size = min(config.batch_size, budget - used_budget)

            info_start_time = time.perf_counter()
            info_values = info_func(state=state)
            info_end_time = time.perf_counter()

            if config.return_info_times:
                results.info_times.append(info_end_time - info_start_time)

            selected_samples_idx = torch.topk(info_values, k=batch_size, dim=0).indices
            selected_samples_idx = selected_samples_idx.reshape(-1)

            state.select_samples(pool_idx=selected_samples_idx, remove_from_pool=True)

            state.refit_model()
            with torch.no_grad():
                add_next_metrics_evaluation(
                    results=results, state=state, test=test, config=config
                )
                add_pool_probas(results=results, state=state, config=config)

            used_budget += batch_size
            progress_bar.update(batch_size)

    print(results)
    return results


def add_next_metrics_evaluation(
    results: LoopResults,
    state: ActiveState,
    test: ALDataset,
    config: LoopConfig,
):
    model: ModelProto = state.get_model()
    metrics = [metric.value(config) for metric in config.metrics]
    metric_names = [metric.name for metric in config.metrics]

    is_any_metric_proba_based = functools.reduce(
        lambda val, metric: val or metric.is_distribution_based, metrics, False
    )
    is_any_metric_not_proba_based = functools.reduce(
        lambda val, metric: val or not metric.is_distribution_based, metrics, False
    )

    probas = model.predict_proba(test) if is_any_metric_proba_based else None
    preds = model.predict(test) if is_any_metric_not_proba_based else None

    for metric_name, metric_fun in zip(metric_names, metrics):
        metric_input = probas if metric_fun.is_distribution_based else preds
        metric_fun.update(metric_input, test.targets)
        score = metric_fun.compute()
        results.metrics.setdefault(metric_name, []).append(score)


def add_pool_probas(
    results: LoopResults,
    state: ActiveState,
    config: LoopConfig,
):
    if config.return_pool_probas:
        results.pool_probas.append(state.get_probas())
