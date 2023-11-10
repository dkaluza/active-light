from typing import Callable, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm import tqdm

from al.base import ModelProto
from al.loops.base import ALDataset, LoopConfig, LoopMetric, LoopResults
from al.sampling.base import InformativenessProto


def active_learning_loop(
    initial_train: Dataset,
    pool: Dataset,
    test: Dataset,
    info_func: InformativenessProto,
    budget: float | int,
    model: ModelProto,
    config: LoopConfig,
) -> LoopResults:
    results = LoopResults()
    test = ALDataset(test)
    used_budget = 0

    if config.n_classes is None:
        config.n_classes = test.n_classes

    if isinstance(budget, float):
        budget = int(budget * len(pool))

    train = initial_train
    model.fit(train=train)
    with torch.no_grad():
        add_next_metrics_evaluation(
            results=results, model=model, test=test, config=config
        )
        add_pool_probas(results=results, model=model, pool=pool, config=config)

    with tqdm(total=budget) as progress_bar:
        while used_budget < budget:
            batch_size = min(config.batch_size, budget - used_budget)
            info_values = info_func(model=model, pool=pool)
            selected_samples_idx = torch.topk(info_values, k=batch_size, dim=0).indices

            train = ConcatDataset([train, Subset(pool, selected_samples_idx)])

            remaining_idx = torch.arange(len(pool), dtype=torch.long)
            remaining_idx = remaining_idx[~selected_samples_idx]
            pool = Subset(pool, remaining_idx)

            model.fit(train=train)
            with torch.no_grad():
                add_next_metrics_evaluation(
                    results=results, model=model, test=test, config=config
                )
                add_pool_probas(results=results, model=model, pool=pool, config=config)

            used_budget += batch_size
            progress_bar.update(batch_size)

    return results


def add_next_metrics_evaluation(
    results: LoopResults,
    model: ModelProto,
    test: ALDataset,
    config: LoopConfig,
):
    probas = model.predict_proba(test)

    for metric in config.metrics:
        metric_fun = metric.value(config)
        metric_fun.update(probas, test.targets)
        score = metric_fun.compute()

        results.metrics.setdefault(metric, []).append(score)


def add_pool_probas(
    results: LoopResults,
    model: ModelProto,
    pool: Dataset,
    config: LoopConfig,
):
    if config.return_pool_probas:
        probas = model.predict_proba(pool)
        results.pool_probas.append(probas)
