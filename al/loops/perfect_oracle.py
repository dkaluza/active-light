from typing import Callable, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset
from tqdm.auto import tqdm

from al.base import ModelProto
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
    results = LoopResults()
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

    with tqdm(total=budget, leave=None) as progress_bar:
        while used_budget < budget:
            batch_size = min(config.batch_size, budget - used_budget)
            info_values = info_func(model=model, dataset=pool)
            selected_samples_idx = torch.topk(info_values, k=batch_size, dim=0).indices
            selected_samples_idx = selected_samples_idx.reshape(-1)

            train = ALDataset(
                ConcatDataset([train, Subset(pool, selected_samples_idx)])
            )

            pool = ALDataset(remove_indices_from_dataset(pool, selected_samples_idx))

            model.fit(train=train)
            with torch.no_grad():
                add_next_metrics_evaluation(
                    results=results, model=model, test=test, config=config
                )
                add_pool_probas(results=results, model=model, pool=pool, config=config)

            used_budget += batch_size
            progress_bar.update(batch_size)

    return results


def remove_indices_from_dataset(dataset: Dataset, indices: list[int]) -> Dataset:
    remaining_idx = torch.arange(len(dataset), dtype=torch.long)

    remaining_idx_mask = torch.full_like(
        remaining_idx, fill_value=True, dtype=torch.bool
    )

    remaining_idx_mask[indices] = False
    remaining_idx = remaining_idx[remaining_idx_mask]

    return Subset(dataset, remaining_idx)


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

        results.metrics.setdefault(metric.name, []).append(score)


def add_pool_probas(
    results: LoopResults,
    model: ModelProto,
    pool: Dataset,
    config: LoopConfig,
):
    if config.return_pool_probas:
        probas = model.predict_proba(pool)
        results.pool_probas.append(probas)
