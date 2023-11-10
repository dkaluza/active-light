from typing import Callable, Sequence

import torch
from torch.utils.data import Dataset, random_split
from tqdm import trange

from al.loops.base import LoopResults
from al.sampling.base import InformativenessProto


def run_experiment(
    loop: Callable[..., LoopResults],
    *,
    data: Dataset,
    infos: Sequence[InformativenessProto],
    init_frac: float | int = 0.001,
    test_frac: float | int = 0.5,
    n_repeats: int = 10,
    seed: int = 42,
    **loop_kwargs
) -> dict[str, Sequence[LoopResults]]:
    assert type(init_frac) == type(
        test_frac
    ), "Init and test fraction have to have the same type int or float"
    generator = torch.Generator().manual_seed(seed)

    if isinstance(test_frac, float):
        pool_frac = 1 - init_frac - test_frac
    else:
        pool_frac = len(data) - init_frac - test_frac

    results: dict[str, list[LoopResults]] = dict()

    for _ in trange(n_repeats, desc="Seeds"):
        initial_train, pool, test = random_split(
            data, (init_frac, pool_frac, test_frac), generator=generator
        )

        for info in infos:
            loop_result = loop(
                initial_train=initial_train,
                pool=pool,
                test=test,
                info_func=info,
                **loop_kwargs
            )
            results.setdefault(info.__name__, []).append(loop_result)

    return results
