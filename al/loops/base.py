from __future__ import annotations

import enum
import functools
from typing import Sequence

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torcheval import metrics


def create_BAC_metric(config: LoopConfig):
    assert config.n_classes
    return metrics.MulticlassAccuracy(average="macro", num_classes=config.n_classes)


class LoopMetric(enum.Enum):
    # we are wrapping enum values with partial
    # otherwise it is interpreted as method of the LoopMetric class
    BAC = functools.partial(create_BAC_metric)


class LoopResults(BaseModel):
    metrics: dict[LoopMetric, list[float]] = Field(default_factory=dict)
    pool_probas: Sequence[torch.FloatTensor] = Field(default_factory=list)

    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # consider pydantic typing for torch? or removing pydantic


class LoopConfig(BaseModel):
    metrics: Sequence[LoopMetric] = Field(default_factory=list)
    return_pool_probas: bool = False
    n_classes: int | None = None
    batch_size: int = 1


# Note: only single label classification supported for now
class ALDataset(Dataset):
    _targets: torch.IntTensor | None = None

    def __init__(self, dataset: Dataset) -> None:
        super().__init__()
        self.dataset = dataset

    @property
    def targets(self) -> torch.IntTensor:
        if self._targets is None:
            self._initialize_targets()

        return self._targets

    def _initialize_targets(self):
        targets = []
        if isinstance(self.dataset, TensorDataset):
            self._targets = self.dataset.tensors[1].long()
        else:
            for _, target in self._iterate_over_dataset():
                targets.append(target)

            self._targets = torch.concat(targets).long()

    @property
    def n_classes(self):
        return len(self.targets.unique())

    def _iterate_over_dataset(self):
        loader = DataLoader(self.dataset)
        yield from loader

    def __len__(self):
        return len(self.dataset)
