from __future__ import annotations

import enum
import functools
from typing import Annotated, Any, Iterable, Self, Sequence, TypeAlias

import torch
from pydantic import BaseModel, Field, GetPydanticSchema
from torch._tensor import Tensor
from torch.utils.data import DataLoader, Dataset, IterableDataset, TensorDataset
from torcheval import metrics
from torcheval.metrics.metric import Metric


class MetricWrapper(Metric[torch.Tensor]):
    def __init__(
        self: Self,
        *,
        metric: Metric,
        is_distribution_based: bool,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(device=device)
        self.metric = metric.to(device=device)
        self.is_distribution_based = is_distribution_based

    def update(self: Self, *_: Any, **__: Any) -> Self:
        self.metric.update(*_, **__)
        return self

    def compute(self: Self) -> Tensor:
        return self.metric.compute()

    def merge_state(self: Self, metrics: Iterable[Self]) -> Self:
        self.metric.merge_state(metrics)
        return self


def create_BAC_metric(config: LoopConfig) -> MetricWrapper:
    assert config.n_classes
    metric = MetricWrapper(
        metric=metrics.MulticlassAccuracy(
            average="macro", num_classes=config.n_classes
        ),
        is_distribution_based=True,
    )
    return metric


def create_R2_metric(config: LoopConfig) -> MetricWrapper:
    metric = MetricWrapper(
        metric=metrics.R2Score(),
        is_distribution_based=False,
    )
    return metric


class LoopMetric(enum.Enum):
    # we are wrapping enum values with partial
    # otherwise it is interpreted as method of the LoopMetric class
    BAC = functools.partial(create_BAC_metric)
    R2 = functools.partial(create_R2_metric)


LoopMetricName: TypeAlias = str

HandleAsAny = GetPydanticSchema(lambda _s, handler: handler(Any))

FloatTensor: TypeAlias = Annotated[torch.FloatTensor, HandleAsAny]


class LoopResults(BaseModel):
    metrics: dict[LoopMetricName, list[float]] | None = None
    """
    Metrics obtained after sampling consecutive batches in the loop.
        
    Keys correspond to `LoopMetric.name` of the requested metrics,
    values correspond to metric values obtained after each iteration.
    Value at the index 0 correspond to metric obtained before choosing
    any elements in the loop.
    """
    pool_probas: Sequence[FloatTensor] | None = None

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, LoopResults):
            return False

        return (
            self.metrics == other.metrics
            and len(self.pool_probas) == len(other.pool_probas)
            and all(
                [
                    (probas_self == probas_other).all()
                    for probas_self, probas_other in zip(
                        self.pool_probas, other.pool_probas
                    )
                ]
            )
        )

    def initialize_from_config(self, config: LoopConfig):
        if len(config.metrics) > 0:
            self.metrics = {}
        if config.return_pool_probas:
            self.pool_probas = []


class LoopConfig(BaseModel):
    metrics: Sequence[LoopMetric] = Field(default_factory=list)
    return_pool_probas: bool = False
    n_classes: int | None = None
    batch_size: int = 1


# we avoid wrapping dataset multiple times by using a metaclass
# to return the same object, otherwise it leads to slow down of
# performance because of cache misses
class TransparentWrapperMeta(type):
    def __call__(cls, *args, **kwargs):
        if isinstance(args[0], cls):
            return args[0]

        self = cls.__new__(cls, *args, **kwargs)
        cls.__init__(self, *args, **kwargs)
        return self


class ALDatasetWithoutTargets(Dataset, metaclass=TransparentWrapperMeta):
    _features: torch.FloatTensor | None = None

    def __init__(self, dataset: Dataset, /) -> None:
        super().__init__()
        self.dataset = dataset

    @property
    def features(self) -> torch.FloatTensor:
        if self._features is None:
            self._initialize_features()

        return self._features

    def _initialize_features(self):
        features = []
        if isinstance(self.dataset, TensorDataset):
            self._features = self.dataset.tensors[0].float()
        else:
            for batch in self._iterate_over_dataset():
                feature_batch = batch[0]
                features.append(feature_batch)

            if len(features) > 0:
                self._features = torch.concat(features).float()
            else:
                self._features = torch.empty((0, 1), dtype=torch.float)

    def __hash__(self) -> int:
        return self.features.__hash__()

    def _iterate_over_dataset(self):
        loader = DataLoader(self.dataset, shuffle=False)
        yield from loader

    def __getitem__(self, index):
        return self.dataset.__getitem__(index)

    def __len__(self):
        return len(self.dataset)

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ALDatasetWithoutTargets):
            return False

        return torch.all(self.features == __value.features)

    def __repr__(self) -> str:
        return f"ALDataset features: {self.features}"


# Note: only single label classification supported for now
class ALDataset(ALDatasetWithoutTargets):
    _targets: torch.Tensor | None = None
    _targets_dtype: torch.dtype

    def __init__(self, dataset: Dataset, targets_dtype=None) -> None:
        super().__init__(dataset)
        self._targets_dtype = targets_dtype

    @property
    def targets(self) -> torch.IntTensor:
        if self._targets is None:
            self._initialize_targets()

        return self._targets

    def _initialize_targets(self):
        targets = []
        if isinstance(self.dataset, TensorDataset):
            self._targets = self.dataset.tensors[1].to(dtype=self._targets_dtype)
        else:
            for batch in self._iterate_over_dataset():
                target_batch = batch[1]
                targets.append(target_batch)

            if len(targets) > 0:
                self._targets = torch.concat(targets).to(dtype=self._targets_dtype)
            else:
                self._targets = torch.empty((0,), dtype=self._targets_dtype)

    @property
    def n_classes(self):
        return len(self.targets.unique())

    def __eq__(self, __value: object) -> bool:
        if not isinstance(__value, ALDataset):
            return False

        return torch.all(self.targets == __value.targets) and super().__eq__(__value)

    def __hash__(self) -> int:
        return self.features.__hash__() + self.targets.__hash__()

    def __repr__(self) -> str:
        return super().__repr__() + f"\nALDataset targets: {self.targets}"
