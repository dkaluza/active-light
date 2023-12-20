from __future__ import annotations

import enum
import functools
from typing import Annotated, Any, Iterable, Self, Sequence, TypeAlias

import torch
from pydantic import BaseModel, Field, GetPydanticSchema
from torch._tensor import Tensor
from torch.utils.data import ConcatDataset, DataLoader, Dataset, Subset, TensorDataset
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
        optimized_initialize_succeeded = (
            self._optimized_initialize_features_for_tensor_types()
        )

        if not optimized_initialize_succeeded:
            self._initialize_features_by_iteration()

    def _optimized_initialize_features_for_tensor_types(self):
        features = self._optimized_retrieve_for_tensor_types(
            considered_dataset=self.dataset,
            index_to_retrieve=0,
            attr_to_retrieve="features",
            required_class_to_select_by_attr=ALDatasetWithoutTargets,
        )
        if features is not None:
            self._features = features
            return True

        return False

    def _optimized_retrieve_for_tensor_types(
        self,
        considered_dataset: Dataset,
        index_to_retrieve: int,
        attr_to_retrieve: str,
        required_class_to_select_by_attr: type,
    ) -> torch.Tensor | None:
        """A set o optimizations to avoid reitaration through tensor based dataset mutliple times
        in active experiments in situations in which models assume whole training set/pool
        loaded in memory.

        This method computes common dataset operations like concat and subset directly on
        underlying tensors, therefore avoiding unnecessary reiterations through set if possible.

        In case of success retrieved tensor is returned after common dataset operations applied.

        Parameters
        ----------
        considered_dataset: Dataset
            Dataset from desired tensor representation should be retrieved.
        index_to_retrieve : int
            Index to retrieve from TensorDataset, usually 0 indicates features and 1 targets.
        attr_to_retrieve : str
            Attribute via which tensor will be retrieved if `required_class_to_select_by_attr`
            is encountered.
        required_class_to_select_by_attr : type
            Class for which `attr_to_retrieve` will be accessed to retrieve tensor from attribute.
            Usually subclass of this class is used.
        Returns
        -------
        torch.Tensor | None
            Retrieved tensor or None if retrival failed.
        """
        if isinstance(considered_dataset, required_class_to_select_by_attr):
            return getattr(considered_dataset, attr_to_retrieve)
        elif isinstance(considered_dataset, TensorDataset):
            return considered_dataset.tensors[index_to_retrieve]
        elif isinstance(considered_dataset, Subset):
            indices = considered_dataset.indices
            values = self._optimized_retrieve_for_tensor_types(
                considered_dataset=considered_dataset.dataset,
                index_to_retrieve=index_to_retrieve,
                attr_to_retrieve=attr_to_retrieve,
                required_class_to_select_by_attr=required_class_to_select_by_attr,
            )
            if values is not None:
                return values[indices]
        elif isinstance(considered_dataset, ConcatDataset):
            gathered_values = []
            for dataset in considered_dataset.datasets:
                dataset_values = self._optimized_retrieve_for_tensor_types(
                    considered_dataset=dataset,
                    index_to_retrieve=index_to_retrieve,
                    attr_to_retrieve=attr_to_retrieve,
                    required_class_to_select_by_attr=required_class_to_select_by_attr,
                )
                if dataset_values is None:
                    return None
                gathered_values.append(dataset_values)

            return torch.concat(gathered_values)

        return None

    def _initialize_features_by_iteration(self):
        values = self._retrieve_by_iteration(index_to_retrieve=0)
        self._features = values

    def _retrieve_by_iteration(self, index_to_retrieve):
        values = []
        for batch in self._iterate_over_dataset():
            values_batch = batch[index_to_retrieve]
            values.append(values_batch)

        if len(values) > 0:
            values = torch.concat(values)
        else:
            values = torch.empty((0, 1))
        return values

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
        if not self._optimized_initialize_targets_for_tensor_types():
            self._initialize_targets_by_iteration()

    def _optimized_initialize_targets_for_tensor_types(self):
        targets = self._optimized_retrieve_for_tensor_types(
            considered_dataset=self.dataset,
            index_to_retrieve=1,
            attr_to_retrieve="targets",
            required_class_to_select_by_attr=ALDataset,
        ).to(self._targets_dtype)
        if targets is not None:
            self._targets = targets
            return True

        return False

    def _initialize_targets_by_iteration(self):
        values = self._retrieve_by_iteration(index_to_retrieve=1)
        self._targets = values.to(self._targets_dtype)

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
