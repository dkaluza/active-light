from dataclasses import dataclass
from typing import Protocol, Sequence

import torch
from torch.utils.data import ConcatDataset, Dataset, Subset

from al.base import ModelProto


class ActiveState(Protocol):
    """State of active computation, giving access to needed data and models.

    Methods needed depend on the query strategies used. E.g. most of the uncert sampling
    strategies will use only get_probas method and Query by Committe will use training data
    model and pool.

    Methods might be implemented in various ways, e.g. they may be a result of database select
    or might be just in memory objects as in case of active experiments.
    """

    def has_probas(self) -> bool:
        ...

    def get_probas(self) -> torch.Tensor:
        ...

    def get_model(self) -> ModelProto:
        ...

    def get_pool(self) -> Dataset:
        ...

    def get_training_data(self) -> Dataset:
        ...

    def select_samples(self, pool_idx: Sequence[int], remove_from_pool: bool):
        # TODO: make pool data without labels to make sure none of methods access them
        ...


@dataclass(kw_only=True)
class ActiveInMemoryState(ActiveState):
    probas: torch.Tensor | None = None
    model: ModelProto | None = None
    pool: Dataset | None = None
    training_data: Dataset | None = None

    def has_probas(self) -> bool:
        return self.probas is not None

    def get_probas(self) -> torch.Tensor:
        if self.probas is None:
            assert (
                self.model is not None and self.pool is not None
            ), "In case of no probas passed model and dataset have to be defined."

            self.probas = self.model.predict_proba(self.pool)

        return self.probas

    def get_model(self) -> ModelProto:
        assert self.model is not None
        return self.model

    def get_pool(self) -> Dataset:
        assert self.pool is not None
        return self.pool

    def get_training_data(self) -> Dataset:
        assert self.training_data is not None
        return self.training_data

    def select_samples(self, pool_idx: Sequence[int], remove_from_pool: bool):
        assert self.training_data is not None and self.pool is not None
        selected_samples = Subset(self.pool, pool_idx)
        if remove_from_pool:
            self.pool = remove_indices_from_dataset(self.pool, pool_idx)
        self.training_data = ConcatDataset([self.training_data, selected_samples])

    def refit_model(self):
        assert self.model is not None
        self.probas = None
        self.model.fit(train=self.training_data)


def remove_indices_from_dataset(dataset: Dataset, indices: list[int]) -> Dataset:
    remaining_idx = torch.arange(len(dataset), dtype=torch.long)

    remaining_idx_mask = torch.full_like(
        remaining_idx, fill_value=True, dtype=torch.bool
    )

    remaining_idx_mask[indices] = False
    remaining_idx = remaining_idx[remaining_idx_mask]

    return Subset(dataset, remaining_idx)


class InformativenessProto(Protocol):
    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        ...

    @property
    def __name__(self) -> str:
        return self.__class__.__name__
