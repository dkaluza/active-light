from typing import Protocol, runtime_checkable

import torch
from torch.utils.data import Dataset

CLASSES_DIM = -1


def get_default_torch_device() -> torch.device:
    return torch.empty(
        1
    ).device  # a trick to get default device since appropriate helper does not exist


@runtime_checkable
class ModelProto(Protocol):
    def fit(self, train: Dataset):
        ...

    def predict_proba(self, data: Dataset) -> torch.FloatTensor:
        """Predict probability distributions for given data samples.

        Parameters
        ----------
        data : Dataset
            Data samples for which probability distributions should be computed.

        Returns
        -------
        torch.FloatTensor
            Probability distribution representations for each sample in
            the same order as samples occur in the dataset.
            In case of discrete events, e.g. classification tasks probabilies
            of each event should be returned directly.
            In case of continous distributions, e.g. regression task
            parameters of the distribution should be returned according to
            documentation of the specific model.
        """
        ...

    def predict(self, data: Dataset) -> torch.FloatTensor:
        ...


@runtime_checkable
class RegressionModelProto(ModelProto, Protocol):
    def get_variance(self, distribution_params: torch.Tensor) -> torch.FloatTensor:
        ...

    def get_mode(self, distribution_params: torch.Tensor) -> torch.FloatTensor:
        ...

    def get_expected_value(
        self, distribution_params: torch.Tensor
    ) -> torch.FloatTensor:
        ...
