from typing import Protocol

import torch
from torch.utils.data import Dataset


class ModelProto(Protocol):
    def fit(self, train: Dataset):
        ...

    def predict_proba(self, data: Dataset) -> torch.FloatTensor:
        ...
