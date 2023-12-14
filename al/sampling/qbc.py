import abc
import copy
from typing import Sequence

import torch
from torch.utils.data import Dataset, random_split

from al.base import ModelProto, RegressionModelProto
from al.sampling.base import ActiveState, InformativenessProto


class QueryByCommitte(InformativenessProto, metaclass=abc.ABCMeta):
    models: Sequence[ModelProto] = []
    generator: torch.Generator

    def __init__(self, n_models=5, seed: int | None = 42) -> None:
        super().__init__()
        self.n_models = n_models
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        model = state.get_model()
        pool_data = state.get_pool()
        training_data = state.get_training_data()
        assert (
            model is not None and pool_data is not None and training_data is not None
        ), "QBC requires model, pool and training data to be defined!"
        self.fit_models(original_model=model, training_data=training_data)
        return self._call(state.get_pool())

    def fit_models(self, original_model: ModelProto, training_data: Dataset):
        # Note: if splitting changes Ambiguity should be updated to match the paper
        datasets = random_split(
            training_data,
            [1 / self.n_models for _ in range(self.n_models)],
            generator=self.generator,
        )
        self.models = []
        for train_dataset in datasets:
            model = copy.deepcopy(original_model)
            model.fit(train=train_dataset)
            self.models.append(model)

    @abc.abstractmethod
    def _call(self, pool: Dataset) -> torch.FloatTensor:
        ...


class AmbiguityQBC(QueryByCommitte):
    """Ambiguity Query by Committe measure suitable for regression problems.

    Utility of the sample is identified with variance prediction models ensemble.
    Each model is trained on distinct part of the training dataset as in the original paper.

    Based on Burbidge, R., Rowland, J.J., King, R.D. (2007).
    "Active Learning for Regression Based on Query by Committee."
    In: Yin, H., Tino, P., Corchado, E., Byrne, W., Yao, X. (eds)
    Intelligent Data Engineering and Automated Learning - IDEAL 2007. IDEAL 2007.
    Lecture Notes in Computer Science, vol 4881. Springer, Berlin, Heidelberg.
    https://doi.org/10.1007/978-3-540-77226-2_22

    """

    def _call(self, pool: Dataset) -> torch.FloatTensor:
        preds = []
        for model in self.models:
            assert isinstance(model, RegressionModelProto)
            preds.append(model.predict(data=pool))

        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0)


ambiguity_qbc = AmbiguityQBC()
