import abc
import copy
from collections.abc import Iterable, Iterator, Sequence
from typing import Protocol

import torch
from torch.utils.data import Dataset, Subset, random_split

from al.base import ModelProto, RegressionModelProto
from al.distances import jensen_shannon_divergence
from al.sampling.base import ActiveState, InformativenessProto


class DataDistributionTactic(Protocol):
    def __call__(
        self, n_splits: int, generator: torch.Generator, training_data: Dataset
    ) -> Iterable[Dataset]:
        ...


class RandomSplitTactic(DataDistributionTactic):
    def __call__(
        self, n_splits: int, generator: torch.Generator, training_data: Dataset
    ) -> list[Dataset]:
        return random_split(
            training_data,
            [1 / n_splits for _ in range(n_splits)],
            generator=generator,
        )


class BootstrapTactic(DataDistributionTactic):
    def __call__(
        self, n_splits: int, generator: torch.Generator, training_data: Dataset
    ) -> Iterator[Dataset]:
        for _ in range(n_splits):
            subset_indices = torch.randint(
                low=0,
                high=len(training_data),
                size=(len(training_data),),
                generator=generator,
            )
            yield Subset(dataset=training_data, indices=subset_indices)


class QueryByCommitte(InformativenessProto, metaclass=abc.ABCMeta):
    models: Sequence[ModelProto] = []
    generator: torch.Generator

    def __init__(
        self,
        data_distribution_tactic: DataDistributionTactic,
        n_models=5,
        seed: int | None = 42,
    ) -> None:
        super().__init__()
        self.n_models = n_models
        self.data_distribution_tactic = data_distribution_tactic
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
        datasets = self.data_distribution_tactic(
            n_splits=self.n_models,
            generator=self.generator,
            training_data=training_data,
        )
        self.models = []
        for train_dataset in datasets:
            model = copy.deepcopy(original_model)
            model.fit(train=train_dataset)
            self.models.append(model)

    @abc.abstractmethod
    def _call(self, pool: Dataset) -> torch.FloatTensor:
        ...


class Ambiguity(QueryByCommitte):
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

    def __init__(
        self,
        n_models=5,
        seed: int | None = 42,
    ) -> None:
        super().__init__(RandomSplitTactic(), n_models=n_models, seed=seed)

    def _call(self, pool: Dataset) -> torch.FloatTensor:
        preds = []
        for model in self.models:
            assert isinstance(model, RegressionModelProto)
            preds.append(model.predict(data=pool))

        preds = torch.stack(preds, dim=0)
        return torch.var(preds, dim=0)


class BootstrapJS(QueryByCommitte):
    """Bootstrap JS Query by Committe measure developed for accurate
    class probability estimation.

    Utility of the sample is identified with Jensen-Shannon divergence of distributions from
    multiple models predictions. Models are trained with samples bootstraped from
    training set.

    Based on P. Melville, S. M. Yang, M. Saar-Tsechansky, & Raymond J. Mooney (2005).
    "Active Learning for Probability Estimation using Jensen-Shannon Divergence."
    In Proceedings of the 16th European Conference on Machine Learning (pp. 268–279).

    """

    def __init__(
        self,
        n_models=5,
        seed: int | None = 42,
    ) -> None:
        super().__init__(BootstrapTactic(), n_models, seed)

    def _call(self, pool: Dataset) -> torch.FloatTensor:
        models_proba = []
        for model in self.models:
            assert isinstance(model, ModelProto)
            models_proba.append(model.predict_proba(data=pool))

        js_div = jensen_shannon_divergence(*models_proba)
        return js_div


ambiguity = Ambiguity()
