import abc

import torch
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset

from al.base import ModelProto, RegressionModelProto
from al.sampling.base import InformativenessProto


class UncertRegressionBase(InformativenessProto, abc.ABC):
    @abc.abstractmethod
    def _call(
        self, distribution_params: Tensor, model: RegressionModelProto
    ) -> FloatTensor:
        ...
        # TODO: Consider if model should be also passed

    def __call__(
        self,
        probas: Tensor = None,
        model: ModelProto | None = None,
        dataset: Dataset | None = None,
    ) -> FloatTensor:
        if probas is None:
            assert (
                model is not None and dataset is not None
            ), "In case of no probas passed model and dataset have to be defined."
            probas = model.predict_proba(dataset)

        if not len(probas):
            return torch.empty(0)

        assert (
            model is not None
        ), " Model has to be always passed for uncert based regression infos."

        assert isinstance(
            model, RegressionModelProto
        ), "Model has to fulfill the RegressionModelProto"

        return self._call(distribution_params=probas, model=model)
