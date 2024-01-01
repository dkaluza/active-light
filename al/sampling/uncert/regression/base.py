import abc

import torch
from torch import FloatTensor, Tensor
from torch.utils.data import Dataset

from al.base import ActiveState, ModelProto, RegressionModelProto
from al.sampling.base import InformativenessProto


class UncertRegressionBase(InformativenessProto, abc.ABC):
    @abc.abstractmethod
    def _call(
        self, distribution_params: Tensor, model: RegressionModelProto
    ) -> FloatTensor:
        ...

    def __call__(self, state: ActiveState) -> FloatTensor:
        probas: Tensor = state.get_probas()
        model: ModelProto = state.get_model()
        if not len(probas):
            return torch.empty(0)

        assert (
            model is not None
        ), " Model has to be always passed for uncert based regression infos."

        assert isinstance(
            model, RegressionModelProto
        ), "Model has to fulfill the RegressionModelProto"

        return self._call(distribution_params=probas, model=model)
