from typing import Protocol, Sequence

import torch

from al.sampling.base import ActiveState, InformativenessProto


class AggregationTactic(Protocol):
    def initialize_result(self, n_values: int) -> torch.FloatTensor:
        ...

    def weight(self, values: torch.FloatTensor, weight: float) -> torch.FloatTensor:
        ...

    def aggregate(
        self, partial_result: torch.FloatTensor, values: torch.FloatTensor
    ) -> torch.FloatTensor:
        ...


class SumAggregation(AggregationTactic):
    def initialize_result(self, n_values: int) -> torch.FloatTensor:
        return torch.zeros((n_values,))

    def weight(self, values: torch.FloatTensor, weight: float) -> torch.FloatTensor:
        return values * weight

    def aggregate(
        self, partial_result: torch.FloatTensor, values: torch.FloatTensor
    ) -> torch.FloatTensor:
        return partial_result + values


class ProductAggregation(AggregationTactic):
    def initialize_result(self, n_values: int) -> torch.FloatTensor:
        return torch.ones((n_values,))

    def weight(self, values: torch.FloatTensor, weight: float) -> torch.FloatTensor:
        return values ** (weight)

    def aggregate(
        self, partial_result: torch.FloatTensor, values: torch.FloatTensor
    ) -> torch.FloatTensor:
        return partial_result * values


class InfoEnsemble(InformativenessProto):
    """Class aggregating multiple informativeness together and aggregating their
    scores accordingh to the defined tactic.

    """

    def __init__(
        self,
        infos: Sequence[InformativenessProto],
        weights: Sequence[float] | None = None,
        aggregation_tactic: AggregationTactic = None,
    ) -> None:
        super().__init__()
        self.infos = infos
        if weights is None:
            weights = [1.0 for _ in infos]
        self.weights = weights
        if aggregation_tactic is None:
            aggregation_tactic = SumAggregation()
        self.aggregation_tactic = aggregation_tactic

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        n_samples = len(state.get_pool())
        results = self.aggregation_tactic.initialize_result(n_samples)
        for weight, info in zip(self.weights, self.infos):
            values = info(state)
            weighted_values = self.aggregation_tactic.weight(values, weight)
            results = self.aggregation_tactic.aggregate(results, weighted_values)

        return results

    @property
    def __name__(self):
        return "Ensemble" + "_".join(
            [
                f"{info.__name__}{weight}"
                for info, weight in zip(self.infos, self.weights)
            ]
        )
