

import numpy as np
import torch
import xgboost
from xgboost import XGBClassifier

from al.base import ActiveState
from al.distances import entropy
from al.loops.base import ALDataset
from al.sampling.base import InformativenessProto


class XgbBald(InformativenessProto):
    """
    Bald uncertainty sampling strategy for xgboost model.
    """

    def __init__(self, num_inferences = 100, dropout_proba= 0.1) -> None:
        super().__init__()
        self.num_inferences = num_inferences
        self.dropout_proba = dropout_proba

    def __call__(self, state: ActiveState) -> torch.FloatTensor:
        model = state.get_model().get_wrapped_model()
        assert isinstance(model, XGBClassifier)
        n_estimators = model.get_params()["n_estimators"] or 100 # assuming default 100

        pool = state.get_pool()
        pool = ALDataset(pool)
        features = pool.features

        features_dmatrix = xgboost.DMatrix(features.cpu())


        booster = model.get_booster()
        leaf_predictions = [(booster.predict(data=features_dmatrix, strict_shape=True, output_margin=True, iteration_range=(i, i+1))) for i in range(n_estimators)]
        leaf_predictions = np.stack(leaf_predictions, axis=-1)
        leaf_predictions = torch.from_numpy(leaf_predictions).cuda()
        leaf_shape = leaf_predictions.shape
        predicted_proba = []
        for i in range(self.num_inferences):
            should_be_leaf_used = torch.rand(*leaf_shape) > self.dropout_proba
            values_for_iter = should_be_leaf_used * leaf_predictions
            n_estimators_after_dropout =  should_be_leaf_used.sum(axis=2, keepdim=True)
            values_for_iter = values_for_iter * n_estimators / n_estimators_after_dropout
            values_from_all_leafs = values_for_iter.sum(axis=2)
            probas_for_iter = torch.softmax(values_from_all_leafs, dim=1)
            predicted_proba.append(probas_for_iter)

        predicted_proba = torch.stack(predicted_proba, dim=-1)

        infos = entropy(predicted_proba.mean(dim=-1), dim=-1) - entropy(predicted_proba, dim=1).mean(dim=-1)
        return infos


