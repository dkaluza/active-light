from __future__ import annotations

import enum
from typing import Any, Callable, NamedTuple, Self, Sequence, TypeAlias

import ngboost
import numpy as np
import openml
import ormsgpack as msgpack
import pandas as pd
import sklearn
import torch
from ngboost.distns import Normal, RegressionDistn
from pydantic import BaseModel
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm.auto import tqdm
from xgboost import XGBModel

from al.base import ModelProto, RegressionModelProto
from al.loops.base import ALDataset, FloatTensor, LoopMetricName, LoopResults
from al.sampling.base import InformativenessProto
from al.sampling.uncert.classification.base import UncertClassificationBase

EXPERIMENT_RESULTS_SERIALIZE_EXTENSION = 9
LOOP_RESULTS_SERIALIZE_EXTENSION = 10
TORCH_SERIALIZE_EXTENSION = 11


def default_serizalize(obj: object):
    if isinstance(obj, torch.Tensor):
        if obj.shape == tuple():
            obj = obj.item()
        else:
            obj = obj.numpy(force=True)
        return msgpack.Ext(
            TORCH_SERIALIZE_EXTENSION,
            msgpack.packb(obj, option=msgpack.OPT_SERIALIZE_NUMPY),
        )

    if isinstance(obj, LoopResults):
        return msgpack.Ext(
            LOOP_RESULTS_SERIALIZE_EXTENSION,
            msgpack.packb(dict(obj), default=default_serizalize),
        )
    if isinstance(obj, ExperimentResults):
        return msgpack.Ext(
            EXPERIMENT_RESULTS_SERIALIZE_EXTENSION,
            msgpack.packb(
                obj.model_dump(),
                default=default_serizalize,
            ),
        )
    raise TypeError(f"{type(obj)} type is not supported in serialization")


def default_unserialize(code, data):
    if code == TORCH_SERIALIZE_EXTENSION:
        # WARNING: ormsgpack serializes ndarrays as lists...
        # therefore we are creating tensor with torch.tensor and not from_numpy
        unpacked_ndarray = msgpack.unpackb(data)
        obj = torch.tensor(
            unpacked_ndarray,
        )
        return obj
    if code == LOOP_RESULTS_SERIALIZE_EXTENSION:
        data_dict = msgpack.unpackb(data, ext_hook=default_unserialize)
        loop_result = LoopResults.model_validate(data_dict)
        return loop_result
    if code == EXPERIMENT_RESULTS_SERIALIZE_EXTENSION:
        data_dict = msgpack.unpackb(data, ext_hook=default_unserialize)
        experiment_results = ExperimentResults.model_validate(data_dict)
        return experiment_results

    raise TypeError(f"Extension type with {code} is not supported")


ConfigurationName: TypeAlias = str


class ConfigurationResults(BaseModel):
    metrics: dict[LoopMetricName, FloatTensor]
    """ 
    Dict from loop metric name to tensors of shape (n_seeds, n_iterations, ...) 
    indicating specific metric values across loop iterations in experiments.
    Common metrics can be found at `al.loops.base.LoopMetric`
    """
    pool_probas: None | FloatTensor = None
    """ Probabilities obtained by model predictions on the samples from the pool. 
    In case of loops that removes samples during pool iteration the tensors are
    padded with NaNs at the end(e.g. classical active learning perfect oracles loop).
    Please keep in mind that due to samples removal and padding the order of samples
    is not maintained after the iterations. 
    The shape of probas is (n_seeds, n_iterations, n_samples, n_classes).
    """

    def __eq__(self, other: Any) -> bool:
        if not isinstance(other, ConfigurationResults):
            return False

        return (
            (
                (self.pool_probas is None and other.pool_probas is None)
                or (
                    self.pool_probas is not None
                    and other.pool_probas is not None
                    and torch.allclose(
                        self.pool_probas, other.pool_probas, equal_nan=True
                    )
                )
            )
            and len(self.metrics) == len(other.metrics)
            and all(
                [
                    (metric_values == other.metrics[metric_name]).all()
                    for metric_name, metric_values in self.metrics.items()
                ]
            )
        )


class ExperimentResults(BaseModel):
    res: dict[ConfigurationName, ConfigurationResults]

    @staticmethod
    def from_loop_results(
        loop_results: dict[str, Sequence[LoopResults]]
    ) -> ExperimentResults:
        result = ExperimentResults(res={})
        for configuration_name, loop_results_for_seeds in loop_results.items():
            pool_probas = ExperimentResults._retrieve_pool_probas_from_loop_results(
                loop_results_for_seeds
            )

            metrics = ExperimentResults._retrieve_metrics_from_loop_results(
                loop_results_for_seeds
            )
            result.res[configuration_name] = ConfigurationResults(
                pool_probas=pool_probas, metrics=metrics
            )

        return result

    @staticmethod
    def _retrieve_pool_probas_from_loop_results(
        loop_results_for_seeds: Sequence[LoopResults],
    ):
        for loop_result in loop_results_for_seeds:
            if loop_result.pool_probas is None:
                return None

        if len(loop_results_for_seeds) == 0:
            return None

        # assumption all of the probas has same n_classes, they differ only in
        # n_samples, therefore we are padding the missing samples
        pool_probas = torch.stack(
            [
                torch.nn.utils.rnn.pad_sequence(
                    loop_result.pool_probas,
                    padding_value=torch.nan,
                    batch_first=True,
                )
                for loop_result in loop_results_for_seeds
            ],
            dim=0,
        )

        return pool_probas

    @staticmethod
    def _retrieve_metrics_from_loop_results(
        loop_results_for_seeds: Sequence[LoopResults],
    ):
        if len(loop_results_for_seeds) == 0:
            return {}

        metrics: dict[str, list] = {}
        for seed_result in loop_results_for_seeds:
            if seed_result.metrics is not None:
                for metric_name, metric_values in seed_result.metrics.items():
                    metrics.setdefault(metric_name, []).append(
                        torch.tensor(metric_values)
                    )

        metrics: dict[str, FloatTensor]
        for metric_name, metric_values in metrics.items():
            metrics[metric_name] = torch.stack(metric_values)

        return metrics

    def save(self, path: str):
        with open(path, "wb") as file:
            buffer = msgpack.packb(
                self,
                default=default_serizalize,
            )
            file.write(buffer)

    @staticmethod
    def load(path: str) -> ExperimentResults:
        with open(path, "rb") as file:
            buffer = file.read()
            obj = msgpack.unpackb(buffer, ext_hook=default_unserialize)
            assert isinstance(obj, ExperimentResults)
            return obj


def run_experiment(
    loop: Callable[..., LoopResults],
    *,
    data: Dataset,
    infos: Sequence[InformativenessProto],
    init_frac: float | int = 0.001,
    test_frac: float | int = 0.5,
    n_repeats: int = 10,
    seed: int = 42,
    save_path: str | None = None,
    **loop_kwargs,
) -> ExperimentResults:
    device = torch.empty(
        1
    ).device  # a trick to get default device since appropriate helper does not exist
    generator = torch.Generator(device=device).manual_seed(seed)

    if isinstance(test_frac, float):
        test_frac = int(len(data) * test_frac)
    if isinstance(init_frac, float):
        init_frac = int(len(data) * init_frac)

    pool_frac = len(data) - init_frac - test_frac

    results: dict[str, list[LoopResults]] = dict()
    with tqdm(total=n_repeats * len(infos), desc="Loops:") as progress_bar:
        for _ in range(n_repeats):
            initial_train, pool, test = random_split(
                data, (init_frac, pool_frac, test_frac), generator=generator
            )

            for info in infos:
                loop_result = loop(
                    initial_train=ALDataset(initial_train),
                    pool=ALDataset(pool),
                    test=ALDataset(test),
                    info_func=info,
                    **loop_kwargs,
                )
                results.setdefault(info.__name__, []).append(loop_result)
                progress_bar.update(1)

            experiment_results = ExperimentResults.from_loop_results(results)
            if save_path is not None:
                experiment_results.save(save_path)

    return experiment_results


def save_results(path: str, results: ExperimentResults):
    with open(path, "wb") as file:
        buffer = msgpack.packb(
            results,
            default=default_serizalize,
        )
        file.write(buffer)


def load_results(path: str) -> ExperimentResults:
    with open(path, "rb") as file:
        buffer = file.read()
        return msgpack.unpackb(buffer, ext_hook=default_unserialize)


class CreateALDatasetResult(NamedTuple):
    dataset: ALDataset
    cat_features: list[bool]
    attr_names: list[str]


def create_AL_dataset_from_openml(dataset_id: int) -> CreateALDatasetResult:
    dataset_handle = openml.datasets.get_dataset(dataset_id, download_data=True)
    dataset_X, dataset_Y, cat_features, attr_names = dataset_handle.get_data(
        dataset_format="dataframe", target=dataset_handle.default_target_attribute
    )
    dataset_X: pd.DataFrame
    dataset_Y: pd.Series
    if sum(cat_features) > 0:
        dataset_X = pd.concat(
            [
                pd.get_dummies(dataset_X.iloc[:, cat_features]),
                dataset_X.iloc[:, ~np.array(cat_features)],
            ],
            axis=1,
        )
    dataset_X, dataset_Y = torch.tensor(
        dataset_X.to_numpy(dtype=np.float64)
    ), torch.tensor(dataset_Y.cat.codes.to_numpy(dtype=np.int64))

    dataset = ALDataset(TensorDataset(dataset_X, dataset_Y))
    return CreateALDatasetResult(dataset, cat_features, attr_names)


def add_uncert_metric_for_probas(
    results: ExperimentResults,
    uncerts: Sequence[UncertClassificationBase],
    metric: Callable[..., torch.FloatTensor],
    name: str | None = None,
) -> ExperimentResults:
    """Adds custom uncertainty metric to results based on pool probas saved in results.

    The metric will be added to results *in place* with key specifed by `name` parameter or `metric.__name__`
    in case it `name` is missing.

    Parameters
    ----------
    results : ExperimentResults
        Results for which uncertainty metric should be computet and added to metric values.
        They should have non empty `pool_proba`.
    uncerts : Sequence[UncertBase]
        Sequence of uncertainty functions that should be used in metric computation.
        They will be matched with results keys using `uncert.__name__`
    metric : Callable[..., torch.FloatTensor]
        Metric that will be computed using uncertainty function probabilities from results.
        For examples see `al.sampling.uncert.metrics`.
    name : str | None
        Name that will be used as key to `LoopResults.metrics` to save newly computed metric values.
        If it is `None` `metric.__name__` will be used instead.

    Returns
    -------
    ExperimentResults
        Results with metric values saved in `LoopResults.metrics` field.
    """
    metric_name = name if name is not None else metric.__name__
    uncerts = dict([(uncert.__name__, uncert) for uncert in uncerts])

    with tqdm(total=len(uncerts), desc="Uncerts:") as progress_bar:
        for config_name, config_results in results.res.items():
            uncert = uncerts[config_name]

            # shape of probas (n_seeds, n_iters, n_samples, n_classes)
            probas = config_results.pool_probas
            metric_values = metric(func=uncert, distribution=probas).nanmean(dim=2)
            config_results.metrics[metric_name] = metric_values
            progress_bar.update(1)

    return results


def add_uncert_metric_for_max_uncert_proba(
    results: ExperimentResults,
    uncerts: Sequence[UncertClassificationBase],
    metric: Callable[..., torch.FloatTensor],
    name: str | None = None,
) -> ExperimentResults:
    metric_name = name if name is not None else metric.__name__
    uncerts = dict([(uncert.__name__, uncert) for uncert in uncerts])

    with tqdm(total=len(uncerts), desc="Uncerts:") as progress_bar:
        for config_name, config_results in results.res.items():
            uncert = uncerts[config_name]

            # shape of probas (n_seeds, n_iters, n_samples, n_classes)
            samples_dim = 2
            classes_dim = -1

            probas = config_results.pool_probas
            uncert_values = uncert(probas=probas)
            max_uncert_proba_idx = nanargmax(
                uncert_values, dim=samples_dim, keepdim=True
            )
            selected_proba = torch.take_along_dim(
                probas, max_uncert_proba_idx.unsqueeze(dim=classes_dim), dim=samples_dim
            )
            selected_proba = selected_proba.unsqueeze(samples_dim)

            # there should be only one element in the 2 dim
            metric_values = metric(func=uncert, distribution=selected_proba)[:, :, 0]
            config_results.metrics[metric_name] = metric_values
            progress_bar.update(1)

    return results


# insipired by https://github.com/pytorch/pytorch/issues/61474 amitabe post
# workaround for no torch nanargmax
def nanargmax(tensor, dim=None, keepdim=False):
    min_value = torch.finfo(tensor.dtype).min
    indices = tensor.nan_to_num(min_value).max(dim=dim, keepdim=keepdim).indices
    return indices


class XGBWrapper(ModelProto):
    def __init__(self, model: XGBModel) -> None:
        super().__init__()
        self.model = model

    def fit(self, train: Dataset):
        train = ALDataset(train)
        features = train.features
        targets = train.targets
        if self.model.device == "cuda":
            import cupy

            features = cupy.asarray(features)
            targets = cupy.asarray(targets)
        else:
            features = features.numpy(force=True)
            targets = targets.numpy(force=True)
        self.model.fit(features, targets)

    def predict_proba(self, data: Dataset) -> torch.FloatTensor:
        data = ALDataset(data)
        features = data.features

        if self.model.device == "cuda":
            import cupy

            features = cupy.asarray(features)
        else:
            features = features.numpy(force=True)
        return torch.tensor(self.model.predict_proba(features))


class NClassesGuaranteeWrapper(ModelProto):
    model: ModelProto
    targets_encoder: None | sklearn.preprocessing.LabelEncoder

    def __init__(self, model: ModelProto, n_classes: int) -> None:
        super().__init__()
        self.model = model
        self.n_classes = n_classes
        self.targets_encoder = None

    def fit(self, train: Dataset):
        train = ALDataset(train)
        features = train.features
        targets = train.targets
        train_n_classes = train.n_classes

        if train_n_classes != self.n_classes:
            self.targets_encoder = sklearn.preprocessing.LabelEncoder()
            targets = self.targets_encoder.fit_transform(targets)
            targets = torch.tensor(targets)
            train = TensorDataset(features, targets)
        else:
            self.targets_encoder = None

        self.model.fit(train)

    def predict_proba(self, data: Dataset) -> torch.FloatTensor:
        data = ALDataset(data)
        predicted_probas = self.model.predict_proba(data)
        if self.targets_encoder is not None:
            probas = torch.zeros((len(data), self.n_classes), dtype=torch.float)
            trained_on_classes = torch.tensor(self.targets_encoder.classes_)
            probas[:, trained_on_classes] = predicted_probas
        else:
            probas = predicted_probas

        return probas


class NGBoostDist(enum.Enum):
    Normal = "normal"


def get_NGBoostDist(dist) -> NGBoostDist:
    if dist == Normal:
        return NGBoostDist.Normal

    raise NotImplementedError()


class NGBoostRegressionWrapper(RegressionModelProto):
    """NGBoost wrapper implementing regression model proto.


    The distribution params are always torch tensor of parameters stacked in dim=1.
    The order of parameters is as follows for the implemented distributions:
    Normal - [loc, scale]

    """

    def __new__(cls, model: ngboost.NGBRegressor) -> Self:
        dist = get_NGBoostDist(model.Dist)
        if dist == NGBoostDist.Normal:
            return super().__new__(NGBoostRegressionNormalWrapper)

    def __init__(self, model: ngboost.NGBRegressor) -> None:
        super().__init__()
        self.model = model

    def fit(self, train: Dataset):
        train = ALDataset(train)
        features = train.features
        targets = train.targets
        print(targets)
        self.model.fit(features.numpy(), targets.numpy())

    def predict_proba(self, data: Dataset) -> FloatTensor:
        data = ALDataset(data)
        features = data.features

        dist_params = self.model.pred_dist(features)
        dist_params = self._get_params_based_on_dist(dist_params)
        return dist_params

    def _get_params_based_on_dist(self, dist: RegressionDistn):
        return torch.stack(
            [
                torch.from_numpy(dist.loc),
                torch.from_numpy(dist.scale),
            ],
            dim=1,
        )

    def predict(self, data: Dataset) -> FloatTensor:
        data = ALDataset(data)
        features = data.features

        preds = self.model.predict(features)
        return torch.from_numpy(preds)


class NGBoostRegressionNormalWrapper(NGBoostRegressionWrapper):
    """
    An NGBoost regression wrapper with model predicting normal distribution.

    Params of the model are always represented with torch tensor with
    (n_samples, n_params) shape with: loc as a first param and scale as the second.

    """

    def get_variance(self, distribution_params: torch.Tensor) -> torch.FloatTensor:
        return distribution_params[:, 1] ** 2

    def get_mode(self, distribution_params: torch.Tensor) -> torch.FloatTensor:
        return distribution_params[:, 0]

    def get_expected_value(
        self, distribution_params: torch.Tensor
    ) -> torch.FloatTensor:
        return distribution_params[:, 0]
