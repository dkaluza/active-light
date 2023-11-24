from typing import Callable, NamedTuple, Sequence, TypeAlias

import numpy as np
import openml
import ormsgpack as msgpack
import pandas as pd
import sklearn
import torch
from torch.utils.data import Dataset, TensorDataset, random_split
from tqdm.auto import tqdm
from xgboost import XGBModel

from al.base import ModelProto
from al.loops.base import ALDataset, LoopResults
from al.sampling.base import InformativenessProto
from al.sampling.uncert.base import UncertBase

ExperimentResults: TypeAlias = dict[str, Sequence[LoopResults]]


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

            if save_path is not None:
                save_results(save_path, results)

    return results


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

    raise TypeError(f"{type(obj)} type is not supported in serialization")


def save_results(path: str, results: ExperimentResults):
    with open(path, "wb") as file:
        buffer = msgpack.packb(
            results,
            default=default_serizalize,
        )
        file.write(buffer)


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
    raise TypeError(f"Extension type with {code} is not supported")


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
    uncerts: Sequence[UncertBase],
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
        for info_name, loop_results_for_seeds in results.items():
            uncert = uncerts[info_name]

            # shape of probas (n_samples, n_seeds, n_iters, n_classes)
            probas = torch.stack(
                [
                    torch.nn.utils.rnn.pad_sequence(
                        loop_result.pool_probas, padding_value=torch.nan
                    )
                    for loop_result in loop_results_for_seeds
                ],
                dim=1,
            )
            metric_values = metric(func=uncert, distribution=probas).nanmean(dim=0)
            for metric_value, loop_result in zip(metric_values, loop_results_for_seeds):
                loop_result.metrics[metric_name] = metric_value.tolist()
            progress_bar.update(1)

    return results


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
