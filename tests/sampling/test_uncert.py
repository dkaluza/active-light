import pytest
import torch

from al.sampling.uncert import confidence_ratio, entropy, least_confidence, margin
from al.sampling.uncert.evidence import _compute_proba_layers


@pytest.fixture(params=[entropy, least_confidence, margin, confidence_ratio])
def uncert_func(request: pytest.FixtureRequest):
    return request.param


@pytest.mark.parametrize(
    "probas",
    [
        torch.eye(2, dtype=torch.float),
        torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float),
    ],
)
def test_uncert_pure_distribution_returns_zero(uncert_func, probas):
    uncert_values = uncert_func(probas)

    assert torch.allclose(uncert_values, torch.zeros_like(uncert_values))


@pytest.mark.parametrize(
    "probas, expected_shape",
    [
        (torch.rand(2, 7), 2),
        (torch.rand(0, 2), 0),
        (torch.rand(1, 3), 1),
        (torch.rand(3, 2), 3),
    ],
)
def test_uncert_return_n_samples_shape(uncert_func, probas, expected_shape):
    assert uncert_func(probas).shape == (expected_shape,)


@pytest.mark.parametrize(
    "probas",
    [
        torch.rand(1, 1),
        torch.rand(3, 1),
        torch.rand(2, 0),
    ],
)
def test_uncert_raises_on_wrong_input_shape(uncert_func, probas):
    with pytest.raises(ValueError):
        uncert_func(probas)


@pytest.mark.parametrize(
    "probas, expected_values",
    [
        (
            torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float),
            torch.tensor([[0, 1], [0, 1]], dtype=torch.float),
        ),
        (
            torch.tensor([[0.5, 0.5, 0], [0.3, 0.2, 0.5]], dtype=torch.float),
            torch.tensor([[0, 0.5, 0], [0.2, 0.1, 0.2]], dtype=torch.float),
        ),
    ],
)
def test_compute_proba_layers_pads_values(probas, expected_values):
    proba_layers = _compute_proba_layers(probas)
    assert torch.allclose(proba_layers.probas, expected_values)


@pytest.mark.parametrize(
    "probas, expected_layers",
    [
        (
            torch.tensor([[0, 0, 1], [1, 0, 0]], dtype=torch.float),
            torch.tensor(
                [
                    [[True, True, True], [False, False, True]],
                    [[True, True, True], [True, False, False]],
                ],
                dtype=torch.bool,
            ),
        ),
        (
            torch.tensor([[0.5, 0.5, 0], [0.3, 0.2, 0.5]], dtype=torch.float),
            torch.tensor(
                [
                    [[True, True, True], [True, True, False], [True, True, True]],
                    [[True, True, True], [True, False, True], [False, False, True]],
                ],
                dtype=torch.bool,
            ),
        ),
    ],
)
def test_compute_proba_layers_pads_values(probas, expected_layers):
    proba_layers = _compute_proba_layers(probas)
    assert torch.all(proba_layers.layers == expected_layers)
