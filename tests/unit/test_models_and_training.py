"""Models, gradients and local training.

The gradients are written by hand, so every one of them is checked against a central finite
difference. That is the test that would catch a sign error or a missing term - and a sign error
in a gradient is exactly the kind of bug that still produces a plausible-looking training curve.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.fl.local import LocalTrainingConfig, local_update, train_local
from federated_secure_aggregation.fl.models import (
    LogisticRegression,
    MLPClassifier,
    sigmoid,
)

pytestmark = pytest.mark.unit


def _problem(n_rows: int, n_features: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    features = rng.normal(size=(n_rows, n_features))
    logits = features @ rng.normal(size=n_features)
    labels = (rng.random(n_rows) < sigmoid(logits)).astype(np.float64)
    if labels.min() == labels.max():  # pragma: no cover - keeps both classes present
        labels[0] = 1.0 - labels[0]
    return features, labels


def _finite_difference(
    model: LogisticRegression | MLPClassifier,
    params: np.ndarray,
    features: np.ndarray,
    labels: np.ndarray,
    *,
    l2: float,
    step: float = 1e-6,
) -> np.ndarray:
    gradient = np.zeros_like(params)
    for index in range(params.size):
        shifted = params.copy()
        shifted[index] += step
        upper, _ = model.loss_and_grad(shifted, features, labels, l2=l2)
        shifted[index] -= 2 * step
        lower, _ = model.loss_and_grad(shifted, features, labels, l2=l2)
        gradient[index] = (upper - lower) / (2 * step)
    return gradient


@given(
    n_features=st.integers(min_value=1, max_value=6),
    seed=st.integers(min_value=0, max_value=10_000),
    l2=st.sampled_from([0.0, 1e-3, 0.1]),
)
@settings(max_examples=25, deadline=None)
def test_logistic_gradient_matches_finite_differences(
    n_features: int, seed: int, l2: float
) -> None:
    model = LogisticRegression(n_features=n_features)
    features, labels = _problem(40, n_features, seed)
    params = np.random.default_rng(seed).normal(size=model.dimension) * 0.4
    _, analytic = model.loss_and_grad(params, features, labels, l2=l2)
    numeric = _finite_difference(model, params, features, labels, l2=l2)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


@given(
    n_features=st.integers(min_value=1, max_value=4),
    hidden=st.integers(min_value=1, max_value=4),
    seed=st.integers(min_value=0, max_value=10_000),
    l2=st.sampled_from([0.0, 1e-2]),
)
@settings(max_examples=20, deadline=None)
def test_mlp_gradient_matches_finite_differences(
    n_features: int, hidden: int, seed: int, l2: float
) -> None:
    model = MLPClassifier(n_features=n_features, hidden=hidden)
    features, labels = _problem(30, n_features, seed)
    params = np.random.default_rng(seed).normal(size=model.dimension) * 0.4
    _, analytic = model.loss_and_grad(params, features, labels, l2=l2)
    numeric = _finite_difference(model, params, features, labels, l2=l2)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_model_dimensions_and_initialisation() -> None:
    logistic = LogisticRegression(n_features=12)
    assert logistic.dimension == 13
    assert np.array_equal(logistic.initial_params(), np.zeros(13))

    mlp = MLPClassifier(n_features=12, hidden=96)
    assert mlp.dimension == 12 * 96 + 96 + 96 + 1
    params = mlp.initial_params(seed=1)
    assert params.size == mlp.dimension
    assert not np.array_equal(params, mlp.initial_params(seed=2))
    assert np.array_equal(params, mlp.initial_params(seed=1))


def test_models_validate_their_inputs() -> None:
    with pytest.raises(ValueError, match="n_features"):
        LogisticRegression(n_features=0)
    with pytest.raises(ValueError, match="hidden"):
        MLPClassifier(n_features=2, hidden=0)

    model = LogisticRegression(n_features=2)
    features, labels = _problem(10, 2, seed=0)
    with pytest.raises(ValueError, match="expected 3 parameters"):
        model.logits(np.zeros(4), features)
    with pytest.raises(ValueError, match="features must have shape"):
        model.logits(np.zeros(3), np.zeros((10, 5)))
    with pytest.raises(ValueError, match="at least one row"):
        model.logits(np.zeros(3), np.zeros((0, 2)))
    with pytest.raises(ValueError, match="expected 10 labels"):
        model.loss_and_grad(np.zeros(3), features, labels[:5])
    with pytest.raises(ValueError, match="labels must be 0 or 1"):
        model.loss_and_grad(np.zeros(3), features, np.full(10, 2.0))


def test_sigmoid_is_saturating_but_finite() -> None:
    assert sigmoid(0.0) == pytest.approx(0.5)
    assert 0.0 < float(sigmoid(-1e6)) < 1e-20
    assert float(sigmoid(1e6)) == pytest.approx(1.0)


def test_predicted_probabilities_stay_in_range() -> None:
    model = MLPClassifier(n_features=3, hidden=4)
    features, _ = _problem(20, 3, seed=5)
    probabilities = model.predict_proba(model.initial_params(seed=0), features)
    assert np.all((probabilities >= 0.0) & (probabilities <= 1.0))

    logistic = LogisticRegression(n_features=3)
    values = logistic.predict_proba(logistic.initial_params(), features)
    assert np.allclose(values, 0.5)


def test_training_reduces_the_loss_and_is_reproducible() -> None:
    model = LogisticRegression(n_features=5)
    features, labels = _problem(300, 5, seed=9)
    config = LocalTrainingConfig(epochs=8, batch_size=64, learning_rate=0.5, l2=1e-4)
    start = model.initial_params()
    trained = train_local(model, start, features, labels, config, seed=1)
    assert (
        model.loss_and_grad(trained, features, labels)[0]
        < model.loss_and_grad(start, features, labels)[0]
    )
    again = train_local(model, start, features, labels, config, seed=1)
    assert np.array_equal(trained, again)
    different = train_local(model, start, features, labels, config, seed=2)
    assert not np.array_equal(trained, different)


def test_fedprox_with_zero_mu_is_exactly_fedavg() -> None:
    model = LogisticRegression(n_features=4)
    features, labels = _problem(200, 4, seed=3)
    start = model.initial_params()
    fedavg = train_local(
        model, start, features, labels, LocalTrainingConfig(proximal_mu=0.0), seed=7
    )
    explicit = train_local(model, start, features, labels, LocalTrainingConfig(), seed=7)
    assert np.array_equal(fedavg, explicit)


def test_fedprox_keeps_the_local_model_closer_to_the_global_one() -> None:
    model = LogisticRegression(n_features=4)
    features, labels = _problem(200, 4, seed=4)
    start = np.ones(model.dimension) * 0.2
    far = train_local(
        model,
        start,
        features,
        labels,
        LocalTrainingConfig(epochs=6, proximal_mu=0.0),
        seed=1,
    )
    near = train_local(
        model,
        start,
        features,
        labels,
        LocalTrainingConfig(epochs=6, proximal_mu=1.5),
        seed=1,
    )
    assert np.linalg.norm(near - start) < np.linalg.norm(far - start)


def test_local_update_is_the_delta() -> None:
    model = LogisticRegression(n_features=3)
    features, labels = _problem(120, 3, seed=6)
    config = LocalTrainingConfig(epochs=2)
    start = np.full(model.dimension, 0.1)
    trained = train_local(model, start, features, labels, config, seed=2)
    delta = local_update(model, start, features, labels, config, seed=2)
    assert np.allclose(trained - start, delta)


def test_training_config_validation() -> None:
    with pytest.raises(ValueError, match="epochs"):
        LocalTrainingConfig(epochs=0)
    with pytest.raises(ValueError, match="batch_size"):
        LocalTrainingConfig(batch_size=0)
    with pytest.raises(ValueError, match="learning_rate"):
        LocalTrainingConfig(learning_rate=0.0)
    with pytest.raises(ValueError, match="l2 and proximal_mu"):
        LocalTrainingConfig(l2=-1.0)
    with pytest.raises(ValueError, match="proximal step diverges"):
        LocalTrainingConfig(learning_rate=1.0, proximal_mu=2.0)


def test_train_local_checks_shapes() -> None:
    model = LogisticRegression(n_features=2)
    features, labels = _problem(20, 2, seed=8)
    with pytest.raises(ValueError, match="disagree on the number of rows"):
        train_local(
            model, model.initial_params(), features, labels[:5], LocalTrainingConfig(), seed=0
        )
