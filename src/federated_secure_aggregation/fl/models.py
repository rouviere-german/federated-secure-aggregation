"""Numpy models with hand-written gradients, exposed as flat parameter vectors.

Secure aggregation sums *vectors*, so every model here is defined by a single flat parameter
vector of fixed length. Packing and unpacking are explicit and round-trip exactly, which is
what lets the protocol treat a model update as an anonymous element of ``R^d``.

Gradients are written out rather than obtained from an autodiff framework. That keeps the
dependency list free of torch or tensorflow (the whole suite must run on a CPU in minutes), and
it makes the gradient itself testable: ``tests/unit/test_models.py`` checks every analytic
gradient against a central finite difference.

Numerical care worth noting:

- the logistic loss is computed from logits with the log-sum-exp form
  ``max(z, 0) - z*y + log1p(exp(-|z|))``, which never overflows;
- ``sigmoid`` clips its argument, so a diverging model produces a large loss rather than a NaN;
- L2 regularization never touches the bias terms.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np
import numpy.typing as npt

__all__ = [
    "FederatedModel",
    "LogisticRegression",
    "MLPClassifier",
    "sigmoid",
]

FloatArray = npt.NDArray[np.float64]


def sigmoid(x: npt.ArrayLike) -> FloatArray:
    """Numerically safe logistic function."""
    clipped = np.clip(np.asarray(x, dtype=np.float64), -60.0, 60.0)
    return np.asarray(1.0 / (1.0 + np.exp(-clipped)), dtype=np.float64)


def _binary_cross_entropy(logits: FloatArray, labels: FloatArray) -> float:
    """Mean logistic loss computed from logits without exponentiating a large number."""
    return float(
        np.mean(np.maximum(logits, 0.0) - logits * labels + np.log1p(np.exp(-np.abs(logits))))
    )


@runtime_checkable
class FederatedModel(Protocol):
    """What the federated loop needs from a model: a flat parameter vector and a gradient."""

    @property
    def dimension(self) -> int: ...

    def initial_params(self) -> FloatArray: ...

    def logits(self, params: FloatArray, features: FloatArray) -> FloatArray: ...

    def loss_and_grad(
        self, params: FloatArray, features: FloatArray, labels: FloatArray, *, l2: float = 0.0
    ) -> tuple[float, FloatArray]: ...


@dataclass(frozen=True, slots=True)
class LogisticRegression:
    """Plain logistic regression. Parameters are ``[weights (n_features), bias]``."""

    n_features: int

    def __post_init__(self) -> None:
        if self.n_features < 1:
            raise ValueError("n_features must be >= 1")

    @property
    def dimension(self) -> int:
        return self.n_features + 1

    def initial_params(self) -> FloatArray:
        return np.zeros(self.dimension, dtype=np.float64)

    def _unpack(self, params: FloatArray) -> tuple[FloatArray, float]:
        flat = _check_params(params, self.dimension)
        return flat[: self.n_features], float(flat[self.n_features])

    def logits(self, params: FloatArray, features: FloatArray) -> FloatArray:
        weights, bias = self._unpack(params)
        matrix = _check_features(features, self.n_features)
        return np.asarray(matrix @ weights + bias, dtype=np.float64)

    def predict_proba(self, params: FloatArray, features: FloatArray) -> FloatArray:
        return sigmoid(self.logits(params, features))

    def loss_and_grad(
        self, params: FloatArray, features: FloatArray, labels: FloatArray, *, l2: float = 0.0
    ) -> tuple[float, FloatArray]:
        weights, _ = self._unpack(params)
        matrix = _check_features(features, self.n_features)
        targets = _check_labels(labels, matrix.shape[0])
        z = self.logits(params, matrix)
        residual = sigmoid(z) - targets
        n = matrix.shape[0]
        grad_weights = matrix.T @ residual / n + l2 * weights
        grad_bias = float(np.mean(residual))
        loss = _binary_cross_entropy(z, targets) + 0.5 * l2 * float(weights @ weights)
        return loss, np.concatenate([grad_weights, [grad_bias]])


@dataclass(frozen=True, slots=True)
class MLPClassifier:
    """One hidden ``tanh`` layer, then a logistic output.

    Parameters are packed as ``[W1 (n_features x hidden), b1 (hidden), w2 (hidden), b2]``. The
    forward pass is ``h = tanh(x W1 + b1)``, ``z = h w2 + b2``; the backward pass uses
    ``tanh'(a) = 1 - tanh(a)^2``.
    """

    n_features: int
    hidden: int

    def __post_init__(self) -> None:
        if self.n_features < 1 or self.hidden < 1:
            raise ValueError("n_features and hidden must be >= 1")

    @property
    def dimension(self) -> int:
        return self.n_features * self.hidden + self.hidden + self.hidden + 1

    def initial_params(self, seed: int = 0) -> FloatArray:
        """Symmetric-breaking init: scaled uniform for ``W1``, zeros elsewhere (Glorot-style)."""
        rng = np.random.default_rng(seed)
        limit = np.sqrt(6.0 / (self.n_features + self.hidden))
        first = rng.uniform(-limit, limit, size=self.n_features * self.hidden)
        second = rng.uniform(-limit, limit, size=self.hidden)
        return np.concatenate([first, np.zeros(self.hidden), second, [0.0]]).astype(np.float64)

    def _unpack(self, params: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray, float]:
        flat = _check_params(params, self.dimension)
        cut1 = self.n_features * self.hidden
        cut2 = cut1 + self.hidden
        cut3 = cut2 + self.hidden
        weights1 = flat[:cut1].reshape(self.n_features, self.hidden)
        bias1 = flat[cut1:cut2]
        weights2 = flat[cut2:cut3]
        return weights1, bias1, weights2, float(flat[cut3])

    def logits(self, params: FloatArray, features: FloatArray) -> FloatArray:
        weights1, bias1, weights2, bias2 = self._unpack(params)
        matrix = _check_features(features, self.n_features)
        hidden = np.tanh(matrix @ weights1 + bias1)
        return np.asarray(hidden @ weights2 + bias2, dtype=np.float64)

    def predict_proba(self, params: FloatArray, features: FloatArray) -> FloatArray:
        return sigmoid(self.logits(params, features))

    def loss_and_grad(
        self, params: FloatArray, features: FloatArray, labels: FloatArray, *, l2: float = 0.0
    ) -> tuple[float, FloatArray]:
        weights1, bias1, weights2, bias2 = self._unpack(params)
        matrix = _check_features(features, self.n_features)
        targets = _check_labels(labels, matrix.shape[0])
        n = matrix.shape[0]

        pre_activation = matrix @ weights1 + bias1
        hidden = np.tanh(pre_activation)
        z = hidden @ weights2 + bias2
        residual = (sigmoid(z) - targets) / n  # d loss / d z

        grad_weights2 = hidden.T @ residual + l2 * weights2
        grad_bias2 = float(np.sum(residual))
        backprop = np.outer(residual, weights2) * (1.0 - hidden**2)
        grad_weights1 = matrix.T @ backprop + l2 * weights1
        grad_bias1 = backprop.sum(axis=0)

        loss = _binary_cross_entropy(z, targets) + 0.5 * l2 * (
            float(np.sum(weights1**2)) + float(weights2 @ weights2)
        )
        gradient = np.concatenate([grad_weights1.ravel(), grad_bias1, grad_weights2, [grad_bias2]])
        return loss, gradient.astype(np.float64)


def _check_params(params: FloatArray, dimension: int) -> FloatArray:
    flat = np.asarray(params, dtype=np.float64).ravel()
    if flat.size != dimension:
        raise ValueError(f"expected {dimension} parameters, got {flat.size}")
    return flat


def _check_features(features: FloatArray, n_features: int) -> FloatArray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != n_features:
        raise ValueError(f"features must have shape (n, {n_features}), got {matrix.shape}")
    if matrix.shape[0] == 0:
        raise ValueError("features must contain at least one row")
    return matrix


def _check_labels(labels: FloatArray, n_rows: int) -> FloatArray:
    targets = np.asarray(labels, dtype=np.float64).ravel()
    if targets.size != n_rows:
        raise ValueError(f"expected {n_rows} labels, got {targets.size}")
    if not np.all(np.isin(targets, (0.0, 1.0))):
        raise ValueError("labels must be 0 or 1")
    return targets
