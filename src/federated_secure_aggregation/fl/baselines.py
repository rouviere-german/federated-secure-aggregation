"""Reference points a federated number is only meaningful against.

Three baselines, all evaluated on the same untouched holdout:

- **centralized**: one model trained on the pooled data. This is the ceiling the federation is
  trying to approach and the thing regulation usually forbids.
- **local-only**: each silo trains alone on its own rows; the reported figure is the average
  holdout AUC across silos. This is the floor - what a lender gets without co-operating.
- **scikit-learn logistic regression** on the pooled data, as an independent check that the
  hand-written numpy model and gradients are not quietly wrong. It uses a different optimizer
  and a different implementation, so agreement is evidence rather than tautology.

Everything is trained with full-batch gradient descent on standardized features, which is
enough for a well-conditioned logistic problem and keeps the whole comparison inside a few
seconds on a CPU.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.evaluation.metrics import log_loss, roc_auc
from federated_secure_aggregation.fl.local import LocalTrainingConfig, train_local
from federated_secure_aggregation.fl.models import FederatedModel, sigmoid

__all__ = ["BaselineScore", "centralized_baseline", "local_only_baseline", "sklearn_baseline"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class BaselineScore:
    name: str
    auc: float
    log_loss: float
    detail: dict[str, float]


def centralized_baseline(
    model: FederatedModel,
    features: FloatArray,
    labels: FloatArray,
    holdout_features: FloatArray,
    holdout_labels: FloatArray,
    *,
    epochs: int = 60,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> BaselineScore:
    """Train one model on the pooled data: the utility ceiling."""
    config = LocalTrainingConfig(epochs=epochs, batch_size=256, learning_rate=learning_rate, l2=l2)
    params = train_local(model, model.initial_params(), features, labels, config, seed=seed)
    scores = sigmoid(model.logits(params, holdout_features))
    return BaselineScore(
        name="centralized",
        auc=roc_auc(scores, holdout_labels),
        log_loss=log_loss(scores, holdout_labels),
        detail={"rows": float(features.shape[0])},
    )


def local_only_baseline(
    model: FederatedModel,
    silo_data: Mapping[str, tuple[FloatArray, FloatArray]],
    holdout_features: FloatArray,
    holdout_labels: FloatArray,
    *,
    epochs: int = 60,
    learning_rate: float = 0.5,
    l2: float = 1e-4,
    seed: int = 0,
) -> BaselineScore:
    """Train each silo alone and average their holdout AUCs: the no-co-operation floor.

    A silo whose local data happens to be single-class cannot produce a usable score; it is
    skipped and the number of contributing silos is reported, rather than silently averaging a
    degenerate value in.
    """
    config = LocalTrainingConfig(epochs=epochs, batch_size=256, learning_rate=learning_rate, l2=l2)
    aucs: list[float] = []
    losses: list[float] = []
    for index, silo in enumerate(sorted(silo_data)):
        features, labels = silo_data[silo]
        if len(np.unique(labels)) < 2:
            continue
        params = train_local(
            model, model.initial_params(), features, labels, config, seed=seed + index
        )
        scores = sigmoid(model.logits(params, holdout_features))
        aucs.append(roc_auc(scores, holdout_labels))
        losses.append(log_loss(scores, holdout_labels))
    if not aucs:
        raise ValueError("no silo had both classes; the local-only baseline is undefined")
    return BaselineScore(
        name="local_only",
        auc=float(np.mean(aucs)),
        log_loss=float(np.mean(losses)),
        detail={
            "silos": float(len(aucs)),
            "auc_min": float(np.min(aucs)),
            "auc_max": float(np.max(aucs)),
        },
    )


def sklearn_baseline(
    features: FloatArray,
    labels: FloatArray,
    holdout_features: FloatArray,
    holdout_labels: FloatArray,
    *,
    l2: float = 1e-4,
) -> BaselineScore:
    """scikit-learn's logistic regression on the pooled data: an independent implementation."""
    from sklearn.linear_model import (  # noqa: PLC0415 - optional cross-check dependency
        LogisticRegression as SklearnLogisticRegression,
    )

    n_rows = float(features.shape[0])
    estimator = SklearnLogisticRegression(
        C=1.0 / max(l2 * n_rows, 1e-12), max_iter=2000, solver="lbfgs"
    )
    estimator.fit(features, labels)
    scores = estimator.predict_proba(holdout_features)[:, 1]
    return BaselineScore(
        name="sklearn_centralized",
        auc=roc_auc(scores, holdout_labels),
        log_loss=log_loss(scores, holdout_labels),
        detail={"rows": n_rows},
    )
