"""Classification metrics implemented directly, so the training loop pulls in no extra library.

``roc_auc`` is the rank-based form of the Mann-Whitney U statistic with ties averaged, which is
exactly equivalent to the area under the ROC curve and costs one sort. It is cross-checked
against scikit-learn in ``tests/unit/test_metrics_and_data.py``, test
``test_roc_auc_matches_scikit_learn``, and scikit-learn is also what the
centralized baseline uses, so the federated numbers are compared against an independent
implementation rather than against this one.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = ["accuracy", "brier_score", "log_loss", "roc_auc"]

FloatArray = npt.NDArray[np.float64]


def _checked(scores: npt.ArrayLike, labels: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
    y_score = np.asarray(scores, dtype=np.float64).ravel()
    y_true = np.asarray(labels, dtype=np.float64).ravel()
    if y_score.size != y_true.size:
        raise ValueError("scores and labels must have the same length")
    if y_score.size == 0:
        raise ValueError("at least one observation is required")
    if not np.all(np.isfinite(y_score)):
        raise ValueError("scores contain NaN or infinite values")
    if not np.all(np.isin(y_true, (0.0, 1.0))):
        raise ValueError("labels must be 0 or 1")
    return y_score, y_true


def roc_auc(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Area under the ROC curve via average ranks. Ties contribute 0.5 as they should."""
    y_score, y_true = _checked(scores, labels)
    positives = float(np.sum(y_true))
    negatives = float(y_true.size) - positives
    if positives == 0.0 or negatives == 0.0:
        raise ValueError("ROC AUC needs both classes to be present")
    order = np.argsort(y_score, kind="mergesort")
    sorted_scores = y_score[order]
    ranks = np.empty(y_score.size, dtype=np.float64)
    start = 0
    while start < sorted_scores.size:
        stop = start + 1
        while stop < sorted_scores.size and sorted_scores[stop] == sorted_scores[start]:
            stop += 1
        ranks[order[start:stop]] = 0.5 * (start + stop + 1)  # 1-based average rank
        start = stop
    rank_sum = float(np.sum(ranks[y_true == 1.0]))
    return (rank_sum - positives * (positives + 1.0) / 2.0) / (positives * negatives)


def log_loss(scores: npt.ArrayLike, labels: npt.ArrayLike, *, eps: float = 1e-12) -> float:
    """Mean binary cross-entropy of predicted probabilities, clipped away from 0 and 1."""
    y_score, y_true = _checked(scores, labels)
    if np.any(y_score < 0.0) or np.any(y_score > 1.0):
        raise ValueError("log_loss expects probabilities in [0, 1]")
    clipped = np.clip(y_score, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(clipped) + (1.0 - y_true) * np.log(1.0 - clipped)))


def brier_score(scores: npt.ArrayLike, labels: npt.ArrayLike) -> float:
    """Mean squared error of predicted probabilities."""
    y_score, y_true = _checked(scores, labels)
    return float(np.mean((y_score - y_true) ** 2))


def accuracy(scores: npt.ArrayLike, labels: npt.ArrayLike, *, threshold: float = 0.5) -> float:
    y_score, y_true = _checked(scores, labels)
    return float(np.mean((y_score >= threshold).astype(np.float64) == y_true))
