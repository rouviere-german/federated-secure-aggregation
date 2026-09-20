"""Local training at one silo, and the update it contributes to a round.

The silo runs mini-batch gradient descent on its own rows, starting from the current global
parameters, and contributes the **delta** ``theta_local - theta_global``. Deltas, not
parameters, are what gets clipped, masked and summed: the clipping bound then means "how much
one silo may move the model", which is both the privacy sensitivity and the quantity the
aggregate norm check bounds.

Two objectives are supported:

- **FedAvg** (McMahan et al. 2017): plain local SGD.
- **FedProx** (Li et al. 2020): local SGD with a proximal term ``mu/2 * ||theta - theta_global||^2``
  that keeps a silo from drifting far from the global model. With non-IID silos this trades a
  little local fit for much more stable aggregation; ``mu = 0`` recovers FedAvg exactly, which
  a test asserts.

Everything is deterministic given the seed: the batch order comes from a generator derived from
(run seed, silo id, round), so re-running a round reproduces the same update bit for bit.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.fl.models import FederatedModel

__all__ = ["LocalTrainingConfig", "local_update", "train_local"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class LocalTrainingConfig:
    """Hyper-parameters every silo shares (they are part of the protocol, not private)."""

    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.5
    l2: float = 1e-4
    proximal_mu: float = 0.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be >= 1")
        if self.batch_size < 1:
            raise ValueError("batch_size must be >= 1")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be > 0")
        if self.l2 < 0 or self.proximal_mu < 0:
            raise ValueError("l2 and proximal_mu must be >= 0")
        if self.learning_rate * self.proximal_mu >= 2.0:
            # The proximal term alone is the linear iteration theta <- theta - lr*mu*(theta-anchor),
            # which diverges once lr*mu >= 2. Catching it here beats shipping NaNs into a round.
            raise ValueError(
                f"learning_rate * proximal_mu = {self.learning_rate * self.proximal_mu:.3f} "
                "must stay below 2, otherwise the proximal step diverges"
            )


def train_local(
    model: FederatedModel,
    global_params: FloatArray,
    features: FloatArray,
    labels: FloatArray,
    config: LocalTrainingConfig,
    *,
    seed: int,
) -> FloatArray:
    """Run local mini-batch SGD and return the new local parameters."""
    params = np.asarray(global_params, dtype=np.float64).copy()
    anchor = params.copy()
    matrix = np.asarray(features, dtype=np.float64)
    targets = np.asarray(labels, dtype=np.float64).ravel()
    if matrix.shape[0] != targets.size:
        raise ValueError("features and labels disagree on the number of rows")
    rng = np.random.default_rng(seed)
    n_rows = matrix.shape[0]
    for _ in range(config.epochs):
        order = rng.permutation(n_rows)
        for start in range(0, n_rows, config.batch_size):
            batch = order[start : start + config.batch_size]
            _, gradient = model.loss_and_grad(params, matrix[batch], targets[batch], l2=config.l2)
            if config.proximal_mu > 0.0:
                gradient = gradient + config.proximal_mu * (params - anchor)
            params -= config.learning_rate * gradient
    return params


def local_update(
    model: FederatedModel,
    global_params: FloatArray,
    features: FloatArray,
    labels: FloatArray,
    config: LocalTrainingConfig,
    *,
    seed: int,
) -> FloatArray:
    """The delta a silo contributes: ``train_local(...) - global_params``."""
    trained = train_local(model, global_params, features, labels, config, seed=seed)
    return trained - np.asarray(global_params, dtype=np.float64)
