"""Assembling an experiment: population, partition, silos, model and holdout in one object.

Every bench script and the demo start from :func:`build_experiment`, so the data pipeline that
produces a README number is the same one the tests exercise. The run manifest it produces
records the seed, the partition fingerprint, the package versions and the model shape, which is
what makes a pasted result reproducible rather than merely plausible.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.fl.data import (
    SiloPartition,
    Standardizer,
    derive_seed,
    make_population,
    partition_dirichlet,
    standardize,
)
from federated_secure_aggregation.fl.local import LocalTrainingConfig
from federated_secure_aggregation.fl.models import LogisticRegression, MLPClassifier
from federated_secure_aggregation.fl.participants import SiloParticipant
from federated_secure_aggregation.ml.contracts import FEATURE_COLUMNS, LABEL_COLUMN
from federated_secure_aggregation.ml.manifest import RunManifest, create_manifest

__all__ = ["Experiment", "build_experiment", "silo_ids", "write_dataset_snapshot"]

FloatArray = npt.NDArray[np.float64]


def silo_ids(count: int, prefix: str = "silo") -> tuple[str, ...]:
    """Stable, protocol-legal participant ids (``silo_00`` ... ), sorted lexicographically."""
    if count < 1:
        raise ValueError("count must be >= 1")
    width = max(2, len(str(count - 1)))
    return tuple(f"{prefix}_{index:0{width}d}" for index in range(count))


@dataclass(slots=True)
class Experiment:
    """One prepared experiment: standardized silo matrices plus an untouched holdout."""

    partition: SiloPartition
    standardizer: Standardizer
    silo_features: dict[str, FloatArray]
    silo_labels: dict[str, FloatArray]
    holdout_features: FloatArray
    holdout_labels: FloatArray
    n_features: int
    seed: int

    @property
    def silos(self) -> tuple[str, ...]:
        return self.partition.silo_ids

    def pooled(self) -> tuple[FloatArray, FloatArray]:
        """The pooled training data. Only baselines may use this; no silo ever sees it."""
        features = np.concatenate([self.silo_features[silo] for silo in self.silos], axis=0)
        labels = np.concatenate([self.silo_labels[silo] for silo in self.silos], axis=0)
        return features, labels

    def silo_data(self) -> dict[str, tuple[FloatArray, FloatArray]]:
        return {silo: (self.silo_features[silo], self.silo_labels[silo]) for silo in self.silos}

    def logistic_model(self) -> LogisticRegression:
        return LogisticRegression(n_features=self.n_features)

    def mlp_model(self, hidden: int = 64) -> MLPClassifier:
        return MLPClassifier(n_features=self.n_features, hidden=hidden)

    def participants(
        self, model: LogisticRegression | MLPClassifier, config: LocalTrainingConfig
    ) -> list[SiloParticipant]:
        return [
            SiloParticipant(
                silo_id=silo,
                model=model,
                features=self.silo_features[silo],
                labels=self.silo_labels[silo],
                config=config,
                seed=derive_seed(self.seed, "silo", silo),
            )
            for silo in self.silos
        ]

    def manifest(self, *, params: dict[str, object], metrics: dict[str, float]) -> RunManifest:
        payload = {
            "partition_fingerprint": self.partition.fingerprint,
            "dirichlet_alpha": self.partition.alpha,
            "silos": list(self.silos),
            "silo_sizes": self.partition.sizes(),
            "holdout_rows": int(self.holdout_features.shape[0]),
            **params,
        }
        return create_manifest(
            seed=self.seed,
            datasets=[],
            params=payload,  # type: ignore[arg-type]
            metrics=metrics,
            packages=("numpy", "pandas", "pandera", "pydantic", "scipy", "cryptography"),
        )


def build_experiment(
    *,
    n_rows: int = 6_000,
    n_silos: int = 5,
    alpha: float = 1.0,
    seed: int = 20260920,
    holdout_fraction: float = 0.2,
    feature_skew: float = 0.15,
    names: Sequence[str] | None = None,
) -> Experiment:
    """Generate the population, split it into silos and standardize every matrix."""
    population = make_population(n_rows, seed=seed)
    ids = tuple(names) if names is not None else silo_ids(n_silos)
    partition = partition_dirichlet(
        population,
        ids,
        alpha=alpha,
        seed=seed,
        holdout_fraction=holdout_fraction,
        feature_skew=feature_skew,
    )
    standardizer = standardize(population)
    features: dict[str, FloatArray] = {}
    labels: dict[str, FloatArray] = {}
    for silo in partition.silo_ids:
        frame = partition.frames[silo]
        features[silo] = standardizer.transform(
            frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        )
        labels[silo] = frame[LABEL_COLUMN].to_numpy(dtype=np.float64)
    return Experiment(
        partition=partition,
        standardizer=standardizer,
        silo_features=features,
        silo_labels=labels,
        holdout_features=standardizer.transform(
            partition.holdout.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
        ),
        holdout_labels=partition.holdout[LABEL_COLUMN].to_numpy(dtype=np.float64),
        n_features=len(FEATURE_COLUMNS),
        seed=seed,
    )


def write_dataset_snapshot(experiment: Experiment, path: Path) -> Path:
    """Write the holdout to CSV so a run manifest can fingerprint a concrete artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    experiment.partition.holdout.to_csv(path, index=False, lineterminator="\n")
    return path
