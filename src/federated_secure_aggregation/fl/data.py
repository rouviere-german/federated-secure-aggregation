"""A seeded synthetic credit-default population, partitioned into non-IID silos.

Nothing is downloaded and no real borrower data is involved. A latent logistic model with known
coefficients generates the labels, so "how close is the federated model to the best achievable
one" has an answer that does not depend on a benchmark dataset.

The partition follows the Dirichlet label-skew construction that is standard in the federated
learning literature (Hsu, Qi and Brown, *Measuring the Effects of Non-Identical Data
Distribution for Federated Visual Classification*, 2019): for each class, a proportion vector is
drawn from ``Dir(alpha * 1_n)`` and the class's rows are split across silos accordingly. Small
``alpha`` concentrates a class in few silos; large ``alpha`` approaches an IID split.

On top of the label skew, each silo gets a small per-feature location shift, so the silos differ
in their covariates too, which is what makes cross-silo credit data awkward in practice.

Every derived quantity is a deterministic function of the run seed: the population, the per-silo
shifts, the partition and each silo's local training order all come from seeds derived with
``numpy.random.SeedSequence``, and the partition is fingerprinted so a run manifest can record
exactly which split produced a result.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
import pandas as pd

from federated_secure_aggregation.ml.contracts import (
    FEATURE_COLUMNS,
    LABEL_COLUMN,
    validate_borrowers,
)

__all__ = [
    "LATENT_COEFFICIENTS",
    "LATENT_INTERCEPT",
    "SiloPartition",
    "Standardizer",
    "derive_seed",
    "dirichlet_proportions",
    "make_population",
    "partition_dirichlet",
    "standardize",
]

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]

# Signs are chosen so the latent model reads like a credit-risk model: more debt, more
# delinquencies and higher utilization raise default risk; income, credit age and employment
# lower it. Magnitudes are on the standardized scale.
LATENT_COEFFICIENTS: FloatArray = np.array(
    [
        -0.15,  # age_years
        -0.55,  # log_income
        0.70,  # debt_to_income
        0.85,  # revolving_utilization
        0.60,  # delinquencies_2y
        -0.40,  # credit_age_years
        0.50,  # loan_to_income
        -0.30,  # employment_years
        0.35,  # inquiries_6m
        0.10,  # open_accounts
        0.80,  # prior_default
        0.45,  # region_risk
    ],
    dtype=np.float64,
)
LATENT_INTERCEPT = -1.6


def derive_seed(root_seed: int, *labels: str | int | float) -> int:
    """Derive a stable child seed from a root seed and a path of labels.

    Using ``SeedSequence`` rather than ``root + i`` keeps sibling streams independent, and the
    label path means "silo 3's local shuffling in round 7" is reproducible on its own.
    """
    digest = hashlib.sha256("/".join(str(label) for label in labels).encode("utf-8")).digest()
    entropy = int.from_bytes(digest[:8], "big")
    sequence = np.random.SeedSequence([root_seed, entropy])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


def _sigmoid(x: FloatArray) -> FloatArray:
    return np.asarray(1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0))), dtype=np.float64)


def _intercept_offset(logits: FloatArray, base_rate: float) -> float:
    """Bisect for the intercept shift that makes ``mean(sigmoid(logits + offset))`` the target.

    Shifting by ``logit(base_rate) - mean(logits)`` would only be right if the mean of the
    sigmoid were the sigmoid of the mean, which it is not, so the realized default rate would
    miss the target by several points.
    """
    low, high = -30.0, 30.0
    for _ in range(80):
        middle = 0.5 * (low + high)
        if float(np.mean(_sigmoid(logits + middle))) < base_rate:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def make_population(n_rows: int, *, seed: int, base_rate: float = 0.18) -> pd.DataFrame:
    """Generate ``n_rows`` borrowers whose default label follows a known logistic model."""
    if n_rows < 10:
        raise ValueError("n_rows must be >= 10")
    if not 0.01 <= base_rate <= 0.5:
        raise ValueError("base_rate must be in [0.01, 0.5]")
    rng = np.random.default_rng(derive_seed(seed, "population"))

    age = np.clip(rng.normal(44.0, 12.0, n_rows), 18.0, 100.0)
    log_income = np.clip(rng.normal(10.9, 0.55, n_rows), 7.0, 16.0)
    debt_to_income = np.clip(rng.gamma(2.0, 0.18, n_rows), 0.0, 5.0)
    utilization = np.clip(rng.beta(2.0, 4.0, n_rows) * 1.4, 0.0, 2.0)
    delinquencies = np.clip(rng.poisson(0.35, n_rows).astype(np.float64), 0.0, 20.0)
    credit_age = np.clip(rng.gamma(3.0, 3.0, n_rows), 0.0, age - 18.0)
    loan_to_income = np.clip(rng.gamma(2.5, 0.35, n_rows), 0.0, 10.0)
    employment = np.clip(rng.gamma(2.0, 3.5, n_rows), 0.0, np.minimum(50.0, age - 18.0))
    inquiries = np.clip(rng.poisson(1.1, n_rows).astype(np.float64), 0.0, 20.0)
    open_accounts = np.clip(rng.poisson(6.0, n_rows).astype(np.float64) + 1.0, 0.0, 40.0)
    prior_default = (rng.random(n_rows) < 0.11).astype(np.float64)
    region_risk = np.clip(rng.beta(2.0, 3.0, n_rows), 0.0, 1.0)

    frame = pd.DataFrame(
        {
            "borrower_id": [f"b_{index:010x}" for index in range(n_rows)],
            "age_years": age,
            "log_income": log_income,
            "debt_to_income": debt_to_income,
            "revolving_utilization": utilization,
            "delinquencies_2y": delinquencies,
            "credit_age_years": credit_age,
            "loan_to_income": loan_to_income,
            "employment_years": employment,
            "inquiries_6m": inquiries,
            "open_accounts": open_accounts,
            "prior_default": prior_default,
            "region_risk": region_risk,
        }
    )
    features = frame.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    standardized = (features - features.mean(axis=0)) / features.std(axis=0)
    logits = LATENT_INTERCEPT + standardized @ LATENT_COEFFICIENTS
    probabilities = _sigmoid(logits + _intercept_offset(logits, base_rate))
    frame[LABEL_COLUMN] = (rng.random(n_rows) < probabilities).astype(np.int64)
    return frame


def dirichlet_proportions(rng: np.random.Generator, n_silos: int, alpha: float) -> FloatArray:
    """One ``Dir(alpha * 1_n)`` draw, with degenerate inputs rejected rather than clipped."""
    if n_silos < 1:
        raise ValueError("n_silos must be >= 1")
    if alpha <= 0:
        raise ValueError("alpha must be > 0")
    return np.asarray(rng.dirichlet(np.full(n_silos, alpha)), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class SiloPartition:
    """A non-IID split of one population, with the fingerprint that identifies it."""

    silo_ids: tuple[str, ...]
    frames: Mapping[str, pd.DataFrame]
    holdout: pd.DataFrame
    alpha: float
    fingerprint: str

    def label_rates(self) -> dict[str, float]:
        return {silo: float(self.frames[silo][LABEL_COLUMN].mean()) for silo in self.silo_ids}

    def sizes(self) -> dict[str, int]:
        return {silo: len(self.frames[silo]) for silo in self.silo_ids}


def partition_dirichlet(
    population: pd.DataFrame,
    silo_ids: Sequence[str],
    *,
    alpha: float,
    seed: int,
    holdout_fraction: float = 0.2,
    feature_skew: float = 0.15,
    min_rows_per_silo: int = 40,
) -> SiloPartition:
    """Split ``population`` into silos with Dirichlet label skew and per-silo feature shifts.

    A global holdout is carved out first and never touched by any silo, so the reported AUC is
    measured on data the federated model has not seen through any participant.
    """
    if len(silo_ids) < 2:
        raise ValueError("at least two silos are required")
    if len(set(silo_ids)) != len(silo_ids):
        raise ValueError("silo ids must be unique")
    if not 0.0 < holdout_fraction < 0.6:
        raise ValueError("holdout_fraction must be in (0, 0.6)")
    if feature_skew < 0:
        raise ValueError("feature_skew must be >= 0")

    rng = np.random.default_rng(derive_seed(seed, "partition", alpha, len(silo_ids)))
    shuffled = population.sample(frac=1.0, random_state=int(rng.integers(0, 2**31 - 1)))
    n_holdout = round(len(shuffled) * holdout_fraction)
    holdout = shuffled.iloc[:n_holdout].reset_index(drop=True)
    pool = shuffled.iloc[n_holdout:].reset_index(drop=True)

    assignment = _assign_dirichlet(pool, len(silo_ids), alpha, rng, min_rows_per_silo)
    frames: dict[str, pd.DataFrame] = {}
    for index, silo in enumerate(silo_ids):
        rows = pool.iloc[assignment == index].reset_index(drop=True)
        frames[silo] = validate_borrowers(_apply_feature_skew(rows, rng, feature_skew))

    fingerprint = _fingerprint(frames, holdout, alpha)
    return SiloPartition(
        silo_ids=tuple(silo_ids),
        frames=frames,
        holdout=holdout.reset_index(drop=True),
        alpha=alpha,
        fingerprint=fingerprint,
    )


def _assign_dirichlet(
    pool: pd.DataFrame,
    n_silos: int,
    alpha: float,
    rng: np.random.Generator,
    min_rows_per_silo: int,
) -> IntArray:
    """Assign each row to a silo, redrawing until every silo clears the minimum size.

    Small ``alpha`` can leave a silo empty, which is a legitimate Dirichlet draw but not a
    usable simulation, so the draw is repeated a bounded number of times and then the smallest
    silos are topped up from the largest.
    """
    labels = pool[LABEL_COLUMN].to_numpy(dtype=np.int64)
    assignment = np.full(len(pool), -1, dtype=np.int64)
    for _ in range(20):
        for label in np.unique(labels):
            indices = np.flatnonzero(labels == label)
            rng.shuffle(indices)
            proportions = dirichlet_proportions(rng, n_silos, alpha)
            cuts = (np.cumsum(proportions)[:-1] * len(indices)).astype(np.int64)
            for silo, chunk in enumerate(np.split(indices, cuts)):
                assignment[chunk] = silo
        counts = np.bincount(assignment, minlength=n_silos)
        if counts.min() >= min_rows_per_silo and _all_silos_have_both_classes(
            assignment, labels, n_silos
        ):
            return assignment
    return _top_up(assignment, labels, n_silos, min_rows_per_silo)


def _all_silos_have_both_classes(assignment: IntArray, labels: IntArray, n_silos: int) -> bool:
    return all(len(np.unique(labels[assignment == silo])) == 2 for silo in range(n_silos))


def _top_up(
    assignment: IntArray, labels: IntArray, n_silos: int, min_rows_per_silo: int
) -> IntArray:
    """Move rows from the largest silos until every silo is viable (both classes, min size)."""
    result = assignment.copy()
    for silo in range(n_silos):
        for label in (0, 1):
            while np.sum((result == silo) & (labels == label)) < max(1, min_rows_per_silo // 4):
                donors = [
                    other
                    for other in range(n_silos)
                    if other != silo
                    and np.sum((result == other) & (labels == label)) > min_rows_per_silo // 2
                ]
                if not donors:
                    raise ValueError(
                        "the population is too small to give every silo both classes; "
                        "increase n_rows or decrease the number of silos"
                    )
                largest = max(
                    donors, key=lambda other: int(np.sum((result == other) & (labels == label)))
                )
                candidates = np.flatnonzero((result == largest) & (labels == label))
                result[candidates[0]] = silo
    return result


def _apply_feature_skew(
    frame: pd.DataFrame, rng: np.random.Generator, feature_skew: float
) -> pd.DataFrame:
    """Shift a silo's continuous features by a small multiple of their own spread."""
    if feature_skew == 0.0 or frame.empty:
        return frame
    shifted = frame.copy()
    shifts = rng.normal(0.0, feature_skew, size=len(FEATURE_COLUMNS))
    bounds = {
        "age_years": (18.0, 100.0),
        "log_income": (7.0, 16.0),
        "debt_to_income": (0.0, 5.0),
        "revolving_utilization": (0.0, 2.0),
        "delinquencies_2y": (0.0, 20.0),
        "credit_age_years": (0.0, 60.0),
        "loan_to_income": (0.0, 10.0),
        "employment_years": (0.0, 50.0),
        "inquiries_6m": (0.0, 20.0),
        "open_accounts": (0.0, 40.0),
        "region_risk": (0.0, 1.0),
    }
    for column, shift in zip(FEATURE_COLUMNS, shifts, strict=True):
        if column not in bounds:  # prior_default is binary and left alone
            continue
        low, high = bounds[column]
        values = shifted[column].to_numpy(dtype=np.float64)
        spread = float(values.std()) or 1.0
        shifted[column] = np.clip(values + shift * spread, low, high)
    # The credit-age check must still hold after shifting.
    shifted["credit_age_years"] = np.minimum(
        shifted["credit_age_years"].to_numpy(dtype=np.float64),
        shifted["age_years"].to_numpy(dtype=np.float64) - 18.0,
    ).clip(0.0, 60.0)
    shifted["employment_years"] = np.minimum(
        shifted["employment_years"].to_numpy(dtype=np.float64),
        shifted["age_years"].to_numpy(dtype=np.float64) - 18.0,
    ).clip(0.0, 50.0)
    return shifted


def _fingerprint(frames: Mapping[str, pd.DataFrame], holdout: pd.DataFrame, alpha: float) -> str:
    digest = hashlib.sha256()
    digest.update(f"alpha={alpha:.6f}".encode())
    for silo in sorted(frames):
        ids = ",".join(frames[silo]["borrower_id"].tolist())
        digest.update(f"{silo}:{ids}".encode())
    digest.update(("holdout:" + ",".join(holdout["borrower_id"].tolist())).encode())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class Standardizer:
    """Feature centering and scaling agreed before training, never refitted per silo.

    In a real deployment these constants come from a public schema or a separate agreement
    round; refitting them per silo would leak each silo's marginals into the shared model and
    silently break the comparison between silos.
    """

    mean: FloatArray
    scale: FloatArray

    def transform(self, features: FloatArray) -> FloatArray:
        return (np.asarray(features, dtype=np.float64) - self.mean) / self.scale


def standardize(population: pd.DataFrame) -> Standardizer:
    """Fit the shared standardizer on the full population (a public, pre-agreed constant)."""
    values = population.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    scale = values.std(axis=0)
    scale[scale == 0.0] = 1.0
    return Standardizer(mean=values.mean(axis=0), scale=scale)
