"""Evaluation metrics, the synthetic population and its non-IID partition.

``roc_auc`` is compared against scikit-learn on random inputs rather than on a handful of fixed
cases, because the rank-with-ties logic is exactly where a hand-written AUC usually goes wrong.
The partition tests pin the reproducibility claims: the same seed must give the same split, the
holdout must never appear in a silo, and every silo must satisfy the data contract.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.metrics import log_loss as sk_log_loss
from sklearn.metrics import roc_auc_score

from federated_secure_aggregation.evaluation.metrics import (
    accuracy,
    brier_score,
    log_loss,
    roc_auc,
)
from federated_secure_aggregation.fl.data import (
    SiloPartition,
    derive_seed,
    dirichlet_proportions,
    make_population,
    partition_dirichlet,
    standardize,
)
from federated_secure_aggregation.fl.experiment import build_experiment, silo_ids
from federated_secure_aggregation.ml.contracts import (
    LABEL_COLUMN,
    DataContractError,
    feature_matrix,
    label_vector,
    validate_borrowers,
)

pytestmark = pytest.mark.unit


@given(
    n=st.integers(min_value=10, max_value=300),
    seed=st.integers(min_value=0, max_value=10_000),
    ties=st.booleans(),
)
@settings(max_examples=40)
def test_roc_auc_matches_scikit_learn(n: int, seed: int, ties: bool) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.normal(size=n)
    if ties:
        scores = np.round(scores, 1)  # force many exact ties
    labels = (rng.random(n) < 0.4).astype(np.float64)
    if labels.min() == labels.max():
        labels[0] = 1.0 - labels[0]
    assert roc_auc(scores, labels) == pytest.approx(float(roc_auc_score(labels, scores)))


@given(n=st.integers(min_value=5, max_value=200), seed=st.integers(min_value=0, max_value=10_000))
@settings(max_examples=25)
def test_log_loss_matches_scikit_learn(n: int, seed: int) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.01, 0.99, size=n)
    labels = (rng.random(n) < 0.5).astype(np.float64)
    if labels.min() == labels.max():
        labels[0] = 1.0 - labels[0]
    assert log_loss(scores, labels) == pytest.approx(float(sk_log_loss(labels, scores)), rel=1e-9)


def test_perfect_and_inverted_rankings() -> None:
    labels = np.array([0.0, 0.0, 1.0, 1.0])
    assert roc_auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert roc_auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert roc_auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5


def test_metrics_validate_their_inputs() -> None:
    labels = np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="same length"):
        roc_auc(np.array([0.5]), labels)
    with pytest.raises(ValueError, match="at least one observation"):
        roc_auc(np.array([]), np.array([]))
    with pytest.raises(ValueError, match="NaN"):
        roc_auc(np.array([float("nan"), 0.5]), labels)
    with pytest.raises(ValueError, match="labels must be"):
        roc_auc(np.array([0.1, 0.2]), np.array([0.0, 2.0]))
    with pytest.raises(ValueError, match="both classes"):
        roc_auc(np.array([0.1, 0.2]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError, match="probabilities"):
        log_loss(np.array([1.5, 0.2]), labels)


def test_accuracy_and_brier() -> None:
    labels = np.array([0.0, 1.0, 1.0, 0.0])
    scores = np.array([0.1, 0.9, 0.4, 0.2])
    assert accuracy(scores, labels) == 0.75
    assert brier_score(scores, labels) == pytest.approx(float(np.mean((scores - labels) ** 2)))


def test_the_population_satisfies_its_own_contract_and_base_rate() -> None:
    population = make_population(3_000, seed=5, base_rate=0.2)
    validate_borrowers(population)
    assert float(population[LABEL_COLUMN].mean()) == pytest.approx(0.2, abs=0.03)
    assert population["borrower_id"].is_unique
    assert make_population(500, seed=5).equals(make_population(500, seed=5))
    assert not make_population(500, seed=5).equals(make_population(500, seed=6))


def test_population_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="n_rows"):
        make_population(5, seed=0)
    with pytest.raises(ValueError, match="base_rate"):
        make_population(100, seed=0, base_rate=0.8)


def test_contract_reports_column_check_and_count_without_values() -> None:
    population = make_population(200, seed=1)
    broken = population.copy()
    broken.loc[0, "debt_to_income"] = -3.0
    broken.loc[1, LABEL_COLUMN] = 7
    with pytest.raises(DataContractError) as info:
        validate_borrowers(broken)
    failures = info.value.failures
    assert {str(item["column"]) for item in failures} >= {"debt_to_income", LABEL_COLUMN}
    assert "-3.0" not in str(info.value)
    assert all(set(item) == {"column", "check", "count"} for item in failures)


def test_contract_rejects_an_impossible_credit_history() -> None:
    population = make_population(200, seed=2)
    broken = population.copy()
    broken.loc[0, "credit_age_years"] = broken.loc[0, "age_years"]
    with pytest.raises(DataContractError):
        validate_borrowers(broken)


def test_contract_rejects_a_single_class_silo() -> None:
    population = make_population(200, seed=3)
    broken = population.copy()
    broken[LABEL_COLUMN] = 0
    with pytest.raises(DataContractError):
        validate_borrowers(broken)


def test_feature_and_label_accessors_report_missing_columns() -> None:
    population = make_population(50, seed=4)
    assert next(iter(feature_matrix(population).columns)) == "age_years"
    assert label_vector(population).dtype == np.int64
    with pytest.raises(DataContractError):
        feature_matrix(population.drop(columns=["age_years"]))
    with pytest.raises(DataContractError):
        label_vector(population.drop(columns=[LABEL_COLUMN]))


def test_partition_is_reproducible_disjoint_and_contract_clean() -> None:
    population = make_population(3_000, seed=11)
    ids = silo_ids(5)
    first = partition_dirichlet(population, ids, alpha=1.0, seed=11)
    second = partition_dirichlet(population, ids, alpha=1.0, seed=11)
    assert first.fingerprint == second.fingerprint
    other = partition_dirichlet(population, ids, alpha=1.0, seed=12)
    assert other.fingerprint != first.fingerprint

    holdout_ids = set(first.holdout["borrower_id"])
    seen: set[str] = set()
    for silo in ids:
        frame = first.frames[silo]
        validate_borrowers(frame)
        frame_ids = set(frame["borrower_id"])
        assert not (frame_ids & holdout_ids)
        assert not (frame_ids & seen)
        seen |= frame_ids
    assert len(seen) + len(holdout_ids) == len(population)


def test_smaller_alpha_produces_more_label_skew() -> None:
    population = make_population(6_000, seed=13)
    ids = silo_ids(6)
    skewed = partition_dirichlet(population, ids, alpha=0.1, seed=13)
    mild = partition_dirichlet(population, ids, alpha=10.0, seed=13)

    def spread(part: SiloPartition) -> float:
        return float(np.std(list(part.label_rates().values())))

    assert spread(skewed) > spread(mild)
    assert sum(mild.sizes().values()) == sum(skewed.sizes().values())


def test_partition_validates_its_arguments() -> None:
    population = make_population(500, seed=14)
    with pytest.raises(ValueError, match="at least two silos"):
        partition_dirichlet(population, ["only"], alpha=1.0, seed=0)
    with pytest.raises(ValueError, match="unique"):
        partition_dirichlet(population, ["a", "a"], alpha=1.0, seed=0)
    with pytest.raises(ValueError, match="holdout_fraction"):
        partition_dirichlet(population, ["a", "b"], alpha=1.0, seed=0, holdout_fraction=0.9)
    with pytest.raises(ValueError, match="feature_skew"):
        partition_dirichlet(population, ["a", "b"], alpha=1.0, seed=0, feature_skew=-1)
    with pytest.raises(ValueError, match="too small"):
        partition_dirichlet(
            make_population(60, seed=1), silo_ids(10), alpha=0.01, seed=0, min_rows_per_silo=40
        )


def test_dirichlet_proportions_validate_and_sum_to_one() -> None:
    rng = np.random.default_rng(0)
    proportions = dirichlet_proportions(rng, 5, 1.0)
    assert proportions.shape == (5,)
    assert float(proportions.sum()) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="n_silos"):
        dirichlet_proportions(rng, 0, 1.0)
    with pytest.raises(ValueError, match="alpha"):
        dirichlet_proportions(rng, 3, 0.0)


def test_derive_seed_is_stable_and_path_sensitive() -> None:
    assert derive_seed(7, "a", 1) == derive_seed(7, "a", 1)
    assert derive_seed(7, "a", 1) != derive_seed(7, "a", 2)
    assert derive_seed(7, "a", 1) != derive_seed(8, "a", 1)


def test_standardizer_centres_and_scales_the_population() -> None:
    population = make_population(2_000, seed=15)
    standardizer = standardize(population)
    from federated_secure_aggregation.ml.contracts import FEATURE_COLUMNS

    values = population.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype=np.float64)
    transformed = standardizer.transform(values)
    assert np.allclose(transformed.mean(axis=0), 0.0, atol=1e-9)
    assert np.allclose(transformed.std(axis=0), 1.0, atol=1e-9)


def test_experiment_builds_matching_matrices_and_a_manifest() -> None:
    experiment = build_experiment(n_rows=1_500, n_silos=4, alpha=1.0, seed=17)
    assert experiment.silos == silo_ids(4)
    pooled_features, pooled_labels = experiment.pooled()
    assert pooled_features.shape[1] == experiment.n_features
    assert pooled_features.shape[0] == pooled_labels.size
    assert experiment.holdout_features.shape[1] == experiment.n_features
    assert set(experiment.silo_data()) == set(experiment.silos)

    manifest = experiment.manifest(params={"rounds": 3}, metrics={"auc": 0.8})
    assert manifest.params["partition_fingerprint"] == experiment.partition.fingerprint
    assert manifest.metrics["auc"] == 0.8
    assert manifest.packages["numpy"] != "not-installed"


def test_silo_ids_are_sorted_and_protocol_legal() -> None:
    ids = silo_ids(12)
    assert ids == tuple(sorted(ids))
    assert all(name.startswith("silo_") for name in ids)
    with pytest.raises(ValueError, match="count"):
        silo_ids(0)
