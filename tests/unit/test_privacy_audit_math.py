"""The auditor's statistics: the Gaussian mechanism inversion, canaries and the estimator.

The estimator turns two samples of cosines into an epsilon, so the tests pin down the pieces
independently: the analytic Gaussian mechanism's delta/epsilon relationship against known
values and its own inverse, the canary directions' geometry, and the estimator's behaviour on
synthetic data whose true separation is known.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.privacy.audit import (
    CanaryBank,
    audit_epsilon,
    epsilon_from_mu,
    gaussian_mechanism_delta,
)

pytestmark = pytest.mark.unit


def test_gaussian_delta_is_zero_for_a_zero_mu_and_decreasing_in_epsilon() -> None:
    assert gaussian_mechanism_delta(0.0, 1.0) == 0.0
    values = [gaussian_mechanism_delta(2.0, eps) for eps in (0.1, 0.5, 1.0, 2.0, 5.0)]
    assert all(later < earlier for earlier, later in itertools.pairwise(values))


@given(
    mu=st.floats(min_value=0.05, max_value=12.0),
    delta=st.floats(min_value=1e-8, max_value=1e-2),
)
@settings(max_examples=40)
def test_epsilon_from_mu_inverts_the_delta_function(mu: float, delta: float) -> None:
    epsilon = epsilon_from_mu(mu, delta)
    assert epsilon >= 0.0
    assert gaussian_mechanism_delta(mu, epsilon) == pytest.approx(delta, rel=1e-4, abs=1e-10)


@given(small=st.floats(min_value=0.1, max_value=3.0), extra=st.floats(min_value=0.1, max_value=4.0))
@settings(max_examples=25)
def test_epsilon_is_increasing_in_mu(small: float, extra: float) -> None:
    assert epsilon_from_mu(small, 1e-5) < epsilon_from_mu(small + extra, 1e-5)


def test_epsilon_from_mu_handles_degenerate_inputs() -> None:
    assert epsilon_from_mu(0.0, 1e-5) == 0.0
    assert epsilon_from_mu(-1.0, 1e-5) == 0.0
    with pytest.raises(ValueError, match="delta"):
        epsilon_from_mu(1.0, 0.0)


def test_canary_directions_are_unit_vectors_and_nearly_orthogonal() -> None:
    bank = CanaryBank.create(512, n_canaries=32, n_reference=64, seed=3)
    assert bank.dimension == 512
    assert np.allclose(np.linalg.norm(bank.inserted, axis=1), 1.0)
    assert np.allclose(np.linalg.norm(bank.reference, axis=1), 1.0)
    gram = bank.inserted @ bank.inserted.T
    off_diagonal = gram[~np.eye(32, dtype=bool)]
    assert float(np.max(np.abs(off_diagonal))) < 0.35  # ~4 sigma at d = 512


def test_canary_bank_validates_its_shape() -> None:
    with pytest.raises(ValueError, match="at least 8 parameters"):
        CanaryBank.create(4, n_canaries=2, n_reference=2, seed=0)
    with pytest.raises(ValueError, match="at least two"):
        CanaryBank.create(64, n_canaries=1, n_reference=4, seed=0)
    with pytest.raises(ValueError, match="near-orthogonal"):
        CanaryBank.create(16, n_canaries=16, n_reference=4, seed=0)


def test_cosines_check_the_delta_and_reject_a_model_that_did_not_move() -> None:
    bank = CanaryBank.create(64, n_canaries=8, n_reference=16, seed=1)
    with pytest.raises(ValueError, match="dimension"):
        bank.cosines(np.zeros(32))
    with pytest.raises(ValueError, match="did not move"):
        bank.cosines(np.zeros(64))
    observed, unobserved = bank.cosines(bank.direction(0) * 3.0)
    assert observed[0] == pytest.approx(1.0)
    assert unobserved.size == 16


def test_the_estimator_recovers_a_known_separation() -> None:
    """Two Gaussian samples separated by exactly 2 standard deviations must estimate mu = 2."""
    rng = np.random.default_rng(0)
    unobserved = rng.normal(0.0, 1.0, size=20_000)
    observed = rng.normal(2.0, 1.0, size=20_000)
    result = audit_epsilon(observed, unobserved, delta=1e-5, claimed_epsilon=100.0)
    assert result.mu == pytest.approx(2.0, abs=0.05)
    assert result.mu_low < result.mu < result.mu_high
    assert result.epsilon_low < result.epsilon < result.epsilon_high
    assert result.consistent_with_claim
    assert not result.exceeds_claim


def test_the_estimator_flags_a_claim_it_exceeds() -> None:
    rng = np.random.default_rng(1)
    unobserved = rng.normal(0.0, 1.0, size=5_000)
    observed = rng.normal(4.0, 1.0, size=5_000)
    result = audit_epsilon(observed, unobserved, delta=1e-5, claimed_epsilon=1.0)
    assert result.exceeds_claim
    summary = result.summary()
    assert summary["exceeds_claim"] is True
    assert summary["claimed_epsilon"] == 1.0
    assert summary["n_canaries"] == 5_000


def test_no_separation_estimates_a_mu_near_zero() -> None:
    rng = np.random.default_rng(2)
    unobserved = rng.normal(0.0, 1.0, size=10_000)
    observed = rng.normal(0.0, 1.0, size=10_000)
    result = audit_epsilon(observed, unobserved, delta=1e-5, claimed_epsilon=1.0)
    assert abs(result.mu) < 0.1
    assert result.mu_low == 0.0
    assert result.epsilon_low == 0.0
    assert not result.exceeds_claim


def test_the_estimator_validates_its_inputs() -> None:
    sample = np.zeros(5) + np.arange(5)
    with pytest.raises(ValueError, match="at least two values"):
        audit_epsilon([1.0], sample, delta=1e-5, claimed_epsilon=1.0)
    with pytest.raises(ValueError, match="confidence"):
        audit_epsilon(sample, sample, delta=1e-5, claimed_epsilon=1.0, confidence=0.4)
    with pytest.raises(ValueError, match="claimed_epsilon"):
        audit_epsilon(sample, sample, delta=1e-5, claimed_epsilon=0.0)
    with pytest.raises(ValueError, match="zero spread"):
        audit_epsilon(sample, np.ones(5), delta=1e-5, claimed_epsilon=1.0)


def test_a_wider_confidence_level_widens_the_interval() -> None:
    rng = np.random.default_rng(3)
    unobserved = rng.normal(0.0, 1.0, size=500)
    observed = rng.normal(1.5, 1.0, size=500)
    narrow = audit_epsilon(observed, unobserved, delta=1e-5, claimed_epsilon=10.0, confidence=0.8)
    wide = audit_epsilon(observed, unobserved, delta=1e-5, claimed_epsilon=10.0, confidence=0.99)
    assert wide.mu_high > narrow.mu_high
    assert wide.mu_low < narrow.mu_low
