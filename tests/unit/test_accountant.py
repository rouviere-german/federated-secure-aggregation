"""The RDP accountant: closed-form cases, monotonicity and the budget conversion.

The accountant is the number the system publishes, so the properties tested here are the ones a
reader would want to hold before believing it: more noise must never cost more privacy, more
rounds must never cost less, full participation must reproduce the textbook Gaussian formula,
and calibration must land on the requested epsilon rather than near it.

``tests/unit/test_accountant_reference.py`` additionally compares every value against Google's
``dp-accounting``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.privacy.accountant import (
    DEFAULT_ORDERS,
    PrivacyAccountingError,
    RDPAccountant,
    calibrate_noise_multiplier,
    compute_epsilon,
    rdp_subsampled_gaussian,
)

pytestmark = pytest.mark.unit

MULTIPLIERS = st.floats(min_value=0.4, max_value=20.0, allow_nan=False, allow_infinity=False)
RATES = st.floats(min_value=0.01, max_value=1.0, allow_nan=False, allow_infinity=False)


def test_full_participation_matches_the_gaussian_closed_form() -> None:
    orders = (2, 3, 8, 32)
    for multiplier in (0.5, 1.0, 4.0):
        values = rdp_subsampled_gaussian(
            sample_rate=1.0, noise_multiplier=multiplier, orders=orders
        )
        expected = [alpha / (2 * multiplier**2) for alpha in orders]
        assert np.allclose(values, expected)


@given(multiplier=MULTIPLIERS, rate=RATES)
@settings(max_examples=40)
def test_rdp_is_positive_and_decreasing_in_the_noise_multiplier(
    multiplier: float, rate: float
) -> None:
    orders = (2, 4, 16, 64)
    low = rdp_subsampled_gaussian(sample_rate=rate, noise_multiplier=multiplier, orders=orders)
    high = rdp_subsampled_gaussian(
        sample_rate=rate, noise_multiplier=multiplier * 1.5, orders=orders
    )
    assert np.all(low > 0)
    assert np.all(high <= low + 1e-12)


@given(rate=RATES)
@settings(max_examples=25)
def test_subsampling_never_costs_more_than_full_participation(rate: float) -> None:
    orders = (2, 4, 16)
    subsampled = rdp_subsampled_gaussian(sample_rate=rate, noise_multiplier=2.0, orders=orders)
    full = rdp_subsampled_gaussian(sample_rate=1.0, noise_multiplier=2.0, orders=orders)
    assert np.all(subsampled <= full + 1e-12)


def test_epsilon_grows_with_rounds_and_shrinks_with_noise() -> None:
    delta = 1e-5
    accountant = RDPAccountant(noise_multiplier=2.0)
    assert accountant.epsilon(delta) == 0.0
    previous = 0.0
    for _ in range(10):
        accountant.step()
        current = accountant.epsilon(delta)
        assert current > previous
        previous = current
    quieter = RDPAccountant(noise_multiplier=4.0, steps=accountant.steps)
    assert quieter.epsilon(delta) < accountant.epsilon(delta)


def test_extra_steps_answers_the_what_if_question() -> None:
    accountant = RDPAccountant(noise_multiplier=2.0, steps=5)
    after = RDPAccountant(noise_multiplier=2.0, steps=8)
    assert accountant.epsilon(1e-5, extra_steps=3) == pytest.approx(after.epsilon(1e-5))


def test_best_order_and_parameters_are_reported() -> None:
    accountant = RDPAccountant(noise_multiplier=2.0, sample_rate=0.5)
    assert accountant.best_order(1e-5) == DEFAULT_ORDERS[0]
    accountant.step(20)
    order = accountant.best_order(1e-5)
    assert order in DEFAULT_ORDERS
    parameters = accountant.parameters()
    assert parameters["steps"] == 20
    assert parameters["noise_multiplier"] == 2.0
    assert parameters["min_order"] == 2


@given(
    target=st.floats(min_value=0.5, max_value=12.0),
    steps=st.integers(min_value=1, max_value=60),
)
@settings(max_examples=20, deadline=None)
def test_calibration_lands_at_or_below_the_target(target: float, steps: int) -> None:
    multiplier = calibrate_noise_multiplier(
        target_epsilon=target, delta=1e-5, steps=steps, sample_rate=1.0
    )
    reached = RDPAccountant(noise_multiplier=multiplier, steps=steps).epsilon(1e-5)
    assert reached <= target + 1e-6
    # And it is not wastefully quiet: a slightly smaller multiplier would overshoot.
    looser = RDPAccountant(noise_multiplier=multiplier * 0.9, steps=steps).epsilon(1e-5)
    assert looser > reached


def test_validation_of_domains() -> None:
    with pytest.raises(PrivacyAccountingError, match="sample_rate"):
        rdp_subsampled_gaussian(sample_rate=0.0, noise_multiplier=1.0, orders=(2,))
    with pytest.raises(PrivacyAccountingError, match="sample_rate"):
        rdp_subsampled_gaussian(sample_rate=1.5, noise_multiplier=1.0, orders=(2,))
    with pytest.raises(PrivacyAccountingError, match="noise_multiplier"):
        rdp_subsampled_gaussian(sample_rate=0.5, noise_multiplier=0.0, orders=(2,))
    with pytest.raises(PrivacyAccountingError, match="at least one order"):
        rdp_subsampled_gaussian(sample_rate=0.5, noise_multiplier=1.0, orders=())
    with pytest.raises(PrivacyAccountingError, match="integers >= 2"):
        rdp_subsampled_gaussian(sample_rate=0.5, noise_multiplier=1.0, orders=(1,))
    with pytest.raises(PrivacyAccountingError, match="delta"):
        compute_epsilon(np.array([1.0]), (2,), 0.0)
    with pytest.raises(PrivacyAccountingError, match="same length"):
        compute_epsilon(np.array([1.0, 2.0]), (2,), 1e-5)
    with pytest.raises(PrivacyAccountingError, match="finite"):
        compute_epsilon(np.array([-1.0]), (2,), 1e-5)
    with pytest.raises(PrivacyAccountingError, match="steps"):
        RDPAccountant(noise_multiplier=1.0, steps=-1)
    with pytest.raises(PrivacyAccountingError, match="count"):
        RDPAccountant(noise_multiplier=1.0).step(0)
    with pytest.raises(PrivacyAccountingError, match="extra_steps"):
        RDPAccountant(noise_multiplier=1.0).epsilon(1e-5, extra_steps=-1)
    with pytest.raises(PrivacyAccountingError, match="target_epsilon"):
        calibrate_noise_multiplier(target_epsilon=0.0, delta=1e-5, steps=1)
    with pytest.raises(PrivacyAccountingError, match="steps"):
        calibrate_noise_multiplier(target_epsilon=1.0, delta=1e-5, steps=0)
    with pytest.raises(PrivacyAccountingError, match="no noise multiplier"):
        calibrate_noise_multiplier(
            target_epsilon=1e-6, delta=1e-12, steps=10_000, max_multiplier=2.0
        )


def test_the_rdp_curve_is_finite_for_a_wide_grid() -> None:
    values = rdp_subsampled_gaussian(sample_rate=0.001, noise_multiplier=0.5, orders=DEFAULT_ORDERS)
    assert np.all(np.isfinite(values))
    assert math.isfinite(float(np.max(values)))
