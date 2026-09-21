"""The one-shot canary audit, wired end to end at a size that fits in a unit test.

``fl/audit_run.py`` is the repo's headline secondary pattern and had no test at all: it was
reached only by ``bench/bench_audit.py``, which CI runs but ``tasks.py test`` does not. So the
most novel module in the tree sat at 0% in the coverage report, and a refactor that broke the
wiring would have surfaced in a bench run rather than in the suite.

These tests use a much smaller configuration than the bench (few rounds, a small hidden layer,
a handful of canaries) -- enough to prove the pipeline runs and that the audit still separates
a correct implementation from a grossly broken one. The bench remains the source of the
published numbers; this is the regression net under the wiring.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from federated_secure_aggregation.fl.audit_run import AuditRunConfig, run_privacy_audit
from federated_secure_aggregation.fl.experiment import Experiment, build_experiment
from federated_secure_aggregation.privacy.noise import MechanismVariant

pytestmark = pytest.mark.unit

SEED = 20260920
FAST = AuditRunConfig(
    rounds=6,
    noise_multiplier=4.0,
    hidden=8,
    n_canaries=10,
    n_reference=128,
    seed=SEED,
)


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return build_experiment(n_rows=900, n_silos=5, alpha=1.0, seed=SEED)


def test_a_correct_implementation_is_not_flagged(experiment: Experiment) -> None:
    """A false positive here would make the audit worthless as evidence."""
    outcome = run_privacy_audit(experiment, FAST, MechanismVariant.CORRECT)
    assert outcome.variant is MechanismVariant.CORRECT
    assert outcome.flagged is False
    assert outcome.rounds_applied == FAST.rounds
    assert outcome.dimension > 0
    # The audit is a lower bound on privacy loss, so the empirical mu must not exceed the
    # theoretical one. Over-estimating would mean the estimator, not the mechanism, is broken.
    assert outcome.result.mu <= outcome.theoretical_mu
    assert outcome.result.mu_low <= outcome.result.mu
    assert outcome.accountant_epsilon > 0.0


def test_weak_noise_is_flagged(experiment: Experiment) -> None:
    """Noise out by a factor of ten is the audit's easiest case; missing it means no power."""
    outcome = run_privacy_audit(experiment, FAST, MechanismVariant.WEAK_NOISE)
    assert outcome.flagged is True
    assert outcome.result.epsilon_low > outcome.accountant_epsilon


def test_a_broken_variant_estimates_a_larger_mu_than_the_correct_one(
    experiment: Experiment,
) -> None:
    """The separation, not just the flag: the estimate has to move in the right direction."""
    correct = run_privacy_audit(experiment, FAST, MechanismVariant.CORRECT)
    broken = run_privacy_audit(experiment, FAST, MechanismVariant.WEAK_NOISE)
    assert broken.result.mu > correct.result.mu


def test_the_row_carries_what_the_bench_publishes(experiment: Experiment) -> None:
    outcome = run_privacy_audit(experiment, FAST, MechanismVariant.CORRECT)
    row = outcome.row()
    assert row["variant"] == str(MechanismVariant.CORRECT)
    assert row["flagged"] is False
    for key in ("mu_hat", "mu_low", "theoretical_mu", "epsilon_hat", "epsilon_low"):
        assert isinstance(row[key], float)


def test_the_config_refuses_settings_that_cannot_produce_a_usable_interval() -> None:
    with pytest.raises(ValueError, match="canary_scale"):
        replace(FAST, canary_scale=0.5)
    with pytest.raises(ValueError, match="at least 8 canaries"):
        replace(FAST, n_canaries=4)
