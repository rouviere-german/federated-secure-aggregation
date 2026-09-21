"""The budget ledger and the central-DP mechanism, including the deliberately broken variants."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from federated_secure_aggregation.privacy.accountant import calibrate_noise_multiplier
from federated_secure_aggregation.privacy.ledger import (
    BudgetExhaustedError,
    BudgetLedger,
    BudgetState,
)
from federated_secure_aggregation.privacy.noise import (
    MechanismVariant,
    add_central_noise,
    clip_update,
    noise_std,
)

pytestmark = pytest.mark.unit


def test_the_ledger_refuses_the_round_that_would_break_the_budget() -> None:
    multiplier = calibrate_noise_multiplier(
        target_epsilon=4.0, delta=1e-5, steps=10, sample_rate=1.0
    )
    ledger = BudgetLedger(noise_multiplier=multiplier, target_epsilon=4.0, delta=1e-5)
    assert ledger.remaining_rounds() >= 10
    spent = 0.0
    rounds = 0
    while ledger.can_run_round():
        spent = ledger.charge_round()
        rounds += 1
        assert rounds < 1_000
    assert rounds >= 10
    assert spent <= 4.0
    assert ledger.epsilon_after_next() > 4.0
    with pytest.raises(BudgetExhaustedError) as info:
        ledger.charge_round()
    assert info.value.target == 4.0
    assert info.value.would_be > 4.0
    assert ledger.remaining_rounds() == 0


def test_the_ledger_survives_a_restart(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    first = BudgetLedger(noise_multiplier=2.0, target_epsilon=40.0, delta=1e-5, path=path)
    for _ in range(3):
        first.charge_round()
    assert first.steps == 3

    second = BudgetLedger(noise_multiplier=2.0, target_epsilon=40.0, delta=1e-5, path=path)
    assert second.steps == 3
    assert second.epsilon_spent() == pytest.approx(first.epsilon_spent())
    second.charge_round()
    assert second.steps == 4

    third = BudgetLedger(noise_multiplier=2.0, target_epsilon=40.0, delta=1e-5, path=path)
    assert third.steps == 4


def test_a_ledger_written_with_other_parameters_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "budget.json"
    BudgetLedger(noise_multiplier=2.0, target_epsilon=40.0, delta=1e-5, path=path).charge_round()
    with pytest.raises(ValueError, match="different mechanism parameters"):
        BudgetLedger(noise_multiplier=3.0, target_epsilon=40.0, delta=1e-5, path=path)


def test_ledger_state_round_trips_and_rejects_unknown_schemas(tmp_path: Path) -> None:
    state = BudgetState(
        steps=2, noise_multiplier=2.0, sample_rate=1.0, target_epsilon=8.0, delta=1e-5
    )
    assert BudgetState.from_json(state.to_json()) == state
    with pytest.raises(ValueError, match="schema version"):
        BudgetState.from_json(json.dumps({"schema_version": 2}))


def test_ledger_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="target_epsilon"):
        BudgetLedger(noise_multiplier=1.0, target_epsilon=0.0, delta=1e-5)
    with pytest.raises(ValueError, match="delta"):
        BudgetLedger(noise_multiplier=1.0, target_epsilon=1.0, delta=1.0)


def test_clipping_bounds_the_update_except_in_the_broken_variant() -> None:
    long = np.array([3.0, 4.0])
    clipped = clip_update(long, 1.0, MechanismVariant.CORRECT)
    assert float(np.linalg.norm(clipped)) == pytest.approx(1.0)
    unclipped = clip_update(long, 1.0, MechanismVariant.NO_CLIPPING)
    assert np.array_equal(unclipped, long)
    short = np.array([0.1, 0.1])
    assert np.array_equal(clip_update(short, 1.0, MechanismVariant.CORRECT), short)
    assert np.array_equal(clip_update(np.zeros(2), 1.0, MechanismVariant.CORRECT), np.zeros(2))
    with pytest.raises(ValueError, match="clip_norm"):
        clip_update(long, 0.0, MechanismVariant.CORRECT)
    with pytest.raises(ValueError, match="NaN"):
        clip_update(np.array([float("nan")]), 1.0, MechanismVariant.CORRECT)


def test_noise_scale_follows_the_variant() -> None:
    common: dict[str, Any] = {"noise_multiplier": 2.0, "clip_norm": 3.0, "survivors": 5}
    assert noise_std(**common, variant=MechanismVariant.CORRECT) == pytest.approx(6.0)
    assert noise_std(**common, variant=MechanismVariant.WEAK_NOISE) == pytest.approx(0.6)
    assert noise_std(**common, variant=MechanismVariant.AVERAGE_SENSITIVITY) == pytest.approx(1.2)
    assert noise_std(**common, variant=MechanismVariant.CLIP_AFTER_NOISE) == pytest.approx(6.0)
    with pytest.raises(ValueError, match="noise_multiplier"):
        noise_std(noise_multiplier=-1, clip_norm=1, survivors=1, variant=MechanismVariant.CORRECT)
    with pytest.raises(ValueError, match="clip_norm"):
        noise_std(noise_multiplier=1, clip_norm=0, survivors=1, variant=MechanismVariant.CORRECT)
    with pytest.raises(ValueError, match="survivors"):
        noise_std(noise_multiplier=1, clip_norm=1, survivors=0, variant=MechanismVariant.CORRECT)


def test_only_three_variants_are_classified_as_breaking_the_guarantee() -> None:
    broken = {variant for variant in MechanismVariant if variant.is_broken}
    assert broken == {
        MechanismVariant.NO_CLIPPING,
        MechanismVariant.WEAK_NOISE,
        MechanismVariant.AVERAGE_SENSITIVITY,
    }


def test_added_noise_has_the_calibrated_scale() -> None:
    rng = np.random.default_rng(0)
    aggregate = np.zeros(50_000)
    noised = add_central_noise(aggregate, rng, noise_multiplier=2.0, clip_norm=1.5, survivors=5)
    assert float(np.std(noised)) == pytest.approx(3.0, rel=0.05)


def test_zero_noise_multiplier_is_the_epsilon_infinity_reference() -> None:
    rng = np.random.default_rng(0)
    aggregate = np.array([1.0, -2.0, 3.0])
    noised = add_central_noise(aggregate, rng, noise_multiplier=0.0, clip_norm=1.0, survivors=3)
    assert np.array_equal(noised, aggregate)


def test_clip_after_noise_only_rescales_and_so_preserves_direction() -> None:
    """Clipping a private output is post-processing: it cannot change what was released."""
    rng_a = np.random.default_rng(7)
    rng_b = np.random.default_rng(7)
    aggregate = np.ones(2_000) * 0.01
    plain = add_central_noise(aggregate, rng_a, noise_multiplier=2.0, clip_norm=1.0, survivors=5)
    clipped = add_central_noise(
        aggregate,
        rng_b,
        noise_multiplier=2.0,
        clip_norm=1.0,
        survivors=5,
        variant=MechanismVariant.CLIP_AFTER_NOISE,
    )
    assert float(np.linalg.norm(clipped)) < float(np.linalg.norm(plain))
    cosine = float(plain @ clipped / (np.linalg.norm(plain) * np.linalg.norm(clipped)))
    assert cosine == pytest.approx(1.0, abs=1e-12)
