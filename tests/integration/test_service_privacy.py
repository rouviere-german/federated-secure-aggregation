"""The deployed coordinator must actually run the DP pipeline, not just claim to.

The HTTP service used to implement no differential privacy at all: ``epsilon_spent`` was the
literal ``0.0`` in the metrics call, no route returned an aggregate, and the budget ledger was
never consulted -- while the module docstring said "the only vector that leaves the service is
the noised aggregate" and the model card listed ``fedsecagg_epsilon_spent`` among the gauges
"exposed on the authenticated /metrics endpoint".

These tests pin the wiring end to end: noise is added to the summed update before anything
leaves, the aggregate goes only to the operator, the ledger is charged per accepted round and
refuses to open a round it cannot pay for, and the gauge carries the real number.
"""

from __future__ import annotations

import base64
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest

from federated_secure_aggregation.privacy.ledger import BudgetLedger
from federated_secure_aggregation.privacy.noise import noise_std
from federated_secure_aggregation.service.api import CentralPrivacy

from .test_service_api import AUTH, DIMENSION, SILO_KEYS, Harness

pytestmark = pytest.mark.integration

NOISE_MULTIPLIER = 1.0
CLIP_NORM = 1.0


def _privacy(target_epsilon: float = 8.0, seed: int = 7) -> CentralPrivacy:
    return CentralPrivacy(
        ledger=BudgetLedger(
            noise_multiplier=NOISE_MULTIPLIER, target_epsilon=target_epsilon, delta=1e-5
        ),
        clip_norm=CLIP_NORM,
        rng=np.random.default_rng(seed),
    )


@pytest.fixture
def private_harness() -> Iterator[Harness]:
    instance = Harness(privacy=_privacy())
    try:
        yield instance
    finally:
        instance.close()


@pytest.fixture
def plain_harness() -> Iterator[Harness]:
    """The same service with no CentralPrivacy: secure aggregation and nothing else."""
    instance = Harness()
    try:
        yield instance
    finally:
        instance.close()


def _release(harness: Harness, headers: dict[str, str] | None = None) -> Any:
    return harness.http.get("/v1/rounds/aggregate", headers=headers if headers else AUTH)


def test_the_released_aggregate_is_noised_and_the_ledger_is_charged(
    private_harness: Harness,
) -> None:
    """The operator receives the noised sum, and it is demonstrably not the raw one."""
    outcome, updates = private_harness.run_round(0)
    assert outcome["accepted"] is True

    epsilon = outcome["epsilon_spent"]
    assert epsilon is not None
    assert 0.0 < epsilon <= 8.0
    assert private_harness.service.privacy is not None
    assert private_harness.service.privacy.ledger.steps == 1

    response = _release(private_harness)
    assert response.status_code == 200
    body = response.json()
    assert body["round_id"] == 0
    assert body["epsilon_spent"] == pytest.approx(epsilon)
    assert body["noise_multiplier"] == NOISE_MULTIPLIER
    released = np.frombuffer(base64.b64decode(body["aggregate"]), dtype="<f8")
    assert released.size == DIMENSION

    truth = np.sum([updates[name] for name in private_harness.ids], axis=0)
    # Noise of this size is unmistakable: the released vector is nowhere near the true sum, but
    # it is still within a few sigma of it, so it is noise rather than a different vector.
    sigma = noise_std(
        noise_multiplier=NOISE_MULTIPLIER,
        clip_norm=CLIP_NORM,
        survivors=len(private_harness.ids),
        variant=private_harness.service.privacy.variant,
    )
    assert sigma > 0.0
    deviation = np.abs(released - truth)
    assert deviation.max() > 0.0
    assert deviation.max() < 6.0 * sigma
    assert not np.allclose(released, truth)


def test_the_aggregate_is_an_operator_route(private_harness: Harness) -> None:
    """A silo contributes to the sum; it does not get to read it."""
    private_harness.run_round(0)
    silo = {"X-API-Key": SILO_KEYS["silo_00"]}
    assert _release(private_harness, silo).status_code == 403
    assert _release(private_harness).status_code == 200


def test_no_aggregate_is_released_before_a_round_is_accepted(private_harness: Harness) -> None:
    assert _release(private_harness).status_code == 409
    assert _release(private_harness).json()["error"]["code"] == "not_available"
    private_harness.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    assert _release(private_harness).status_code == 409


def test_a_service_without_privacy_configured_releases_nothing(plain_harness: Harness) -> None:
    """Fail closed: no ledger means no release, rather than an unaccounted plaintext sum."""
    outcome, _ = plain_harness.run_round(0)
    assert outcome["accepted"] is True
    assert outcome["epsilon_spent"] is None
    response = _release(plain_harness)
    assert response.status_code == 409
    assert response.json()["error"]["code"] == "not_available"


def test_the_ledger_refuses_to_open_a_round_it_cannot_pay_for() -> None:
    """The budget stops training rather than quietly overspending the claimed epsilon.

    At noise_multiplier 1.0 and delta 1e-5 the accountant charges 4.7527 for the first round
    and 7.0879 cumulatively for the second, so a target of 8.0 buys exactly two rounds. The
    third must be refused before it opens rather than run and then discovered unaffordable.
    """
    instance = Harness(privacy=_privacy(target_epsilon=8.0))
    try:
        ledger = instance.service.privacy.ledger if instance.service.privacy else None
        assert ledger is not None
        rounds = 0
        while ledger.can_run_round():
            outcome, _ = instance.run_round(rounds)
            assert outcome["accepted"] is True
            rounds += 1
            assert rounds < 50, "the ledger never ran out; the test would not prove anything"
        assert rounds == 2, rounds
        refused = instance.http.post("/v1/rounds", json={"round_id": rounds}, headers=AUTH)
        assert refused.status_code == 409
        assert refused.json()["error"]["code"] == "budget_exhausted"
        assert ledger.epsilon_spent() == pytest.approx(7.0879, abs=1e-4)
    finally:
        instance.close()


def test_the_epsilon_gauge_carries_the_real_number(private_harness: Harness) -> None:
    """MODEL_CARD promises fedsecagg_epsilon_spent on /metrics; it must not be a constant 0."""
    outcome, _ = private_harness.run_round(0)
    scraped = private_harness.http.get("/metrics", headers=AUTH)
    assert scraped.status_code == 200
    line = next(
        row
        for row in scraped.text.splitlines()
        if row.startswith("fedsecagg_epsilon_spent") and not row.startswith("#")
    )
    assert float(line.split()[-1]) == pytest.approx(outcome["epsilon_spent"])
