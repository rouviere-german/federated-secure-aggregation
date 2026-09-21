"""The round state machine: stage order, quorum, timeouts, duplicate messages and the norm bound.

The coordinator is the part an attacker reaches first, so these tests are mostly about what it
*refuses*. Timeouts are driven through the injected clock, never by sleeping.
"""

from __future__ import annotations

import numpy as np
import pytest

from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.secagg.coordinator import (
    RoundConfig,
    RoundResult,
    SecAggCoordinator,
)
from federated_secure_aggregation.secagg.errors import (
    AggregateRejectedError,
    ProtocolStateError,
    QuorumError,
)
from federated_secure_aggregation.secagg.messages import (
    MessageKind,
    ProtocolMessageError,
    RoundStage,
    sign_message,
)
from federated_secure_aggregation.secagg.session import run_round
from tests.conftest import Federation, build_federation

pytestmark = pytest.mark.unit


def _advance(federation: Federation) -> None:
    federation.clock.skip_timeout(federation.config.stage_timeout_seconds + 1.0)


def test_a_clean_round_completes_and_reports_its_participants(federation: Federation) -> None:
    updates = federation.updates()
    result = run_round(
        federation.coordinator, federation.clients, updates, 0, clock=federation.clock
    )
    assert result.stage is RoundStage.COMPLETED
    assert result.accepted
    assert result.survivors == tuple(federation.ids)
    assert result.dropouts == ()
    assert result.bytes_received > 0
    assert len(result.digest()) == 64


def test_messages_are_refused_outside_their_stage(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    client = federation.clients["silo_00"]
    advertisement = client.advertise_keys(0, nonce)
    coordinator.submit_key_advertisement(advertisement)

    with pytest.raises(ProtocolStateError, match="already advertised"):
        coordinator.submit_key_advertisement(advertisement)
    with pytest.raises(ProtocolStateError, match="not MASKED_INPUT"):
        coordinator.submit_masked_input(advertisement)  # type: ignore[arg-type]
    with pytest.raises(ProtocolStateError, match="has 1/5 responses"):
        coordinator.close_advertise_keys()


def test_unknown_clients_and_forged_senders_are_refused(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    outsider = IdentityKeyPair.generate()
    body = federation.clients["silo_00"].advertise_keys(0, nonce).body
    forged = sign_message(
        body,
        kind=MessageKind.KEY_ADVERTISEMENT,
        round_id=0,
        round_nonce=nonce,
        sender="silo_99",
        identity=outsider,
    )
    with pytest.raises(ProtocolMessageError, match="unknown client"):
        coordinator.submit_key_advertisement(forged)

    impersonating = sign_message(
        body,
        kind=MessageKind.KEY_ADVERTISEMENT,
        round_id=0,
        round_nonce=nonce,
        sender="silo_01",
        identity=outsider,
    )
    with pytest.raises(ProtocolMessageError, match="invalid signature"):
        coordinator.submit_key_advertisement(impersonating)


def test_a_stage_closes_on_timeout_with_whoever_answered(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids[:4]:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    with pytest.raises(ProtocolStateError, match="has not timed out"):
        coordinator.close_advertise_keys()
    _advance(federation)
    pairing = coordinator.close_advertise_keys()
    assert set(pairing.body.participants) == set(federation.ids[:4])


def test_a_stage_below_quorum_fails_the_round(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids[:2]:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    _advance(federation)
    with pytest.raises(QuorumError, match="below threshold 3"):
        coordinator.close_advertise_keys()
    assert coordinator.stage is RoundStage.FAILED


def test_too_many_dropouts_fail_the_round_rather_than_leak() -> None:
    federation = build_federation(n_clients=5, threshold=4)
    with pytest.raises(QuorumError, match="refusing to unmask"):
        run_round(
            federation.coordinator,
            federation.clients,
            federation.updates(),
            0,
            clock=federation.clock,
            drop_before_masked_input=("silo_00", "silo_01"),
        )
    assert federation.coordinator.stage is RoundStage.FAILED


def test_survivors_that_go_silent_at_unmask_fail_the_round() -> None:
    federation = build_federation(n_clients=5, threshold=4)
    with pytest.raises(QuorumError, match="answered the unmask request"):
        run_round(
            federation.coordinator,
            federation.clients,
            federation.updates(),
            0,
            clock=federation.clock,
            drop_before_unmask=("silo_00", "silo_01"),
        )


def test_round_ids_are_never_reused(federation: Federation) -> None:
    run_round(
        federation.coordinator, federation.clients, federation.updates(), 0, clock=federation.clock
    )
    with pytest.raises(ProtocolStateError, match="already ran"):
        federation.coordinator.start_round(0)


def test_a_round_cannot_start_while_another_is_open(federation: Federation) -> None:
    federation.coordinator.start_round(0)
    with pytest.raises(ProtocolStateError, match="already in progress"):
        federation.coordinator.start_round(1)


def test_the_byte_budget_fails_the_round() -> None:
    federation = build_federation(max_round_bytes=200)
    nonce = federation.coordinator.start_round(0)

    def submit_all() -> None:
        for name in federation.ids:
            federation.coordinator.submit_key_advertisement(
                federation.clients[name].advertise_keys(0, nonce)
            )

    with pytest.raises(QuorumError, match="byte budget"):
        submit_all()
    assert federation.coordinator.stage is RoundStage.FAILED


def test_the_norm_bound_rejects_and_rolls_back_an_oversized_aggregate() -> None:
    federation = build_federation(n_clients=5, threshold=3, dimension=16, clip_norm=1.0)
    updates = federation.updates()
    updates["silo_04"] = updates["silo_04"] * 500.0
    result = run_round(
        federation.coordinator,
        federation.clients,
        updates,
        0,
        clock=federation.clock,
        clip_in_client=False,
    )
    assert result.stage is RoundStage.REJECTED
    assert not result.accepted
    assert result.aggregate is None
    assert result.aggregate_norm > result.norm_bound
    assert result.reason is not None
    assert "rolled back" in result.reason


def test_norm_bound_formula_and_errors(federation: Federation) -> None:
    coordinator = federation.coordinator
    assert coordinator.norm_bound(1) == pytest.approx(
        federation.config.clip_norm
        + np.sqrt(federation.config.dimension) * coordinator.scheme.resolution
    )
    assert coordinator.norm_bound(4) == pytest.approx(4 * coordinator.norm_bound(1))
    with pytest.raises(ValueError, match="survivors"):
        coordinator.norm_bound(0)
    error = AggregateRejectedError(9.0, 5.0)
    assert error.norm == 9.0
    assert error.bound == 5.0


def test_round_config_validation() -> None:
    with pytest.raises(ValueError, match="threshold must be >= 2"):
        RoundConfig(dimension=4, threshold=1)
    with pytest.raises(ValueError, match="cannot exceed max_clients"):
        RoundConfig(dimension=4, threshold=9, max_clients=5)
    with pytest.raises(ValueError, match="stage_timeout_seconds"):
        RoundConfig(dimension=4, threshold=2, stage_timeout_seconds=0)
    with pytest.raises(ValueError, match="norm_bound_slack"):
        RoundConfig(dimension=4, threshold=2, norm_bound_slack=-1)


def test_coordinator_roster_validation() -> None:
    identity = IdentityKeyPair.generate()
    keys = {"silo_00": IdentityKeyPair.generate().public_bytes}
    with pytest.raises(ValueError, match="below threshold"):
        SecAggCoordinator(identity, keys, RoundConfig(dimension=4, threshold=2))
    many = {f"silo_{i:02d}": IdentityKeyPair.generate().public_bytes for i in range(6)}
    with pytest.raises(ValueError, match="more registered clients"):
        SecAggCoordinator(identity, many, RoundConfig(dimension=4, threshold=2, max_clients=5))


def test_introspection_without_a_round(federation: Federation) -> None:
    with pytest.raises(ProtocolStateError, match="no round in progress"):
        _ = federation.coordinator.round_id
    assert federation.coordinator.registered_clients == tuple(federation.ids)


def test_abandon_reports_a_failed_round(federation: Federation) -> None:
    federation.coordinator.start_round(0)
    result = federation.coordinator.abandon("operator cancelled")
    assert isinstance(result, RoundResult)
    assert result.stage is RoundStage.FAILED
    assert result.reason == "operator cancelled"
    assert result.aggregate is None
    assert federation.coordinator.stage is RoundStage.FAILED


def test_share_bundles_must_address_exactly_the_pairing_set(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    pairing = coordinator.close_advertise_keys()
    bundle = federation.clients["silo_00"].build_share_bundle(pairing)
    # A validly signed message from the real silo that simply omits some recipients: the
    # signature check passes, so the coordinator's own completeness check has to catch it.
    trimmed = sign_message(
        bundle.body.model_copy(
            update={
                "ciphertexts": dict(list(bundle.body.ciphertexts.items())[:2]),
            }
        ),
        kind=MessageKind.SHARE_BUNDLE,
        round_id=0,
        round_nonce=nonce,
        sender="silo_00",
        identity=federation.identities["silo_00"],
    )
    with pytest.raises(ProtocolMessageError, match="addressed shares to"):
        coordinator.submit_share_bundle(trimmed)


def test_unmask_responses_are_checked_against_the_survivor_partition(
    federation: Federation,
) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    pairing = coordinator.close_advertise_keys()
    for name in federation.ids:
        coordinator.submit_share_bundle(federation.clients[name].build_share_bundle(pairing))
    masking, inbox = coordinator.close_share_keys()
    updates = federation.updates()
    for name in federation.ids:
        federation.clients[name].receive_shares(inbox[name])
        federation.clients[name].accept_masking_set(masking)
    for name in federation.ids[:4]:
        coordinator.submit_masked_input(
            federation.clients[name].masked_input(updates[name], coordinator.scheme)
        )
    _advance(federation)
    survivors = coordinator.close_masked_input()
    assert survivors.body.dropouts == ["silo_04"]

    response = federation.clients["silo_00"].unmask(survivors)
    coordinator.submit_unmask_shares(response)
    with pytest.raises(ProtocolStateError, match="already revealed"):
        coordinator.submit_unmask_shares(response)

    bogus = sign_message(
        response.body.model_copy(update={"share_index": 5}),
        kind=MessageKind.UNMASK_SHARES,
        round_id=0,
        round_nonce=nonce,
        sender="silo_01",
        identity=federation.identities["silo_01"],
    )
    with pytest.raises(ProtocolMessageError, match="wrong share index"):
        coordinator.submit_unmask_shares(bogus)

    offering_a_dropout_seed = sign_message(
        response.body.model_copy(
            update={
                "share_index": 2,
                "self_seed_shares": {**response.body.self_seed_shares, "silo_04": b"" * 32},
            }
        ),
        kind=MessageKind.UNMASK_SHARES,
        round_id=0,
        round_nonce=nonce,
        sender="silo_01",
        identity=federation.identities["silo_01"],
    )
    with pytest.raises(ProtocolMessageError, match="non-survivor"):
        coordinator.submit_unmask_shares(offering_a_dropout_seed)
