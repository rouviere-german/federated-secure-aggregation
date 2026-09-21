"""The client half: what a silo refuses to do, even when the coordinator asks.

The client is the only party that can protect a silo from a curious coordinator, so these tests
are about the checks it performs on the coordinator's announcements rather than about the happy
path (which the round tests cover).
"""

from __future__ import annotations

import numpy as np
import pytest

from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.secagg.errors import ProtocolStateError
from federated_secure_aggregation.secagg.messages import (
    MaskingSet,
    MessageKind,
    ProtocolMessageError,
    SignedMessage,
    SurvivorSet,
    sign_message,
)
from tests.conftest import Federation, build_federation

pytestmark = pytest.mark.unit


def _open_round(
    federation: Federation,
) -> tuple[bytes, SignedMessage[MaskingSet], dict[str, dict[str, bytes]]]:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    pairing = coordinator.close_advertise_keys()
    for name in federation.ids:
        coordinator.submit_share_bundle(federation.clients[name].build_share_bundle(pairing))
    masking, inbox = coordinator.close_share_keys()
    return nonce, masking, inbox


def test_stages_must_run_in_order(federation: Federation) -> None:
    client = federation.clients["silo_00"]
    with pytest.raises(ProtocolStateError, match="no round in progress"):
        _ = client.round
    nonce = federation.coordinator.start_round(0)
    client.advertise_keys(0, nonce)
    with pytest.raises(ProtocolStateError, match="pairing set has not been accepted"):
        client.receive_shares({})
    with pytest.raises(ProtocolStateError, match="before the pairing set"):
        client.masked_input(np.zeros(federation.config.dimension), federation.coordinator.scheme)


def test_round_ids_must_strictly_increase(federation: Federation) -> None:
    client = federation.clients["silo_00"]
    client.advertise_keys(5, bytes(16))
    with pytest.raises(ProtocolStateError, match="strictly increase"):
        client.advertise_keys(5, bytes(16))
    with pytest.raises(ProtocolStateError, match="strictly increase"):
        client.advertise_keys(4, bytes(16))


def test_a_client_refuses_a_pairing_set_that_lost_its_own_key(federation: Federation) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    pairing = coordinator.close_advertise_keys()
    without_self = sign_message(
        pairing.body.model_copy(
            update={
                "advertisements": {
                    key: value
                    for key, value in pairing.body.advertisements.items()
                    if key != "silo_00"
                }
            }
        ),
        kind=MessageKind.PAIRING_SET,
        round_id=0,
        round_nonce=nonce,
        sender="coordinator",
        identity=federation.coordinator_identity,
    )
    with pytest.raises(ProtocolMessageError, match="advertised mask key"):
        federation.clients["silo_00"].build_share_bundle(without_self)


def test_a_client_refuses_announcements_signed_by_the_wrong_coordinator(
    federation: Federation,
) -> None:
    coordinator = federation.coordinator
    nonce = coordinator.start_round(0)
    for name in federation.ids:
        coordinator.submit_key_advertisement(federation.clients[name].advertise_keys(0, nonce))
    pairing = coordinator.close_advertise_keys()
    impostor = sign_message(
        pairing.body,
        kind=MessageKind.PAIRING_SET,
        round_id=0,
        round_nonce=nonce,
        sender="coordinator",
        identity=IdentityKeyPair.generate(),
    )
    with pytest.raises(ProtocolMessageError, match="invalid signature"):
        federation.clients["silo_00"].build_share_bundle(impostor)


def test_a_client_refuses_a_masking_set_it_cannot_support(federation: Federation) -> None:
    nonce, _masking, inbox = _open_round(federation)
    client = federation.clients["silo_00"]
    client.receive_shares(inbox["silo_00"])

    def announce(members: list[str]) -> object:
        return sign_message(
            MaskingSet(members=members),
            kind=MessageKind.MASKING_SET,
            round_id=0,
            round_nonce=nonce,
            sender="coordinator",
            identity=federation.coordinator_identity,
        )

    with pytest.raises(ProtocolMessageError, match="outside the pairing set"):
        client.accept_masking_set(announce(["silo_00", "silo_99"]))  # type: ignore[arg-type]
    with pytest.raises(ProtocolMessageError, match="not in the masking set"):
        client.accept_masking_set(announce(["silo_01", "silo_02", "silo_03"]))  # type: ignore[arg-type]
    with pytest.raises(ProtocolMessageError, match="below threshold"):
        client.accept_masking_set(announce(["silo_00", "silo_01"]))  # type: ignore[arg-type]
    with pytest.raises(ProtocolMessageError, match="duplicates"):
        client.accept_masking_set(announce(["silo_00", "silo_00", "silo_01"]))  # type: ignore[arg-type]


def test_a_client_refuses_to_mask_before_every_share_arrived(federation: Federation) -> None:
    _nonce, masking, inbox = _open_round(federation)
    client = federation.clients["silo_00"]
    partial = dict(list(inbox["silo_00"].items())[:1])
    client.receive_shares(partial)
    with pytest.raises(ProtocolMessageError, match="shares never arrived"):
        client.accept_masking_set(masking)


def test_a_client_refuses_shares_from_strangers_and_mismatched_payloads(
    federation: Federation,
) -> None:
    _nonce, _masking, inbox = _open_round(federation)
    client = federation.clients["silo_00"]
    with pytest.raises(ProtocolMessageError, match="unexpected share"):
        client.receive_shares({"silo_00": b"x" * 64})
    with pytest.raises(ProtocolMessageError, match="unexpected share"):
        client.receive_shares({"silo_99": b"x" * 64})
    with pytest.raises(ProtocolMessageError, match="failed to decrypt"):
        client.receive_shares({"silo_01": inbox["silo_02"]["silo_01"]})


def test_a_client_refuses_a_survivor_set_that_does_not_partition_the_masking_set(
    federation: Federation,
) -> None:
    nonce, masking, inbox = _open_round(federation)
    coordinator = federation.coordinator
    updates = federation.updates()
    for name in federation.ids:
        federation.clients[name].receive_shares(inbox[name])
        federation.clients[name].accept_masking_set(masking)
        coordinator.submit_masked_input(
            federation.clients[name].masked_input(updates[name], coordinator.scheme)
        )
    coordinator.close_masked_input()
    client = federation.clients["silo_00"]

    def announce(survivors: list[str], dropouts: list[str]) -> object:
        return sign_message(
            SurvivorSet(survivors=survivors, dropouts=dropouts),
            kind=MessageKind.SURVIVOR_SET,
            round_id=0,
            round_nonce=nonce,
            sender="coordinator",
            identity=federation.coordinator_identity,
        )

    with pytest.raises(ProtocolMessageError, match="both survivor and dropout"):
        client.unmask(announce(federation.ids, ["silo_01"]))  # type: ignore[arg-type]
    with pytest.raises(ProtocolMessageError, match="does not partition"):
        client.unmask(announce(federation.ids[:3], []))  # type: ignore[arg-type]
    with pytest.raises(ProtocolMessageError, match="not listed as a survivor"):
        client.unmask(announce(federation.ids[1:], ["silo_00"]))  # type: ignore[arg-type]


def test_a_client_refuses_to_unmask_below_the_threshold() -> None:
    """Fewer than ``t`` survivors means revealing shares could expose a single update."""
    federation = build_federation(n_clients=5, threshold=4)
    nonce, masking, inbox = _open_round(federation)
    coordinator = federation.coordinator
    updates = federation.updates()
    for name in federation.ids:
        federation.clients[name].receive_shares(inbox[name])
        federation.clients[name].accept_masking_set(masking)
        coordinator.submit_masked_input(
            federation.clients[name].masked_input(updates[name], coordinator.scheme)
        )
    coordinator.close_masked_input()
    announcement = sign_message(
        SurvivorSet(survivors=federation.ids[:3], dropouts=federation.ids[3:]),
        kind=MessageKind.SURVIVOR_SET,
        round_id=0,
        round_nonce=nonce,
        sender="coordinator",
        identity=federation.coordinator_identity,
    )
    with pytest.raises(ProtocolMessageError, match="refusing to reveal shares"):
        federation.clients["silo_00"].unmask(announcement)


def test_a_client_submits_at_most_one_input_and_one_reveal(federation: Federation) -> None:
    _nonce, masking, inbox = _open_round(federation)
    coordinator = federation.coordinator
    updates = federation.updates()
    for name in federation.ids:
        federation.clients[name].receive_shares(inbox[name])
        federation.clients[name].accept_masking_set(masking)
    client = federation.clients["silo_00"]
    coordinator.submit_masked_input(client.masked_input(updates["silo_00"], coordinator.scheme))
    with pytest.raises(ProtocolStateError, match="already submitted"):
        client.masked_input(updates["silo_00"], coordinator.scheme)

    for name in federation.ids[1:]:
        coordinator.submit_masked_input(
            federation.clients[name].masked_input(updates[name], coordinator.scheme)
        )
    survivors = coordinator.close_masked_input()
    client.unmask(survivors)
    with pytest.raises(ProtocolStateError, match="already revealed shares"):
        client.unmask(survivors)
    # A silo that never submitted an input holds no claim on the round and must not reveal.
    silent = federation.clients["silo_01"]
    silent.round.submitted_input = False
    silent.round.revealed = False
    with pytest.raises(ProtocolStateError, match="only clients that submitted"):
        silent.unmask(survivors)


def test_a_client_checks_the_scheme_dimension(federation: Federation) -> None:
    _nonce, masking, inbox = _open_round(federation)
    from federated_secure_aggregation.secagg.quantize import AggregationScheme

    client = federation.clients["silo_00"]
    client.receive_shares(inbox["silo_00"])
    client.accept_masking_set(masking)
    wrong = AggregationScheme.create(dimension=7, clip_norm=1.0, max_clients=5)
    with pytest.raises(ProtocolStateError, match="scheme dimension"):
        client.masked_input(np.zeros(7), wrong)
