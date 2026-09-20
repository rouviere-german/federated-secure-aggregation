"""What the coordinator does when a survivor reveals a share that is wrong.

The unmask stage is the one place where a *registered* silo hands the coordinator material that
the coordinator then uses to compute the aggregate. A silo that lies there is inside the trust
boundary for authentication but outside it for correctness, so the coordinator has to survive
three distinct failures without help:

- the share is corrupt but enough honest shares remain: reconstruct from a subset that verifies,
  and name the liar (this failure class *is* attributable, unlike a norm-respecting poisoned
  update);
- too many shares are corrupt, or too few were revealed at all: fail the round and move on;
- either way, the round must reach a terminal stage, because a round parked in ``AGGREGATE``
  would block every future round on this coordinator process for ever.

Every tampered message here is re-signed with the *sender's own* identity key, so it is a
message a registered silo could genuinely send. Corrupting the bytes without re-signing would
only exercise the signature check, which is a different (and already covered) control.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import numpy as np
import pytest

from federated_secure_aggregation.secagg.coordinator import RoundResult
from federated_secure_aggregation.secagg.masking import plaintext_sum
from federated_secure_aggregation.secagg.messages import (
    MessageKind,
    RoundStage,
    SignedMessage,
    UnmaskShares,
    sign_message,
)
from tests.conftest import Federation, build_federation

pytestmark = pytest.mark.unit

Tamper = Callable[[str, UnmaskShares], UnmaskShares]


def _flip(blob: bytes) -> bytes:
    """Change one byte of a share, keeping its length."""
    return bytes([blob[0] ^ 0xFF]) + blob[1:]


def _corrupt_seed_share(holder: str, victim: str) -> Tamper:
    def tamper(sender: str, body: UnmaskShares) -> UnmaskShares:
        if sender != holder or victim not in body.self_seed_shares:
            return body
        shares = dict(body.self_seed_shares)
        shares[victim] = _flip(shares[victim])
        return body.model_copy(update={"self_seed_shares": shares})

    return tamper


def _corrupt_key_share(holder: str, victim: str) -> Tamper:
    def tamper(sender: str, body: UnmaskShares) -> UnmaskShares:
        if sender != holder or victim not in body.mask_key_shares:
            return body
        shares = dict(body.mask_key_shares)
        shares[victim] = _flip(shares[victim])
        return body.model_copy(update={"mask_key_shares": shares})

    return tamper


def _omit_seed_share(holder: str, victim: str) -> Tamper:
    def tamper(sender: str, body: UnmaskShares) -> UnmaskShares:
        if sender != holder:
            return body
        shares = {peer: blob for peer, blob in body.self_seed_shares.items() if peer != victim}
        return body.model_copy(update={"self_seed_shares": shares})

    return tamper


def _chain(*tampers: Tamper) -> Tamper:
    def tamper(sender: str, body: UnmaskShares) -> UnmaskShares:
        for step in tampers:
            body = step(sender, body)
        return body

    return tamper


def _run_round(
    federation: Federation,
    updates: dict[str, np.ndarray],
    round_id: int = 0,
    *,
    drop: Collection[str] = (),
    tamper: Tamper | None = None,
) -> RoundResult:
    """One round, with a hook that rewrites (and re-signs) each survivor's unmask message."""
    coordinator = federation.coordinator
    clients = federation.clients
    timeout = federation.config.stage_timeout_seconds
    dropped = set(drop)

    nonce = coordinator.start_round(round_id)
    for name in federation.ids:
        coordinator.submit_key_advertisement(clients[name].advertise_keys(round_id, nonce))
    pairing = coordinator.close_advertise_keys()

    for name in federation.ids:
        coordinator.submit_share_bundle(clients[name].build_share_bundle(pairing))
    masking, inbox = coordinator.close_share_keys()

    for name in sorted(inbox):
        clients[name].receive_shares(inbox[name])
        clients[name].accept_masking_set(masking)
    for name in sorted(inbox):
        if name in dropped:
            continue
        coordinator.submit_masked_input(
            clients[name].masked_input(updates[name], coordinator.scheme)
        )
    if dropped:
        federation.clock.skip_timeout(timeout + 1.0)
    survivors = coordinator.close_masked_input()

    for name in survivors.body.survivors:
        message = clients[name].unmask(survivors)
        if tamper is not None:
            body = tamper(name, message.body)
            if body != message.body:
                message = _resign(federation, name, body, round_id=round_id, nonce=nonce)
        coordinator.submit_unmask_shares(message)
    return coordinator.finalize()


def _resign(
    federation: Federation,
    sender: str,
    body: UnmaskShares,
    *,
    round_id: int,
    nonce: bytes,
) -> SignedMessage[UnmaskShares]:
    """Re-sign a rewritten body with the sender's own key: a message it really could send."""
    return sign_message(
        body,
        kind=MessageKind.UNMASK_SHARES,
        round_id=round_id,
        round_nonce=nonce,
        sender=sender,
        identity=federation.identities[sender],
    )


def test_a_clean_round_names_nobody(federation: Federation) -> None:
    result = _run_round(federation, federation.updates())
    assert result.stage is RoundStage.COMPLETED
    assert result.corrupt_share_holders == ()


def test_one_corrupt_seed_share_is_routed_around_and_attributed(federation: Federation) -> None:
    """A lying holder costs the round nothing: another subset of shares reconstructs the seed.

    The coordinator can say *which* silo lied here, because the seed commitment in the signed
    advertisement fixes the true secret and ``t`` good shares fix the whole polynomial.
    """
    updates = federation.updates()
    result = _run_round(federation, updates, tamper=_corrupt_seed_share("silo_02", "silo_04"))
    assert result.stage is RoundStage.COMPLETED
    assert result.corrupt_share_holders == ("silo_02",)
    assert result.aggregate is not None
    scheme = federation.coordinator.scheme
    expected = scheme.decode(
        plaintext_sum(scheme.encode(updates[name]) for name in federation.ids),
        count=len(federation.ids),
    )
    assert np.array_equal(result.aggregate, expected)


def test_too_many_corrupt_seed_shares_fail_the_round_without_wedging(
    federation: Federation,
) -> None:
    """Three liars out of five leave no verifying subset, so the round must simply fail.

    The important half of this test is the last line: before the fix the machine stayed in
    ``AGGREGATE``, which is not a terminal stage, and every later ``start_round`` was refused.
    """
    victim = "silo_04"
    tamper = _chain(
        _corrupt_seed_share("silo_00", victim),
        _corrupt_seed_share("silo_01", victim),
        _corrupt_seed_share("silo_02", victim),
    )
    result = _run_round(federation, federation.updates(), tamper=tamper)
    assert result.stage is RoundStage.FAILED
    assert result.aggregate is None
    assert "self-mask seed" in str(result.reason)
    assert federation.coordinator.stage is RoundStage.FAILED
    assert federation.coordinator.start_round(1) is not None


def test_a_withheld_seed_share_fails_the_round_without_wedging(federation: Federation) -> None:
    """Below-threshold shares for one owner: a quorum failure, not a permanent wedge."""
    strict = build_federation(n_clients=5, threshold=5)
    result = _run_round(strict, strict.updates(), tamper=_omit_seed_share("silo_00", "silo_04"))
    assert result.stage is RoundStage.FAILED
    assert "below threshold" in str(result.reason)
    assert strict.coordinator.start_round(1) is not None


def test_one_corrupt_dropout_key_share_is_routed_around_and_attributed() -> None:
    federation = build_federation(n_clients=6, threshold=3)
    updates = federation.updates()
    result = _run_round(
        federation,
        updates,
        drop=("silo_00",),
        tamper=_corrupt_key_share("silo_03", "silo_00"),
    )
    assert result.stage is RoundStage.COMPLETED
    assert result.dropouts == ("silo_00",)
    assert result.corrupt_share_holders == ("silo_03",)


def test_an_unrecoverable_dropout_key_fails_the_round_without_wedging() -> None:
    federation = build_federation(n_clients=5, threshold=3)
    tamper = _chain(
        _corrupt_key_share("silo_01", "silo_00"),
        _corrupt_key_share("silo_02", "silo_00"),
        _corrupt_key_share("silo_03", "silo_00"),
        _corrupt_key_share("silo_04", "silo_00"),
    )
    result = _run_round(federation, federation.updates(), drop=("silo_00",), tamper=tamper)
    assert result.stage is RoundStage.FAILED
    assert "mask key" in str(result.reason)
    assert federation.coordinator.start_round(1) is not None
