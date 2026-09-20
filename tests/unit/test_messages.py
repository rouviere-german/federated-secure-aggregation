"""Signed messages: what the signature covers and what the receiver must check.

The three attacks the binding exists to stop each get a test: substituting an advertised key,
replaying a message into another round, and sending under someone else's name. The positive
case - a well-formed message survives a JSON round trip and still verifies - is tested too,
because a signature over a payload that does not survive serialization is useless.
"""

from __future__ import annotations

import base64

import numpy as np
import pytest
from pydantic import ValidationError

from federated_secure_aggregation.crypto.keys import IdentityKeyPair, KeyAgreementKeyPair
from federated_secure_aggregation.secagg.masking import self_seed_commitment
from federated_secure_aggregation.secagg.messages import (
    MAX_BINARY_BYTES,
    KeyAdvertisement,
    MaskedInput,
    MessageKind,
    PairingSet,
    ProtocolMessageError,
    SignedMessage,
    decode_vector,
    encode_vector,
    share_payload_aad,
    sign_message,
    verify_message,
)

pytestmark = pytest.mark.unit

NONCE = bytes(range(16))
OTHER_NONCE = bytes(range(16, 32))
SEED = bytes(range(32, 64))


def _advertisement(
    identity: IdentityKeyPair, sender: str = "silo_00", round_id: int = 1, nonce: bytes = NONCE
) -> SignedMessage[KeyAdvertisement]:
    keys = KeyAgreementKeyPair.generate()
    body = KeyAdvertisement(
        mask_public_key=keys.public_bytes,
        wrap_public_key=keys.public_bytes,
        self_seed_commitment=self_seed_commitment(SEED, NONCE),
    )
    return sign_message(
        body,
        kind=MessageKind.KEY_ADVERTISEMENT,
        round_id=round_id,
        round_nonce=nonce,
        sender=sender,
        identity=identity,
    )


def _verify(message: SignedMessage[KeyAdvertisement], public: bytes, **overrides: object) -> None:
    kwargs: dict[str, object] = {
        "public_key": public,
        "expect_kind": MessageKind.KEY_ADVERTISEMENT,
        "expect_round_id": 1,
        "expect_round_nonce": NONCE,
        "expect_sender": "silo_00",
    }
    kwargs.update(overrides)
    verify_message(message, **kwargs)  # type: ignore[arg-type]


def test_a_well_formed_message_survives_json_and_verifies() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    restored = SignedMessage[KeyAdvertisement].model_validate_json(message.model_dump_json())
    _verify(restored, identity.public_bytes)
    assert restored.body.mask_public_key == message.body.mask_public_key


def test_a_modified_body_fails_verification() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    tampered = message.model_copy(
        update={
            "body": KeyAdvertisement(
                mask_public_key=KeyAgreementKeyPair.generate().public_bytes,
                wrap_public_key=message.body.wrap_public_key,
                self_seed_commitment=message.body.self_seed_commitment,
            )
        }
    )
    with pytest.raises(ProtocolMessageError, match="invalid signature"):
        _verify(tampered, identity.public_bytes)


def test_replaying_a_message_into_another_round_fails() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    with pytest.raises(ProtocolMessageError, match="round 1"):
        _verify(message, identity.public_bytes, expect_round_id=2)
    with pytest.raises(ProtocolMessageError, match="round nonce"):
        _verify(message, identity.public_bytes, expect_round_nonce=OTHER_NONCE)


def test_impersonation_and_wrong_kind_are_refused() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    with pytest.raises(ProtocolMessageError, match="expected a message from"):
        _verify(message, identity.public_bytes, expect_sender="silo_01")
    with pytest.raises(ProtocolMessageError, match="expected a"):
        _verify(message, identity.public_bytes, expect_kind=MessageKind.MASKED_INPUT)
    with pytest.raises(ProtocolMessageError, match="invalid signature"):
        _verify(message, IdentityKeyPair.generate().public_bytes)


def test_unsupported_protocol_version_is_refused() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    payload = message.model_dump(mode="json")
    payload["protocol"] = "fedsecagg/99"
    with pytest.raises(ValidationError):
        SignedMessage[KeyAdvertisement].model_validate(payload)


def test_pairing_set_verifies_relayed_advertisements() -> None:
    identities = {f"silo_{index:02d}": IdentityKeyPair.generate() for index in range(3)}
    advertisements = {
        name: _advertisement(identity, sender=name) for name, identity in identities.items()
    }
    pairing = PairingSet(
        threshold=2,
        dimension=8,
        clip_norm=1.0,
        scale=1000.0,
        advertisements=advertisements,
    )
    public = {name: identity.public_bytes for name, identity in identities.items()}
    mask_keys, wrap_keys = pairing.verified_keys(public, round_id=1, round_nonce=NONCE)
    assert set(mask_keys) == set(identities)
    assert set(wrap_keys) == set(identities)
    assert pairing.participants == tuple(sorted(identities))
    assert pairing.share_index("silo_01") == 2


def test_pairing_set_rejects_a_substituted_key() -> None:
    """A coordinator that swaps one silo's advertisement is caught by that silo's signature."""
    identities = {f"silo_{index:02d}": IdentityKeyPair.generate() for index in range(3)}
    advertisements = {
        name: _advertisement(identity, sender=name) for name, identity in identities.items()
    }
    attacker = IdentityKeyPair.generate()
    advertisements["silo_01"] = _advertisement(attacker, sender="silo_01")
    pairing = PairingSet(
        threshold=2, dimension=8, clip_norm=1.0, scale=1000.0, advertisements=advertisements
    )
    public = {name: identity.public_bytes for name, identity in identities.items()}
    with pytest.raises(ProtocolMessageError, match="invalid signature"):
        pairing.verified_keys(public, round_id=1, round_nonce=NONCE)


def test_pairing_set_rejects_unknown_ids_duplicates_and_undersized_sets() -> None:
    identities = {f"silo_{index:02d}": IdentityKeyPair.generate() for index in range(3)}
    advertisements = {
        name: _advertisement(identity, sender=name) for name, identity in identities.items()
    }
    pairing = PairingSet(
        threshold=2, dimension=8, clip_norm=1.0, scale=1000.0, advertisements=advertisements
    )
    partial = {name: identities[name].public_bytes for name in list(identities)[:2]}
    with pytest.raises(ProtocolMessageError, match="no identity key"):
        pairing.verified_keys(partial, round_id=1, round_nonce=NONCE)
    with pytest.raises(ProtocolMessageError, match="not in the pairing set"):
        pairing.share_index("silo_99")

    small = PairingSet(
        threshold=3,
        dimension=8,
        clip_norm=1.0,
        scale=1000.0,
        advertisements={"silo_00": advertisements["silo_00"]},
    )
    with pytest.raises(ProtocolMessageError, match="below threshold"):
        small.verified_keys(
            {"silo_00": identities["silo_00"].public_bytes}, round_id=1, round_nonce=NONCE
        )


def test_pairing_set_rejects_two_clients_sharing_a_mask_key() -> None:
    identities = {f"silo_{index:02d}": IdentityKeyPair.generate() for index in range(2)}
    shared = KeyAgreementKeyPair.generate()
    body = KeyAdvertisement(
        mask_public_key=shared.public_bytes,
        wrap_public_key=shared.public_bytes,
        self_seed_commitment=self_seed_commitment(SEED, NONCE),
    )
    advertisements = {
        name: sign_message(
            body,
            kind=MessageKind.KEY_ADVERTISEMENT,
            round_id=1,
            round_nonce=NONCE,
            sender=name,
            identity=identity,
        )
        for name, identity in identities.items()
    }
    pairing = PairingSet(
        threshold=2, dimension=8, clip_norm=1.0, scale=1000.0, advertisements=advertisements
    )
    with pytest.raises(ProtocolMessageError, match="same mask key"):
        pairing.verified_keys(
            {name: identity.public_bytes for name, identity in identities.items()},
            round_id=1,
            round_nonce=NONCE,
        )


def test_base64_fields_reject_garbage_and_oversized_values() -> None:
    identity = IdentityKeyPair.generate()
    payload = _advertisement(identity).model_dump(mode="json")
    payload["signature"] = "not base64!!"
    with pytest.raises(ValidationError):
        SignedMessage[KeyAdvertisement].model_validate(payload)
    with pytest.raises(ValidationError):
        MaskedInput(dimension=1, vector="A" * (4 * (MAX_BINARY_BYTES // 3 + 8)))


def test_client_ids_must_match_the_protocol_pattern() -> None:
    identity = IdentityKeyPair.generate()
    with pytest.raises(ValidationError):
        _advertisement(identity, sender="Silo With Spaces")


def test_vector_encoding_round_trips_and_checks_lengths() -> None:
    vector = np.arange(8, dtype=np.uint32)
    raw = encode_vector(vector)
    assert np.array_equal(decode_vector(raw, 8), vector)
    with pytest.raises(ProtocolMessageError, match="expected 36 bytes"):
        decode_vector(raw, 9)
    with pytest.raises(ProtocolMessageError, match="dimension must be"):
        decode_vector(raw, 0)
    with pytest.raises(ProtocolMessageError, match="1-D"):
        encode_vector(np.zeros((2, 2), dtype=np.uint32))
    with pytest.raises(ProtocolMessageError, match="maximum message size"):
        encode_vector(np.zeros(MAX_BINARY_BYTES, dtype=np.uint32))


def test_share_aad_binds_round_sender_and_recipient() -> None:
    base = share_payload_aad(round_id=1, round_nonce=NONCE, sender="a", recipient="b")
    assert base != share_payload_aad(round_id=2, round_nonce=NONCE, sender="a", recipient="b")
    assert base != share_payload_aad(round_id=1, round_nonce=OTHER_NONCE, sender="a", recipient="b")
    assert base != share_payload_aad(round_id=1, round_nonce=NONCE, sender="b", recipient="a")
    assert base64.b64encode(NONCE).decode("ascii") in base.decode("utf-8")


def test_signing_payload_excludes_only_the_signature() -> None:
    identity = IdentityKeyPair.generate()
    message = _advertisement(identity)
    payload = message.signing_payload()
    assert b'"signature"' not in payload
    assert b'"sender":"silo_00"' in payload
