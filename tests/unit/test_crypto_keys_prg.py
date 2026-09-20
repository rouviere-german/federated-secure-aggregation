"""Key agreement, signatures, AEAD and the mask PRG.

These wrap ``cryptography``, so the tests are about *how* the wrapping is used: that the two
sides of an exchange agree, that derived keys are domain-separated by purpose and round, that
associated data actually binds a ciphertext to its context, and that the PRG is deterministic
in its seed and uniform enough to serve as a mask.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy.stats import chisquare

from federated_secure_aggregation.crypto.keys import (
    AEAD_KEY_BYTES,
    IdentityKeyPair,
    InvalidCiphertextError,
    KeyAgreementKeyPair,
    aead_open,
    aead_seal,
    derive_key,
    shared_secret,
    sign,
    verify,
    x25519_public_bytes,
)
from federated_secure_aggregation.crypto.prg import PRG_SEED_BYTES, prg_bytes, prg_uint32

pytestmark = pytest.mark.unit


def test_identity_signatures_verify_and_reject_tampering() -> None:
    identity = IdentityKeyPair.generate()
    message = b"round 3 pairing set"
    signature = identity.sign(message)
    assert verify(identity.public_bytes, message, signature)
    assert not verify(identity.public_bytes, message + b"!", signature)
    assert not verify(IdentityKeyPair.generate().public_bytes, message, signature)


def test_verify_returns_false_instead_of_raising_on_malformed_input() -> None:
    identity = IdentityKeyPair.generate()
    assert not verify(b"short", b"m", identity.sign(b"m"))
    assert not verify(identity.public_bytes, b"m", b"short")
    assert not verify(identity.public_bytes, b"m", bytes(64))


def test_identity_round_trips_through_private_bytes() -> None:
    identity = IdentityKeyPair.generate()
    restored = IdentityKeyPair.from_private_bytes(identity.private_bytes)
    assert restored.public_bytes == identity.public_bytes
    assert verify(identity.public_bytes, b"m", sign(restored.private_bytes, b"m"))


def test_x25519_exchange_agrees_on_both_sides() -> None:
    alice = KeyAgreementKeyPair.generate()
    bob = KeyAgreementKeyPair.generate()
    assert shared_secret(alice.private_bytes, bob.public_bytes) == shared_secret(
        bob.private_bytes, alice.public_bytes
    )
    assert x25519_public_bytes(alice.private_bytes) == alice.public_bytes


def test_x25519_rejects_malformed_peer_keys() -> None:
    alice = KeyAgreementKeyPair.generate()
    with pytest.raises(ValueError, match="32 bytes"):
        shared_secret(alice.private_bytes, b"short")
    with pytest.raises(ValueError, match="32 bytes"):
        KeyAgreementKeyPair.from_private_bytes(b"short")


def test_low_order_peer_keys_are_refused() -> None:
    """An all-zero X25519 result would make the mask predictable; ``cryptography`` refuses it."""
    alice = KeyAgreementKeyPair.generate()
    with pytest.raises(ValueError, match=r"(?i)shared key"):
        shared_secret(alice.private_bytes, bytes(32))


def test_derive_key_separates_purposes_and_rounds() -> None:
    secret = os.urandom(32)
    mask = derive_key(secret, salt=b"round-1" + bytes(16), info=b"mask")
    wrap = derive_key(secret, salt=b"round-1" + bytes(16), info=b"wrap")
    next_round = derive_key(secret, salt=b"round-2" + bytes(16), info=b"mask")
    assert len({mask, wrap, next_round}) == 3
    with pytest.raises(ValueError, match="must not be empty"):
        derive_key(b"", salt=b"s", info=b"i")
    with pytest.raises(ValueError, match="length"):
        derive_key(secret, salt=b"s", info=b"i", length=0)


def test_aead_binds_the_ciphertext_to_its_associated_data() -> None:
    key = os.urandom(AEAD_KEY_BYTES)
    blob = aead_seal(key, b"share payload", b"round=1;from=a;to=b")
    assert aead_open(key, blob, b"round=1;from=a;to=b") == b"share payload"
    with pytest.raises(InvalidCiphertextError):
        aead_open(key, blob, b"round=1;from=a;to=c")
    with pytest.raises(InvalidCiphertextError):
        aead_open(os.urandom(AEAD_KEY_BYTES), blob, b"round=1;from=a;to=b")
    with pytest.raises(InvalidCiphertextError):
        aead_open(key, blob[:-1], b"round=1;from=a;to=b")


def test_aead_rejects_implausible_lengths_and_keys() -> None:
    key = os.urandom(AEAD_KEY_BYTES)
    with pytest.raises(InvalidCiphertextError, match="implausible"):
        aead_open(key, b"tiny", b"")
    with pytest.raises(ValueError, match="AEAD key"):
        aead_seal(b"short", b"x", b"")
    with pytest.raises(ValueError, match="AEAD key"):
        aead_open(b"short", bytes(32), b"")


def test_aead_nonces_differ_between_seals() -> None:
    key = os.urandom(AEAD_KEY_BYTES)
    first = aead_seal(key, b"same", b"")
    second = aead_seal(key, b"same", b"")
    assert first != second


@given(seed=st.binary(min_size=PRG_SEED_BYTES, max_size=PRG_SEED_BYTES))
@settings(max_examples=25)
def test_prg_is_deterministic_in_its_seed(seed: bytes) -> None:
    assert prg_uint32(seed, 64).tolist() == prg_uint32(seed, 64).tolist()
    assert prg_bytes(seed, 16) == prg_bytes(seed, 32)[:16]


def test_prg_validates_seed_and_length() -> None:
    with pytest.raises(ValueError, match="32 bytes"):
        prg_bytes(b"short", 4)
    with pytest.raises(ValueError, match="length must be"):
        prg_bytes(bytes(32), -1)
    with pytest.raises(ValueError, match="size must be"):
        prg_uint32(bytes(32), -1)
    with pytest.raises(ValueError, match="refusing to expand"):
        prg_bytes(bytes(32), 1 << 29)
    assert prg_bytes(bytes(32), 0) == b""
    assert prg_uint32(bytes(32), 0).size == 0


def test_prg_output_passes_a_uniformity_check() -> None:
    """The top byte of 200k PRG words should be uniform over 256 buckets."""
    values = prg_uint32(os.urandom(PRG_SEED_BYTES), 200_000)
    counts = np.bincount((values >> 24).astype(np.int64), minlength=256)
    assert chisquare(counts).pvalue > 0.001


def test_different_seeds_give_uncorrelated_streams() -> None:
    first = prg_uint32(os.urandom(PRG_SEED_BYTES), 20_000).astype(np.float64)
    second = prg_uint32(os.urandom(PRG_SEED_BYTES), 20_000).astype(np.float64)
    correlation = float(np.corrcoef(first, second)[0, 1])
    assert abs(correlation) < 0.05
