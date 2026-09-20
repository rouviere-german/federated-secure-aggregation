"""Identity signatures (Ed25519), key agreement (X25519), key derivation (HKDF) and AEAD.

Three kinds of key material appear in the protocol:

- **Identity keys** (Ed25519) are long-lived and provisioned out of band. Every protocol message
  is signed with one, so the coordinator cannot substitute a key advertisement and a client
  cannot replay another client's message. Only public identity keys are ever distributed.
- **Mask keys** (X25519) are fresh per round. Two clients derive a shared secret from their mask
  key pair, and from it the seed of the pairwise mask. A client's *private* mask key is what
  peers hold Shamir shares of, so a dropped client's masks can be rebuilt.
- **Share-wrapping keys** (X25519) are also fresh per round and used only to encrypt Shamir
  shares in transit, so the coordinator relays ciphertext it cannot read. They are separate from
  the mask keys because reconstructing a dropped client's mask key must not also hand the
  coordinator the ability to decrypt that client's incoming shares.

Everything here wraps ``cryptography``; no primitive is implemented by hand.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.asymmetric.x25519 import X25519PrivateKey, X25519PublicKey
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

__all__ = [
    "AEAD_KEY_BYTES",
    "AEAD_NONCE_BYTES",
    "KEY_BYTES",
    "SIGNATURE_BYTES",
    "IdentityKeyPair",
    "InvalidCiphertextError",
    "KeyAgreementKeyPair",
    "aead_open",
    "aead_seal",
    "derive_key",
    "shared_secret",
    "sign",
    "verify",
    "x25519_public_bytes",
]

KEY_BYTES = 32
SIGNATURE_BYTES = 64
AEAD_KEY_BYTES = 32
AEAD_NONCE_BYTES = 12
_MAX_CIPHERTEXT_BYTES = 1 << 20


class InvalidCiphertextError(ValueError):
    """AEAD decryption failed: wrong key, wrong associated data or tampering."""


@dataclass(frozen=True, slots=True)
class IdentityKeyPair:
    """A long-lived Ed25519 signing key. ``private_bytes`` never leaves its owner."""

    private_bytes: bytes
    public_bytes: bytes

    @classmethod
    def generate(cls) -> IdentityKeyPair:
        private = Ed25519PrivateKey.generate()
        return cls(
            private_bytes=private.private_bytes_raw(),
            public_bytes=private.public_key().public_bytes_raw(),
        )

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> IdentityKeyPair:
        private = Ed25519PrivateKey.from_private_bytes(private_bytes)
        return cls(
            private_bytes=private_bytes, public_bytes=private.public_key().public_bytes_raw()
        )

    def sign(self, message: bytes) -> bytes:
        return sign(self.private_bytes, message)


@dataclass(frozen=True, slots=True)
class KeyAgreementKeyPair:
    """A per-round X25519 key pair used for masks or for wrapping shares."""

    private_bytes: bytes
    public_bytes: bytes

    @classmethod
    def generate(cls) -> KeyAgreementKeyPair:
        private = X25519PrivateKey.generate()
        return cls(
            private_bytes=private.private_bytes_raw(),
            public_bytes=private.public_key().public_bytes_raw(),
        )

    @classmethod
    def from_private_bytes(cls, private_bytes: bytes) -> KeyAgreementKeyPair:
        if len(private_bytes) != KEY_BYTES:
            raise ValueError(f"X25519 private key must be {KEY_BYTES} bytes")
        private = X25519PrivateKey.from_private_bytes(private_bytes)
        return cls(
            private_bytes=private_bytes, public_bytes=private.public_key().public_bytes_raw()
        )


def x25519_public_bytes(private_bytes: bytes) -> bytes:
    """Public key matching an X25519 private key; used to check reconstructed keys."""
    return KeyAgreementKeyPair.from_private_bytes(private_bytes).public_bytes


def sign(private_bytes: bytes, message: bytes) -> bytes:
    return Ed25519PrivateKey.from_private_bytes(private_bytes).sign(message)


def verify(public_bytes: bytes, message: bytes, signature: bytes) -> bool:
    """Constant-time-ish signature check that returns a bool instead of raising."""
    if len(public_bytes) != KEY_BYTES or len(signature) != SIGNATURE_BYTES:
        return False
    try:
        Ed25519PublicKey.from_public_bytes(public_bytes).verify(signature, message)
    except (InvalidSignature, ValueError):
        return False
    return True


def shared_secret(private_bytes: bytes, peer_public_bytes: bytes) -> bytes:
    """X25519 ECDH. Raises on a malformed peer key; low-order points give an all-zero secret.

    ``cryptography`` rejects the all-zero output itself, so a peer that advertises a low-order
    point produces ``ValueError`` here rather than a predictable shared secret.
    """
    if len(peer_public_bytes) != KEY_BYTES:
        raise ValueError(f"peer public key must be {KEY_BYTES} bytes")
    private = X25519PrivateKey.from_private_bytes(private_bytes)
    return private.exchange(X25519PublicKey.from_public_bytes(peer_public_bytes))


def derive_key(secret: bytes, *, salt: bytes, info: bytes, length: int = KEY_BYTES) -> bytes:
    """HKDF-SHA256. ``salt`` carries the round nonce, ``info`` the purpose label."""
    if not secret:
        raise ValueError("secret must not be empty")
    if length < 1 or length > 255 * 32:
        raise ValueError("length must be in 1..8160")
    return HKDF(algorithm=hashes.SHA256(), length=length, salt=salt, info=info).derive(secret)


def aead_seal(key: bytes, plaintext: bytes, associated_data: bytes) -> bytes:
    """AES-256-GCM with a fresh random nonce, returned as ``nonce || ciphertext``."""
    if len(key) != AEAD_KEY_BYTES:
        raise ValueError(f"AEAD key must be {AEAD_KEY_BYTES} bytes")
    nonce = os.urandom(AEAD_NONCE_BYTES)
    return nonce + AESGCM(key).encrypt(nonce, plaintext, associated_data)


def aead_open(key: bytes, blob: bytes, associated_data: bytes) -> bytes:
    """Reverse of :func:`aead_seal`; raises :class:`InvalidCiphertextError` on any failure."""
    if len(key) != AEAD_KEY_BYTES:
        raise ValueError(f"AEAD key must be {AEAD_KEY_BYTES} bytes")
    if len(blob) <= AEAD_NONCE_BYTES or len(blob) > _MAX_CIPHERTEXT_BYTES:
        raise InvalidCiphertextError("ciphertext has an implausible length")
    nonce, ciphertext = blob[:AEAD_NONCE_BYTES], blob[AEAD_NONCE_BYTES:]
    try:
        return AESGCM(key).decrypt(nonce, ciphertext, associated_data)
    except Exception as exc:  # cryptography raises InvalidTag, a subclass of Exception
        raise InvalidCiphertextError("AEAD decryption failed") from exc
