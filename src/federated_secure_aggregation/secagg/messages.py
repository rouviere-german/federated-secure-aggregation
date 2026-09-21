"""Signed wire messages for the secure-aggregation protocol.

Every message carries the protocol version, the stage it belongs to, the round id and the
round's random nonce, and is signed with the sender's long-lived Ed25519 identity key over the
canonical JSON of all of that. :func:`verify_message` checks the signature *and* that the four
context fields are the ones the receiver expected, which is what stops three concrete attacks:

- **key substitution**: a coordinator (or a network attacker) that swaps a client's advertised
  X25519 mask key for one it knows would be able to recompute that client's pairwise masks.
  The advertisement is signed by an identity key provisioned out of band, so the swap fails.
- **replay across rounds**: a message from round ``r`` presented in round ``r + 1`` has the wrong
  ``round_id``/``round_nonce`` inside the signed payload.
- **impersonation**: the sender id is inside the signed payload and is checked against the key
  the receiver looked up for that id.

Binary fields travel as standard base64 and are size-limited on the way in, so a single message
cannot exhaust the coordinator's memory before the HTTP body limit is reached.
"""

from __future__ import annotations

import base64
import binascii
from collections.abc import Mapping
from enum import StrEnum
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, Field, PlainSerializer

from federated_secure_aggregation.core.canonical import canonical_json
from federated_secure_aggregation.crypto.keys import IdentityKeyPair, verify
from federated_secure_aggregation.secagg.quantize import RingArray

__all__ = [
    "CLIENT_ID_PATTERN",
    "MAX_BINARY_BYTES",
    "MAX_PEERS",
    "PROTOCOL_VERSION",
    "B64Bytes",
    "ClientId",
    "KeyAdvertisement",
    "MaskedInput",
    "MaskingSet",
    "MessageBody",
    "MessageKind",
    "PairingSet",
    "ProtocolMessageError",
    "RoundStage",
    "ShareBundle",
    "SharePayload",
    "SignedMessage",
    "SurvivorSet",
    "UnmaskShares",
    "decode_vector",
    "encode_vector",
    "share_payload_aad",
    "sign_message",
    "verify_message",
]

PROTOCOL_VERSION = "fedsecagg/1"
MAX_BINARY_BYTES = 1 << 20
MAX_PEERS = 255
_SIGNING_DOMAIN = b"federated-secure-aggregation/message/v1\x00"


class ProtocolMessageError(ValueError):
    """A message failed verification or context binding."""


def _decode_b64(value: object) -> object:
    if isinstance(value, bytes | bytearray):
        return bytes(value)
    if not isinstance(value, str):
        return value
    if len(value) > 4 * (MAX_BINARY_BYTES // 3 + 4):
        raise ValueError("base64 field is too large")
    try:
        return base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("field is not valid base64") from exc


def _encode_b64(value: bytes) -> str:
    return base64.b64encode(value).decode("ascii")


B64Bytes = Annotated[
    bytes,
    BeforeValidator(_decode_b64),
    PlainSerializer(_encode_b64, return_type=str),
    Field(max_length=MAX_BINARY_BYTES),
]
# The single source of truth for silo ids. The roster loader in service/runtime.py reuses
# this constant, so a roster file can never introduce an id the protocol models would reject.
CLIENT_ID_PATTERN = r"^[a-z0-9][a-z0-9_-]{0,62}$"
ClientId = Annotated[str, Field(pattern=CLIENT_ID_PATTERN)]


class RoundStage(StrEnum):
    """Stages of one aggregation round, in order."""

    ADVERTISE_KEYS = "ADVERTISE_KEYS"
    SHARE_KEYS = "SHARE_KEYS"
    MASKED_INPUT = "MASKED_INPUT"
    UNMASK = "UNMASK"
    AGGREGATE = "AGGREGATE"
    COMPLETED = "COMPLETED"
    REJECTED = "REJECTED"
    FAILED = "FAILED"


class MessageKind(StrEnum):
    """Message types, each valid in exactly one stage."""

    KEY_ADVERTISEMENT = "key_advertisement"
    PAIRING_SET = "pairing_set"
    SHARE_BUNDLE = "share_bundle"
    MASKING_SET = "masking_set"
    MASKED_INPUT = "masked_input"
    SURVIVOR_SET = "survivor_set"
    UNMASK_SHARES = "unmask_shares"


KIND_STAGE: Mapping[MessageKind, RoundStage] = {
    MessageKind.KEY_ADVERTISEMENT: RoundStage.ADVERTISE_KEYS,
    MessageKind.PAIRING_SET: RoundStage.SHARE_KEYS,
    MessageKind.SHARE_BUNDLE: RoundStage.SHARE_KEYS,
    MessageKind.MASKING_SET: RoundStage.MASKED_INPUT,
    MessageKind.MASKED_INPUT: RoundStage.MASKED_INPUT,
    MessageKind.SURVIVOR_SET: RoundStage.UNMASK,
    MessageKind.UNMASK_SHARES: RoundStage.UNMASK,
}


class MessageBody(BaseModel):
    """Base class for message bodies: frozen and closed to unknown fields."""

    model_config = ConfigDict(frozen=True, extra="forbid")


class KeyAdvertisement(MessageBody):
    """A client's two per-round X25519 public keys, plus a commitment to its self-mask seed.

    The commitment is what makes a *reconstructed* self-mask seed checkable. The mask key has a
    natural commitment already -- its public half -- so a corrupted mask-key share is caught by
    re-deriving the public key. A self-mask seed has no such public half, so without this field
    a survivor could hand over a wrong seed share, silently corrupt the sum and have the round
    rejected by the norm bound as if a client had poisoned it. See
    :func:`~federated_secure_aggregation.secagg.masking.self_seed_commitment`.
    """

    mask_public_key: B64Bytes = Field(min_length=32, max_length=32)
    wrap_public_key: B64Bytes = Field(min_length=32, max_length=32)
    self_seed_commitment: B64Bytes = Field(min_length=32, max_length=32)


class SignedMessage[BodyT: MessageBody](BaseModel):
    """A body plus its protocol context and the sender's Ed25519 signature over both."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    protocol: Literal["fedsecagg/1"] = "fedsecagg/1"
    kind: MessageKind
    round_id: int = Field(ge=0)
    round_nonce: B64Bytes = Field(min_length=16, max_length=32)
    sender: ClientId
    body: BodyT
    signature: B64Bytes = Field(min_length=64, max_length=64)

    def signing_payload(self) -> bytes:
        """Canonical bytes covered by the signature (everything except the signature)."""
        payload: dict[str, Any] = self.model_dump(mode="json")
        payload.pop("signature")
        return _SIGNING_DOMAIN + canonical_json(payload).encode("utf-8")


class PairingSet(MessageBody):
    """The coordinator's announcement of the round's participants and parameters.

    The advertisements are relayed **verbatim**, still carrying each client's own signature, so
    every client re-verifies its peers' keys instead of trusting the coordinator's copy of them.
    That is what makes key substitution by the coordinator detectable rather than merely
    discouraged.
    """

    threshold: int = Field(ge=2, le=MAX_PEERS)
    dimension: int = Field(ge=1)
    clip_norm: float = Field(gt=0)
    scale: float = Field(gt=0)
    advertisements: dict[ClientId, SignedMessage[KeyAdvertisement]] = Field(
        min_length=1, max_length=MAX_PEERS
    )

    @property
    def participants(self) -> tuple[str, ...]:
        return tuple(sorted(self.advertisements))

    def share_index(self, client_id: str) -> int:
        """1-based Shamir evaluation point of ``client_id`` in this pairing set."""
        participants = self.participants
        if client_id not in participants:
            raise ProtocolMessageError(f"{client_id} is not in the pairing set")
        return participants.index(client_id) + 1

    def verified_keys(
        self, identity_keys: Mapping[str, bytes], *, round_id: int, round_nonce: bytes
    ) -> tuple[dict[str, bytes], dict[str, bytes]]:
        """Verify every relayed advertisement and return the mask and wrap key maps."""
        if len(self.advertisements) < self.threshold:
            raise ProtocolMessageError(
                f"pairing set has {len(self.advertisements)} clients, below threshold "
                f"{self.threshold}"
            )
        mask_keys: dict[str, bytes] = {}
        wrap_keys: dict[str, bytes] = {}
        for client_id in self.participants:
            advertisement = self.advertisements[client_id]
            identity = identity_keys.get(client_id)
            if identity is None:
                raise ProtocolMessageError(f"no identity key provisioned for {client_id}")
            verify_message(
                advertisement,
                public_key=identity,
                expect_kind=MessageKind.KEY_ADVERTISEMENT,
                expect_round_id=round_id,
                expect_round_nonce=round_nonce,
                expect_sender=client_id,
            )
            mask_keys[client_id] = advertisement.body.mask_public_key
            wrap_keys[client_id] = advertisement.body.wrap_public_key
        if len(set(mask_keys.values())) != len(mask_keys):
            raise ProtocolMessageError("two clients advertised the same mask key")
        return mask_keys, wrap_keys

    def seed_commitments(self) -> dict[str, bytes]:
        """Each participant's committed self-mask seed digest.

        Only meaningful after :meth:`verified_keys` has returned without raising: that is what
        checks each advertisement's signature, and this reads the same objects.
        """
        return {
            client_id: advertisement.body.self_seed_commitment
            for client_id, advertisement in self.advertisements.items()
        }


class ShareBundle(MessageBody):
    """One AEAD ciphertext per peer, each holding that peer's two Shamir shares."""

    ciphertexts: dict[ClientId, B64Bytes] = Field(min_length=1, max_length=MAX_PEERS)


class MaskingSet(MessageBody):
    """The coordinator's announcement of the set each client masks against.

    This is narrower than the pairing set: a client that advertised keys but never delivered its
    Shamir shares is excluded here, because nobody holds shares of its secrets and it could
    therefore never be recovered as a dropout.
    """

    members: list[ClientId] = Field(min_length=2, max_length=MAX_PEERS)


class MaskedInput(MessageBody):
    """A masked update: ``dimension`` little-endian ``uint32`` values."""

    dimension: int = Field(ge=1)
    vector: B64Bytes = Field(min_length=4)


class SurvivorSet(MessageBody):
    """The coordinator's announcement of who submitted a masked input and who dropped out."""

    survivors: list[ClientId] = Field(min_length=1, max_length=MAX_PEERS)
    dropouts: list[ClientId] = Field(max_length=MAX_PEERS)


class UnmaskShares(MessageBody):
    """A survivor's response: self-mask seed shares for survivors, key shares for dropouts."""

    share_index: int = Field(ge=1, le=MAX_PEERS)
    self_seed_shares: dict[ClientId, B64Bytes] = Field(max_length=MAX_PEERS)
    mask_key_shares: dict[ClientId, B64Bytes] = Field(max_length=MAX_PEERS)


class SharePayload(MessageBody):
    """Plaintext inside a share ciphertext: the dealer's two shares for one recipient."""

    dealer: ClientId
    recipient: ClientId
    share_index: int = Field(ge=1, le=MAX_PEERS)
    self_seed_share: B64Bytes = Field(min_length=1, max_length=64)
    mask_key_share: B64Bytes = Field(min_length=1, max_length=64)


def _unsigned_payload(
    *, kind: MessageKind, round_id: int, round_nonce: bytes, sender: str, body: MessageBody
) -> bytes:
    payload = {
        "protocol": PROTOCOL_VERSION,
        "kind": str(kind),
        "round_id": round_id,
        "round_nonce": _encode_b64(round_nonce),
        "sender": sender,
        "body": body.model_dump(mode="json"),
    }
    return _SIGNING_DOMAIN + canonical_json(payload).encode("utf-8")


def sign_message[T: MessageBody](
    body: T,
    *,
    kind: MessageKind,
    round_id: int,
    round_nonce: bytes,
    sender: str,
    identity: IdentityKeyPair,
) -> SignedMessage[T]:
    """Build a signed message. The signature covers the protocol context and the body."""
    payload = _unsigned_payload(
        kind=kind, round_id=round_id, round_nonce=round_nonce, sender=sender, body=body
    )
    return SignedMessage[T](
        kind=kind,
        round_id=round_id,
        round_nonce=round_nonce,
        sender=sender,
        body=body,
        signature=identity.sign(payload),
    )


def verify_message(
    message: SignedMessage[Any],
    *,
    public_key: bytes,
    expect_kind: MessageKind,
    expect_round_id: int,
    expect_round_nonce: bytes,
    expect_sender: str | None = None,
) -> None:
    """Raise :class:`ProtocolMessageError` unless signature and context all match."""
    if message.protocol != PROTOCOL_VERSION:
        raise ProtocolMessageError(f"unsupported protocol {message.protocol!r}")
    if message.kind is not expect_kind:
        raise ProtocolMessageError(f"expected a {expect_kind} message, got {message.kind}")
    if message.round_id != expect_round_id:
        raise ProtocolMessageError(
            f"message belongs to round {message.round_id}, not {expect_round_id}"
        )
    if message.round_nonce != expect_round_nonce:
        raise ProtocolMessageError("round nonce does not match this round (replay?)")
    if expect_sender is not None and message.sender != expect_sender:
        raise ProtocolMessageError(f"expected a message from {expect_sender}, got {message.sender}")
    if not verify(public_key, message.signing_payload(), message.signature):
        raise ProtocolMessageError(f"invalid signature on {message.kind} from {message.sender}")


def share_payload_aad(*, round_id: int, round_nonce: bytes, sender: str, recipient: str) -> bytes:
    """Associated data binding a share ciphertext to one round, sender and recipient."""
    return canonical_json(
        {
            "protocol": PROTOCOL_VERSION,
            "purpose": "share",
            "round_id": round_id,
            "round_nonce": _encode_b64(round_nonce),
            "from": sender,
            "to": recipient,
        }
    ).encode("utf-8")


def encode_vector(vector: RingArray) -> bytes:
    """Serialize a ring vector as little-endian ``uint32`` bytes."""
    array = np.asarray(vector, dtype=np.uint32)
    if array.ndim != 1:
        raise ProtocolMessageError("only 1-D vectors can be encoded")
    if array.size * 4 > MAX_BINARY_BYTES:
        raise ProtocolMessageError("vector exceeds the maximum message size")
    return array.astype("<u4").tobytes()


def decode_vector(raw: bytes, dimension: int) -> RingArray:
    """Inverse of :func:`encode_vector`, with an exact length check."""
    if dimension < 1:
        raise ProtocolMessageError("dimension must be >= 1")
    if len(raw) != dimension * 4:
        raise ProtocolMessageError(
            f"expected {dimension * 4} bytes for dimension {dimension}, got {len(raw)}"
        )
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
