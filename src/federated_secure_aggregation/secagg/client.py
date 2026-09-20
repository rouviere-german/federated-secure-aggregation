"""The client half of the round state machine.

A :class:`SecAggClient` owns exactly one long-lived identity key and, per round, one mask key
pair, one share-wrapping key pair and one self-mask seed. It refuses to run a stage out of
order, refuses to reuse a round id and never hands the coordinator both its self-mask seed and
its mask private key.

The client is a pure state machine over messages: nothing here touches the network, files or a
clock. The HTTP driver in :mod:`federated_secure_aggregation.service.client_http` and the
in-process simulation both call the same methods, so the tests exercise the same code the demo
runs.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.crypto import shamir
from federated_secure_aggregation.crypto.keys import (
    AEAD_KEY_BYTES,
    IdentityKeyPair,
    InvalidCiphertextError,
    KeyAgreementKeyPair,
    aead_open,
    aead_seal,
)
from federated_secure_aggregation.crypto.prg import PRG_SEED_BYTES
from federated_secure_aggregation.secagg.errors import ProtocolStateError
from federated_secure_aggregation.secagg.masking import (
    MaskDeriver,
    mask_vector,
    self_seed_commitment,
)
from federated_secure_aggregation.secagg.messages import (
    KeyAdvertisement,
    MaskedInput,
    MaskingSet,
    MessageKind,
    PairingSet,
    ProtocolMessageError,
    ShareBundle,
    SharePayload,
    SignedMessage,
    SurvivorSet,
    UnmaskShares,
    encode_vector,
    share_payload_aad,
    sign_message,
    verify_message,
)
from federated_secure_aggregation.secagg.quantize import AggregationScheme

__all__ = ["ClientRound", "SecAggClient"]

RandomBytes = Callable[[int], bytes]


@dataclass(slots=True)
class ClientRound:
    """Per-round secrets and the peer state a client accumulated so far."""

    round_id: int
    round_nonce: bytes
    mask_keys: KeyAgreementKeyPair
    wrap_keys: KeyAgreementKeyPair
    self_mask_seed: bytes
    pairing: PairingSet | None = None
    peer_mask_public: dict[str, bytes] = field(default_factory=dict)
    peer_wrap_public: dict[str, bytes] = field(default_factory=dict)
    held_shares: dict[str, SharePayload] = field(default_factory=dict)
    masking_set: tuple[str, ...] | None = None
    submitted_input: bool = False
    revealed: bool = False


class SecAggClient:
    """One silo. Holds its identity key and drives one round at a time."""

    def __init__(
        self,
        client_id: str,
        identity: IdentityKeyPair,
        *,
        peer_identity_keys: Mapping[str, bytes],
        coordinator_identity_key: bytes,
        random_bytes: RandomBytes = os.urandom,
    ) -> None:
        self.client_id = client_id
        self._identity = identity
        self._peer_identity_keys = dict(peer_identity_keys)
        self._peer_identity_keys[client_id] = identity.public_bytes
        self._coordinator_identity_key = coordinator_identity_key
        self._random_bytes = random_bytes
        self._round: ClientRound | None = None
        self._last_round_id: int | None = None

    @property
    def round(self) -> ClientRound:
        if self._round is None:
            raise ProtocolStateError(f"{self.client_id} has no round in progress")
        return self._round

    # --- stage 1: ADVERTISE_KEYS -------------------------------------------------------

    def advertise_keys(self, round_id: int, round_nonce: bytes) -> SignedMessage[KeyAdvertisement]:
        """Start a round: generate fresh per-round keys and sign their advertisement."""
        if self._last_round_id is not None and round_id <= self._last_round_id:
            raise ProtocolStateError(
                f"{self.client_id} already participated in round {self._last_round_id}; "
                f"round ids must strictly increase (got {round_id})"
            )
        mask_keys = KeyAgreementKeyPair.generate()
        wrap_keys = KeyAgreementKeyPair.generate()
        self_mask_seed = self._random_bytes(PRG_SEED_BYTES)
        self._round = ClientRound(
            round_id=round_id,
            round_nonce=round_nonce,
            mask_keys=mask_keys,
            wrap_keys=wrap_keys,
            self_mask_seed=self_mask_seed,
        )
        self._last_round_id = round_id
        body = KeyAdvertisement(
            mask_public_key=mask_keys.public_bytes,
            wrap_public_key=wrap_keys.public_bytes,
            # Signed alongside the public keys, so the coordinator can check that a seed
            # rebuilt from peers' shares is the seed this client actually used.
            self_seed_commitment=self_seed_commitment(self_mask_seed, round_nonce),
        )
        return self._sign(MessageKind.KEY_ADVERTISEMENT, body)

    # --- stage 2: SHARE_KEYS -----------------------------------------------------------

    def build_share_bundle(
        self, pairing_message: SignedMessage[PairingSet]
    ) -> SignedMessage[ShareBundle]:
        """Verify the pairing set, then Shamir-share both secrets to every peer."""
        state = self.round
        verify_message(
            pairing_message,
            public_key=self._coordinator_identity_key,
            expect_kind=MessageKind.PAIRING_SET,
            expect_round_id=state.round_id,
            expect_round_nonce=state.round_nonce,
        )
        pairing = pairing_message.body
        mask_keys, wrap_keys = pairing.verified_keys(
            self._peer_identity_keys, round_id=state.round_id, round_nonce=state.round_nonce
        )
        if mask_keys.get(self.client_id) != state.mask_keys.public_bytes:
            raise ProtocolMessageError(
                "the pairing set does not carry this client's advertised mask key"
            )
        if wrap_keys.get(self.client_id) != state.wrap_keys.public_bytes:
            raise ProtocolMessageError(
                "the pairing set does not carry this client's advertised wrap key"
            )
        state.pairing = pairing
        state.peer_mask_public = {k: v for k, v in mask_keys.items() if k != self.client_id}
        state.peer_wrap_public = {k: v for k, v in wrap_keys.items() if k != self.client_id}

        participants = pairing.participants
        seed_shares = shamir.split(
            state.self_mask_seed,
            threshold=pairing.threshold,
            shares=len(participants),
            random_bytes=self._random_bytes,
        )
        key_shares = shamir.split(
            state.mask_keys.private_bytes,
            threshold=pairing.threshold,
            shares=len(participants),
            random_bytes=self._random_bytes,
        )
        deriver = MaskDeriver(round_nonce=state.round_nonce, dimension=pairing.dimension)
        ciphertexts: dict[str, bytes] = {}
        for position, recipient in enumerate(participants):
            payload = SharePayload(
                dealer=self.client_id,
                recipient=recipient,
                share_index=position + 1,
                self_seed_share=seed_shares[position].data,
                mask_key_share=key_shares[position].data,
            )
            if recipient == self.client_id:
                # A client keeps the share addressed to itself instead of encrypting it to
                # itself. That is a deliberate change from Bonawitz et al. Figure 1, where a
                # dealer holds no share of its own secret: keeping it means the seed of a
                # surviving client can be rebuilt from the ``t`` survivors themselves, so the
                # quorum rule stays the single condition ``|survivors| >= t`` with no
                # ``t + 1`` special case. It leaks nothing: the dealer already knows the secret.
                state.held_shares[self.client_id] = payload
                continue
            key = deriver.wrap_key(state.wrap_keys.private_bytes, state.peer_wrap_public[recipient])
            ciphertexts[recipient] = aead_seal(
                key[:AEAD_KEY_BYTES],
                payload.model_dump_json().encode("utf-8"),
                share_payload_aad(
                    round_id=state.round_id,
                    round_nonce=state.round_nonce,
                    sender=self.client_id,
                    recipient=recipient,
                ),
            )
        if not ciphertexts:
            raise ProtocolStateError("a round needs at least two participants")
        return self._sign(MessageKind.SHARE_BUNDLE, ShareBundle(ciphertexts=ciphertexts))

    def receive_shares(self, ciphertexts: Mapping[str, bytes]) -> None:
        """Decrypt and check the shares peers addressed to this client."""
        state = self.round
        if state.pairing is None:
            raise ProtocolStateError("the pairing set has not been accepted yet")
        deriver = MaskDeriver(round_nonce=state.round_nonce, dimension=state.pairing.dimension)
        own_index = state.pairing.share_index(self.client_id)
        for dealer, blob in sorted(ciphertexts.items()):
            if dealer == self.client_id or dealer not in state.peer_wrap_public:
                raise ProtocolMessageError(f"unexpected share from {dealer}")
            key = deriver.wrap_key(state.wrap_keys.private_bytes, state.peer_wrap_public[dealer])
            try:
                plaintext = aead_open(
                    key[:AEAD_KEY_BYTES],
                    blob,
                    share_payload_aad(
                        round_id=state.round_id,
                        round_nonce=state.round_nonce,
                        sender=dealer,
                        recipient=self.client_id,
                    ),
                )
            except InvalidCiphertextError as exc:
                raise ProtocolMessageError(f"share from {dealer} failed to decrypt") from exc
            payload = SharePayload.model_validate_json(plaintext)
            if payload.dealer != dealer or payload.recipient != self.client_id:
                raise ProtocolMessageError(f"share from {dealer} names the wrong parties")
            if payload.share_index != own_index:
                raise ProtocolMessageError(
                    f"share from {dealer} has index {payload.share_index}, expected {own_index}"
                )
            state.held_shares[dealer] = payload

    # --- stage 3: MASKED_INPUT ---------------------------------------------------------

    def accept_masking_set(self, masking_message: SignedMessage[MaskingSet]) -> None:
        """Narrow the peer set to the clients that actually delivered their shares.

        Masking against a client whose shares nobody holds would make the round unrecoverable
        if that client then dropped out, so the coordinator announces the reduced set and the
        client checks it is a subset of the pairing set, still meets the threshold, and
        contains this client.
        """
        state = self.round
        if state.pairing is None:
            raise ProtocolStateError("the pairing set has not been accepted yet")
        verify_message(
            masking_message,
            public_key=self._coordinator_identity_key,
            expect_kind=MessageKind.MASKING_SET,
            expect_round_id=state.round_id,
            expect_round_nonce=state.round_nonce,
        )
        members = set(masking_message.body.members)
        if len(members) != len(masking_message.body.members):
            raise ProtocolMessageError("masking set contains duplicates")
        if not members <= set(state.pairing.participants):
            raise ProtocolMessageError("masking set names clients outside the pairing set")
        if self.client_id not in members:
            raise ProtocolMessageError("this client is not in the masking set")
        if len(members) < state.pairing.threshold:
            raise ProtocolMessageError(
                f"masking set has {len(members)} members, below threshold {state.pairing.threshold}"
            )
        peers = members - {self.client_id}
        missing = peers - set(state.held_shares)
        if missing:
            raise ProtocolMessageError(
                "masking set includes clients whose shares never arrived: "
                + ", ".join(sorted(missing))
            )
        state.masking_set = tuple(sorted(members))
        state.peer_mask_public = {
            peer: key for peer, key in state.peer_mask_public.items() if peer in peers
        }

    def masked_input(
        self,
        update: npt.NDArray[np.float64],
        scheme: AggregationScheme,
        *,
        clip: bool = True,
    ) -> SignedMessage[MaskedInput]:
        """Clip, quantize and mask a local update, then sign it.

        ``clip=False`` skips the L2 clip and is how the tests and the demo model a *malicious*
        client: nothing in the protocol can force a client to clip, which is precisely why the
        coordinator needs the aggregate norm bound.
        """
        state = self.round
        if state.pairing is None:
            raise ProtocolStateError("cannot submit an input before the pairing set is known")
        if state.masking_set is None:
            raise ProtocolStateError("cannot submit an input before the masking set is known")
        if state.submitted_input:
            raise ProtocolStateError(f"{self.client_id} already submitted an input this round")
        if scheme.dimension != state.pairing.dimension:
            raise ProtocolStateError("scheme dimension does not match the pairing set")
        missing = set(state.peer_mask_public) - set(state.held_shares)
        if missing:
            raise ProtocolStateError(
                "refusing to mask before shares arrived from " + ", ".join(sorted(missing))
            )
        deriver = MaskDeriver(round_nonce=state.round_nonce, dimension=scheme.dimension)
        encoded = scheme.encode(update) if clip else scheme.encode_raw(update)
        masked = mask_vector(
            deriver,
            encoded,
            own_id=self.client_id,
            own_mask_private=state.mask_keys.private_bytes,
            self_mask_seed=state.self_mask_seed,
            peer_mask_public=state.peer_mask_public,
        )
        state.submitted_input = True
        return self._sign(
            MessageKind.MASKED_INPUT,
            MaskedInput(dimension=scheme.dimension, vector=encode_vector(masked)),
        )

    # --- stage 4: UNMASK ---------------------------------------------------------------

    def unmask(self, survivor_message: SignedMessage[SurvivorSet]) -> SignedMessage[UnmaskShares]:
        """Reveal one share per peer: a seed share for survivors, a key share for dropouts.

        The rule that keeps the protocol private is enforced here, on the client: a peer is
        either a survivor or a dropout, never both, and the client never reveals anything about
        itself beyond the seed shares it holds *of other clients*.
        """
        state = self.round
        if state.pairing is None:
            raise ProtocolStateError("cannot unmask before the pairing set is known")
        if not state.submitted_input:
            raise ProtocolStateError("only clients that submitted an input may unmask")
        if state.revealed:
            raise ProtocolStateError(f"{self.client_id} already revealed shares this round")
        verify_message(
            survivor_message,
            public_key=self._coordinator_identity_key,
            expect_kind=MessageKind.SURVIVOR_SET,
            expect_round_id=state.round_id,
            expect_round_nonce=state.round_nonce,
        )
        survivors = set(survivor_message.body.survivors)
        dropouts = set(survivor_message.body.dropouts)
        overlap = survivors & dropouts
        if overlap:
            raise ProtocolMessageError(
                "coordinator listed clients as both survivor and dropout: "
                + ", ".join(sorted(overlap))
            )
        if state.masking_set is None:
            raise ProtocolStateError("cannot unmask before the masking set is known")
        members = set(state.masking_set)
        if (survivors | dropouts) != members:
            raise ProtocolMessageError("survivor set does not partition the masking set")
        if self.client_id not in survivors:
            raise ProtocolMessageError("this client is not listed as a survivor")
        if len(survivors) < state.pairing.threshold:
            raise ProtocolMessageError(
                f"only {len(survivors)} survivors, below threshold {state.pairing.threshold}; "
                "refusing to reveal shares"
            )

        self_seed_shares = {
            peer: payload.self_seed_share
            for peer, payload in state.held_shares.items()
            if peer in survivors
        }
        mask_key_shares = {
            peer: payload.mask_key_share
            for peer, payload in state.held_shares.items()
            if peer in dropouts
        }
        state.revealed = True
        return self._sign(
            MessageKind.UNMASK_SHARES,
            UnmaskShares(
                share_index=state.pairing.share_index(self.client_id),
                self_seed_shares=self_seed_shares,
                mask_key_shares=mask_key_shares,
            ),
        )

    def _sign[
        T: (
            KeyAdvertisement,
            ShareBundle,
            MaskedInput,
            UnmaskShares,
        )
    ](self, kind: MessageKind, body: T) -> SignedMessage[T]:
        state = self.round
        return sign_message(
            body,
            kind=kind,
            round_id=state.round_id,
            round_nonce=state.round_nonce,
            sender=self.client_id,
            identity=self._identity,
        )
