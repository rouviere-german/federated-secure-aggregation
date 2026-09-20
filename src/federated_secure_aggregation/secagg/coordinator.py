"""The coordinator half of the round state machine.

The coordinator is *honest but curious* in this threat model: it follows the protocol and may
try to learn from what it sees, but it does not forge messages. What it sees is the point of
the design: masked vectors, one revealed secret per client, and the sum. It never holds a
plaintext individual update, and :func:`masking.reconstruct_aggregate` refuses the combination
of secrets that would produce one.

The round runs as a strict state machine::

    ADVERTISE_KEYS -> SHARE_KEYS -> MASKED_INPUT -> UNMASK -> AGGREGATE -> COMPLETED
                                                                        \\-> REJECTED
    (any stage below quorum)                                             -> FAILED

Each stage closes when every expected client has answered or when its timeout elapses,
whichever comes first, and a stage that closes below the threshold ``t`` fails the round rather
than continuing with too few participants. Time comes from an injected monotonic clock, so
timeout behaviour is tested deterministically rather than by sleeping.

Integrity controls applied on the way in: signature and context verification for every message,
one message per client per stage (a second is refused, not overwritten), a declared-dimension
check, and a byte budget per round. Integrity control applied on the way out: the unmasked
aggregate must satisfy ``||sum|| <= survivors * (clip_norm + quantization slack)``, otherwise
the round is REJECTED and the caller keeps the previous model. Because the coordinator cannot
inspect individual updates, that check is necessarily aggregate-level: it cannot say *which*
client misbehaved. Attribution is a documented non-goal.
"""

from __future__ import annotations

import math
import os
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.core.canonical import canonical_json, sha256_hex
from federated_secure_aggregation.crypto import shamir
from federated_secure_aggregation.crypto.keys import IdentityKeyPair, x25519_public_bytes
from federated_secure_aggregation.crypto.prg import PRG_SEED_BYTES
from federated_secure_aggregation.secagg.errors import (
    AggregateRejectedError,
    ProtocolStateError,
    QuorumError,
)
from federated_secure_aggregation.secagg.masking import MaskDeriver, reconstruct_aggregate
from federated_secure_aggregation.secagg.messages import (
    KeyAdvertisement,
    MaskedInput,
    MaskingSet,
    MessageKind,
    PairingSet,
    ProtocolMessageError,
    RoundStage,
    ShareBundle,
    SignedMessage,
    SurvivorSet,
    UnmaskShares,
    decode_vector,
    sign_message,
    verify_message,
)
from federated_secure_aggregation.secagg.quantize import (
    AggregationScheme,
    RingArray,
    dequantize,
)

__all__ = ["RoundConfig", "RoundResult", "SecAggCoordinator"]

FloatArray = npt.NDArray[np.float64]
Clock = Callable[[], float]
RandomBytes = Callable[[int], bytes]

NONCE_BYTES = 16


@dataclass(frozen=True, slots=True)
class RoundConfig:
    """Parameters fixed for the whole run."""

    dimension: int
    threshold: int
    clip_norm: float = 1.0
    max_clients: int = 32
    stage_timeout_seconds: float = 30.0
    max_round_bytes: int = 64 << 20
    norm_bound_slack: float = 0.0

    def __post_init__(self) -> None:
        if self.threshold < 2:
            raise ValueError("threshold must be >= 2; a 1-of-n scheme protects nothing")
        if self.threshold > self.max_clients:
            raise ValueError("threshold cannot exceed max_clients")
        if self.stage_timeout_seconds <= 0:
            raise ValueError("stage_timeout_seconds must be > 0")
        if self.norm_bound_slack < 0:
            raise ValueError("norm_bound_slack must be >= 0")

    def scheme(self) -> AggregationScheme:
        return AggregationScheme.create(
            dimension=self.dimension, clip_norm=self.clip_norm, max_clients=self.max_clients
        )


@dataclass(frozen=True, slots=True)
class RoundResult:
    """What one round produced, whether it succeeded or not."""

    round_id: int
    stage: RoundStage
    survivors: tuple[str, ...]
    dropouts: tuple[str, ...]
    excluded: tuple[str, ...]
    aggregate: FloatArray | None
    aggregate_norm: float
    norm_bound: float
    bytes_received: int
    duration_seconds: float
    reason: str | None = None

    @property
    def accepted(self) -> bool:
        return self.stage is RoundStage.COMPLETED

    def digest(self) -> str:
        """Content hash of the round outcome, used by the hash-chained audit log."""
        payload = {
            "round_id": self.round_id,
            "stage": str(self.stage),
            "survivors": list(self.survivors),
            "dropouts": list(self.dropouts),
            "excluded": list(self.excluded),
            "aggregate_sha256": (
                sha256_hex(np.ascontiguousarray(self.aggregate, dtype="<f8").tobytes())
                if self.aggregate is not None
                else None
            ),
            "aggregate_norm": round(self.aggregate_norm, 9),
            "norm_bound": round(self.norm_bound, 9),
            "reason": self.reason,
        }
        return sha256_hex(canonical_json(payload))


@dataclass(slots=True)
class _RoundState:
    round_id: int
    nonce: bytes
    stage: RoundStage
    stage_started: float
    advertisements: dict[str, SignedMessage[KeyAdvertisement]] = field(default_factory=dict)
    pairing: PairingSet | None = None
    bundles: dict[str, ShareBundle] = field(default_factory=dict)
    masking_members: tuple[str, ...] = ()
    masked_inputs: dict[str, RingArray] = field(default_factory=dict)
    survivors: tuple[str, ...] = ()
    dropouts: tuple[str, ...] = ()
    unmask_responses: dict[str, UnmaskShares] = field(default_factory=dict)
    bytes_received: int = 0
    started: float = 0.0


class SecAggCoordinator:
    """Runs one round at a time over a fixed roster of clients."""

    def __init__(
        self,
        identity: IdentityKeyPair,
        client_identity_keys: Mapping[str, bytes],
        config: RoundConfig,
        *,
        clock: Clock = time.monotonic,
        random_bytes: RandomBytes = os.urandom,
    ) -> None:
        if len(client_identity_keys) < config.threshold:
            raise ValueError(
                f"{len(client_identity_keys)} registered clients is below threshold "
                f"{config.threshold}"
            )
        if len(client_identity_keys) > config.max_clients:
            raise ValueError("more registered clients than max_clients allows")
        self.identity = identity
        self.config = config
        self.scheme = config.scheme()
        self._client_keys = dict(client_identity_keys)
        self._clock = clock
        self._random_bytes = random_bytes
        self._state: _RoundState | None = None
        self._completed_round_ids: set[int] = set()

    # --- introspection -----------------------------------------------------------------

    @property
    def stage(self) -> RoundStage:
        return self._state.stage if self._state is not None else RoundStage.COMPLETED

    @property
    def round_id(self) -> int:
        return self._require_state().round_id

    @property
    def stage_answers(self) -> int:
        """How many clients have answered the current stage.

        Published so an operator can close a stage as soon as everyone has answered instead of
        always waiting out the timeout. It leaks nothing new: the survivor and dropout sets are
        already public to participants, because unmasking cannot work without them.
        """
        state = self._state
        if state is None:
            return 0
        if state.stage is RoundStage.ADVERTISE_KEYS:
            return len(state.advertisements)
        if state.stage is RoundStage.SHARE_KEYS:
            return len(state.bundles)
        if state.stage is RoundStage.MASKED_INPUT:
            return len(state.masked_inputs)
        if state.stage is RoundStage.UNMASK:
            return len(state.unmask_responses)
        return 0

    @property
    def round_nonce(self) -> bytes:
        """The open round's nonce.

        Public on purpose: it is the HKDF salt every participant must agree on, and it is
        published to all of them at ADVERTISE_KEYS. Its job is freshness (no mask seed repeats
        across rounds), not secrecy, so exposing it lets a silo that restarted mid-run rejoin
        the round in progress instead of waiting for the next one.
        """
        return self._require_state().nonce

    @property
    def registered_clients(self) -> tuple[str, ...]:
        return tuple(sorted(self._client_keys))

    def _require_state(self) -> _RoundState:
        if self._state is None:
            raise ProtocolStateError("no round in progress")
        return self._state

    def _require_stage(self, stage: RoundStage) -> _RoundState:
        state = self._require_state()
        if state.stage is not stage:
            raise ProtocolStateError(f"round is in stage {state.stage}, not {stage}")
        return state

    def _elapsed(self, state: _RoundState) -> float:
        return self._clock() - state.stage_started

    def _advance(self, state: _RoundState, stage: RoundStage) -> None:
        state.stage = stage
        state.stage_started = self._clock()

    def _charge(self, state: _RoundState, message: SignedMessage[Any]) -> None:
        size = len(message.model_dump_json())
        state.bytes_received += size
        if state.bytes_received > self.config.max_round_bytes:
            self._advance(state, RoundStage.FAILED)
            raise QuorumError(
                f"round {state.round_id} exceeded its {self.config.max_round_bytes} byte budget"
            )

    def _verify(self, message: SignedMessage[Any], kind: MessageKind, state: _RoundState) -> None:
        key = self._client_keys.get(message.sender)
        if key is None:
            raise ProtocolMessageError(f"unknown client {message.sender!r}")
        verify_message(
            message,
            public_key=key,
            expect_kind=kind,
            expect_round_id=state.round_id,
            expect_round_nonce=state.nonce,
            expect_sender=message.sender,
        )

    # --- stage 1: ADVERTISE_KEYS -------------------------------------------------------

    def start_round(self, round_id: int) -> bytes:
        """Open a round and return its fresh nonce, which every message must carry."""
        if round_id in self._completed_round_ids:
            raise ProtocolStateError(f"round {round_id} already ran; nonces are never reused")
        if self._state is not None and self._state.stage not in {
            RoundStage.COMPLETED,
            RoundStage.REJECTED,
            RoundStage.FAILED,
        }:
            raise ProtocolStateError("a round is already in progress")
        now = self._clock()
        nonce = self._random_bytes(NONCE_BYTES)
        self._state = _RoundState(
            round_id=round_id,
            nonce=nonce,
            stage=RoundStage.ADVERTISE_KEYS,
            stage_started=now,
            started=now,
        )
        return nonce

    def submit_key_advertisement(self, message: SignedMessage[KeyAdvertisement]) -> None:
        state = self._require_stage(RoundStage.ADVERTISE_KEYS)
        self._verify(message, MessageKind.KEY_ADVERTISEMENT, state)
        if message.sender in state.advertisements:
            raise ProtocolStateError(f"{message.sender} already advertised keys this round")
        self._charge(state, message)
        state.advertisements[message.sender] = message

    def close_advertise_keys(self) -> SignedMessage[PairingSet]:
        """Freeze the pairing set and publish it, signed and with every advertisement intact."""
        state = self._require_stage(RoundStage.ADVERTISE_KEYS)
        self._require_ready(state, len(state.advertisements), len(self._client_keys))
        if len(state.advertisements) < self.config.threshold:
            self._advance(state, RoundStage.FAILED)
            raise QuorumError(
                f"only {len(state.advertisements)} clients advertised keys, below threshold "
                f"{self.config.threshold}"
            )
        pairing = PairingSet(
            threshold=self.config.threshold,
            dimension=self.config.dimension,
            clip_norm=self.config.clip_norm,
            scale=self.scheme.scale,
            advertisements=dict(state.advertisements),
        )
        state.pairing = pairing
        self._advance(state, RoundStage.SHARE_KEYS)
        return self._sign(MessageKind.PAIRING_SET, pairing, state)

    # --- stage 2: SHARE_KEYS -----------------------------------------------------------

    def submit_share_bundle(self, message: SignedMessage[ShareBundle]) -> None:
        state = self._require_stage(RoundStage.SHARE_KEYS)
        self._verify(message, MessageKind.SHARE_BUNDLE, state)
        pairing = state.pairing
        if pairing is None:  # pragma: no cover - guarded by the stage check
            raise ProtocolStateError("pairing set is missing")
        if message.sender not in pairing.participants:
            raise ProtocolMessageError(f"{message.sender} is not in the pairing set")
        if message.sender in state.bundles:
            raise ProtocolStateError(f"{message.sender} already submitted shares this round")
        recipients = set(message.body.ciphertexts)
        expected = set(pairing.participants) - {message.sender}
        if recipients != expected:
            raise ProtocolMessageError(
                f"{message.sender} addressed shares to {sorted(recipients)}, expected "
                f"{sorted(expected)}"
            )
        self._charge(state, message)
        state.bundles[message.sender] = message.body

    def close_share_keys(self) -> tuple[SignedMessage[MaskingSet], dict[str, dict[str, bytes]]]:
        """Publish the masking set and the per-recipient share ciphertexts to relay."""
        state = self._require_stage(RoundStage.SHARE_KEYS)
        pairing = state.pairing
        if pairing is None:  # pragma: no cover - guarded by the stage check
            raise ProtocolStateError("pairing set is missing")
        self._require_ready(state, len(state.bundles), len(pairing.participants))
        members = tuple(sorted(state.bundles))
        if len(members) < self.config.threshold:
            self._advance(state, RoundStage.FAILED)
            raise QuorumError(
                f"only {len(members)} clients delivered shares, below threshold "
                f"{self.config.threshold}"
            )
        inbox: dict[str, dict[str, bytes]] = {member: {} for member in members}
        for dealer in members:
            for recipient, ciphertext in state.bundles[dealer].ciphertexts.items():
                if recipient in inbox:
                    inbox[recipient][dealer] = ciphertext
        state.masking_members = members
        self._advance(state, RoundStage.MASKED_INPUT)
        return self._sign(MessageKind.MASKING_SET, MaskingSet(members=list(members)), state), inbox

    # --- stage 3: MASKED_INPUT ---------------------------------------------------------

    def submit_masked_input(self, message: SignedMessage[MaskedInput]) -> None:
        state = self._require_stage(RoundStage.MASKED_INPUT)
        self._verify(message, MessageKind.MASKED_INPUT, state)
        if message.sender not in state.masking_members:
            raise ProtocolMessageError(f"{message.sender} is not in the masking set")
        if message.sender in state.masked_inputs:
            raise ProtocolStateError(f"{message.sender} already submitted an input this round")
        if message.body.dimension != self.config.dimension:
            raise ProtocolMessageError(
                f"declared dimension {message.body.dimension} != {self.config.dimension}"
            )
        self._charge(state, message)
        state.masked_inputs[message.sender] = decode_vector(
            message.body.vector, self.config.dimension
        )

    def close_masked_input(self) -> SignedMessage[SurvivorSet]:
        """Freeze who survived and publish the partition of the masking set."""
        state = self._require_stage(RoundStage.MASKED_INPUT)
        self._require_ready(state, len(state.masked_inputs), len(state.masking_members))
        survivors = tuple(sorted(state.masked_inputs))
        if len(survivors) < self.config.threshold:
            self._advance(state, RoundStage.FAILED)
            raise QuorumError(
                f"only {len(survivors)} clients submitted inputs, below threshold "
                f"{self.config.threshold}; refusing to unmask"
            )
        state.survivors = survivors
        state.dropouts = tuple(sorted(set(state.masking_members) - set(survivors)))
        self._advance(state, RoundStage.UNMASK)
        return self._sign(
            MessageKind.SURVIVOR_SET,
            SurvivorSet(survivors=list(state.survivors), dropouts=list(state.dropouts)),
            state,
        )

    # --- stage 4: UNMASK ---------------------------------------------------------------

    def submit_unmask_shares(self, message: SignedMessage[UnmaskShares]) -> None:
        state = self._require_stage(RoundStage.UNMASK)
        self._verify(message, MessageKind.UNMASK_SHARES, state)
        if message.sender not in state.survivors:
            raise ProtocolMessageError(f"{message.sender} is not a survivor of this round")
        if message.sender in state.unmask_responses:
            raise ProtocolStateError(f"{message.sender} already revealed shares this round")
        pairing = state.pairing
        if pairing is None:  # pragma: no cover - guarded by the stage check
            raise ProtocolStateError("pairing set is missing")
        if message.body.share_index != pairing.share_index(message.sender):
            raise ProtocolMessageError(f"{message.sender} claimed the wrong share index")
        if set(message.body.self_seed_shares) - set(state.survivors):
            raise ProtocolMessageError("seed shares offered for a non-survivor")
        if set(message.body.mask_key_shares) - set(state.dropouts):
            raise ProtocolMessageError("key shares offered for a client that is not a dropout")
        self._charge(state, message)
        state.unmask_responses[message.sender] = message.body

    def finalize(self) -> RoundResult:
        """Reconstruct the secrets, unmask the sum and apply the aggregate norm bound."""
        state = self._require_stage(RoundStage.UNMASK)
        self._require_ready(state, len(state.unmask_responses), len(state.survivors))
        if len(state.unmask_responses) < self.config.threshold:
            self._advance(state, RoundStage.FAILED)
            self._completed_round_ids.add(state.round_id)
            raise QuorumError(
                f"only {len(state.unmask_responses)} survivors answered the unmask request, "
                f"below threshold {self.config.threshold}"
            )
        pairing = state.pairing
        if pairing is None:  # pragma: no cover - guarded by the stage check
            raise ProtocolStateError("pairing set is missing")
        self._advance(state, RoundStage.AGGREGATE)

        mask_public_keys, _ = self._pairing_keys(pairing, state)
        self_seeds = self._recover_self_seeds(state)
        dropout_privates = self._recover_dropout_keys(state, mask_public_keys)

        deriver = MaskDeriver(round_nonce=state.nonce, dimension=self.config.dimension)
        ring_sum = reconstruct_aggregate(
            deriver,
            {client: state.masked_inputs[client] for client in state.survivors},
            mask_public_keys=mask_public_keys,
            survivor_self_seeds=self_seeds,
            dropout_mask_privates=dropout_privates,
        )
        aggregate = dequantize(ring_sum, self.scheme.scale)
        norm = float(np.linalg.norm(aggregate))
        bound = self.norm_bound(len(state.survivors))
        duration = self._clock() - state.started
        self._completed_round_ids.add(state.round_id)

        if norm > bound:
            self._advance(state, RoundStage.REJECTED)
            return RoundResult(
                round_id=state.round_id,
                stage=RoundStage.REJECTED,
                survivors=state.survivors,
                dropouts=state.dropouts,
                excluded=self._excluded(state),
                aggregate=None,
                aggregate_norm=norm,
                norm_bound=bound,
                bytes_received=state.bytes_received,
                duration_seconds=duration,
                reason=str(AggregateRejectedError(norm, bound)),
            )
        self._advance(state, RoundStage.COMPLETED)
        return RoundResult(
            round_id=state.round_id,
            stage=RoundStage.COMPLETED,
            survivors=state.survivors,
            dropouts=state.dropouts,
            excluded=self._excluded(state),
            aggregate=aggregate,
            aggregate_norm=norm,
            norm_bound=bound,
            bytes_received=state.bytes_received,
            duration_seconds=duration,
        )

    def norm_bound(self, survivors: int) -> float:
        """``survivors * (clip_norm + rounding slack)``, the tightest honest upper bound.

        Every client's update is clipped to ``clip_norm`` before quantization, so by the
        triangle inequality an honest sum of ``k`` updates has L2 norm at most ``k * clip_norm``.
        Quantization can add at most ``0.5 / scale`` per coordinate, hence
        ``sqrt(dimension) * 0.5 / scale`` per client. ``norm_bound_slack`` is an extra relative
        margin, zero by default.
        """
        if survivors < 1:
            raise ValueError("survivors must be >= 1")
        rounding = math.sqrt(self.config.dimension) * self.scheme.resolution
        return survivors * (self.config.clip_norm + rounding) * (1.0 + self.config.norm_bound_slack)

    def abandon(self, reason: str) -> RoundResult:
        """Give up on the current round (used when a stage times out below quorum)."""
        state = self._require_state()
        self._advance(state, RoundStage.FAILED)
        self._completed_round_ids.add(state.round_id)
        return RoundResult(
            round_id=state.round_id,
            stage=RoundStage.FAILED,
            survivors=state.survivors,
            dropouts=state.dropouts,
            excluded=self._excluded(state),
            aggregate=None,
            aggregate_norm=0.0,
            norm_bound=self.norm_bound(max(len(state.survivors), 1)),
            bytes_received=state.bytes_received,
            duration_seconds=self._clock() - state.started,
            reason=reason,
        )

    # --- internals ---------------------------------------------------------------------

    def _excluded(self, state: _RoundState) -> tuple[str, ...]:
        seen = set(state.masking_members) or set(state.advertisements)
        return tuple(sorted(set(self._client_keys) - seen))

    def _require_ready(self, state: _RoundState, received: int, expected: int) -> None:
        if received < expected and self._elapsed(state) < self.config.stage_timeout_seconds:
            raise ProtocolStateError(
                f"stage {state.stage} has {received}/{expected} responses and has not timed out"
            )

    def _pairing_keys(
        self, pairing: PairingSet, state: _RoundState
    ) -> tuple[dict[str, bytes], dict[str, bytes]]:
        return pairing.verified_keys(
            self._client_keys, round_id=state.round_id, round_nonce=state.nonce
        )

    def _recover_self_seeds(self, state: _RoundState) -> dict[str, bytes]:
        pairing = state.pairing
        if pairing is None:
            raise ProtocolStateError("the pairing set has not been published yet")
        seeds: dict[str, bytes] = {}
        for owner in state.survivors:
            shares = [
                shamir.Share(pairing.share_index(holder), response.self_seed_shares[owner])
                for holder, response in sorted(state.unmask_responses.items())
                if owner in response.self_seed_shares
            ]
            if len(shares) < self.config.threshold:
                raise QuorumError(
                    f"only {len(shares)} shares of {owner}'s self-mask seed, below threshold "
                    f"{self.config.threshold}"
                )
            seed = shamir.combine(shares[: self.config.threshold])
            if len(seed) != PRG_SEED_BYTES:
                raise ProtocolMessageError(f"reconstructed seed for {owner} has the wrong length")
            seeds[owner] = seed
        return seeds

    def _recover_dropout_keys(
        self, state: _RoundState, mask_public_keys: Mapping[str, bytes]
    ) -> dict[str, bytes]:
        pairing = state.pairing
        if pairing is None:
            raise ProtocolStateError("the pairing set has not been published yet")
        privates: dict[str, bytes] = {}
        for owner in state.dropouts:
            shares = [
                shamir.Share(pairing.share_index(holder), response.mask_key_shares[owner])
                for holder, response in sorted(state.unmask_responses.items())
                if owner in response.mask_key_shares
            ]
            if len(shares) < self.config.threshold:
                raise QuorumError(
                    f"only {len(shares)} shares of {owner}'s mask key, below threshold "
                    f"{self.config.threshold}"
                )
            private = shamir.combine(shares[: self.config.threshold])
            # The reconstruction is only trustworthy if it matches the signed advertisement.
            # A wrong share (corrupt or malicious) would otherwise silently corrupt the sum.
            if x25519_public_bytes(private) != mask_public_keys[owner]:
                raise ProtocolMessageError(
                    f"reconstructed mask key for {owner} does not match its advertised public key"
                )
            privates[owner] = private
        return privates

    def _sign[T: (PairingSet, MaskingSet, SurvivorSet)](
        self, kind: MessageKind, body: T, state: _RoundState
    ) -> SignedMessage[T]:
        return sign_message(
            body,
            kind=kind,
            round_id=state.round_id,
            round_nonce=state.nonce,
            sender="coordinator",
            identity=self.identity,
        )
