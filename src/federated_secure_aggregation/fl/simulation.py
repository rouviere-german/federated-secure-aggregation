"""The federated run: rounds of local training, secure aggregation, noise and evaluation.

One round is:

1. every participant computes a raw update from the current global parameters;
2. each update is clipped to ``clip_norm`` (this is the DP sensitivity and the norm-bound unit);
3. the updates are summed - either through the full secure-aggregation protocol or, for the
   experiments that do not need the transport, by plain addition;
4. the coordinator applies the aggregate norm bound and rolls the round back if it fails;
5. Gaussian noise is added to the accepted sum, the budget ledger is charged, and the global
   parameters move by ``server_lr * (noised_sum / survivors)``;
6. the round is evaluated on the untouched holdout and appended to the hash chain.

``AggregationMode.SECURE`` runs the real protocol: X25519 key agreement, pairwise masks, Shamir
shares, dropout recovery, signatures. ``AggregationMode.PLAINTEXT`` sums the same clipped
updates directly. The two are not "the real one and a toy": they are proven to produce the same
global model for the same seed in ``tests/integration/test_aggregation_modes.py``, which is what
makes it legitimate to run the longer privacy and utility experiments on the faster path.

Rejected rounds are rolled back completely: the global parameters, the holdout metrics and the
privacy budget are all left exactly as they were, because a round whose aggregate was discarded
released nothing.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.evaluation.metrics import log_loss, roc_auc
from federated_secure_aggregation.fl.data import derive_seed
from federated_secure_aggregation.fl.models import FederatedModel, sigmoid
from federated_secure_aggregation.fl.participants import Participant
from federated_secure_aggregation.monitoring.chain import RoundAuditLog, RoundRecord
from federated_secure_aggregation.monitoring.metrics import RoundMetrics
from federated_secure_aggregation.privacy.ledger import BudgetExhaustedError, BudgetLedger
from federated_secure_aggregation.privacy.noise import (
    MechanismVariant,
    add_central_noise,
    clip_update,
)
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import (
    RoundConfig,
    RoundResult,
    SecAggCoordinator,
)
from federated_secure_aggregation.secagg.errors import QuorumError
from federated_secure_aggregation.secagg.messages import RoundStage
from federated_secure_aggregation.secagg.quantize import AggregationScheme, add_mod, dequantize
from federated_secure_aggregation.secagg.session import (
    SimulationClock,
    TranscriptRecord,
    run_round,
)

__all__ = [
    "AggregationMode",
    "FederatedConfig",
    "FederatedRun",
    "RoundOutcome",
    "RunSummary",
]

FloatArray = npt.NDArray[np.float64]


class AggregationMode(StrEnum):
    """How the per-round sum is computed."""

    SECURE = "secure"
    PLAINTEXT = "plaintext"


@dataclass(frozen=True, slots=True)
class FederatedConfig:
    """Everything a run needs that is not the data or the participants."""

    rounds: int = 20
    clip_norm: float = 1.0
    noise_multiplier: float = 0.0
    target_epsilon: float = 8.0
    delta: float = 1e-5
    threshold: int = 3
    server_learning_rate: float = 1.0
    aggregation: AggregationMode = AggregationMode.SECURE
    variant: MechanismVariant = MechanismVariant.CORRECT
    stage_timeout_seconds: float = 10.0
    seed: int = 20260920
    enforce_budget: bool = True
    # Relaxation of the aggregate norm bound, as a multiple of ``survivors * clip_norm``. It is
    # 1.0 (the tight honest bound) everywhere except in the privacy-audit runs, which raise it
    # so that an integrity control cannot mask the privacy bug they are measuring.
    norm_bound_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.clip_norm <= 0:
            raise ValueError("clip_norm must be > 0")
        if self.noise_multiplier < 0:
            raise ValueError("noise_multiplier must be >= 0")
        if self.server_learning_rate <= 0:
            raise ValueError("server_learning_rate must be > 0")
        if self.threshold < 2:
            raise ValueError("threshold must be >= 2")
        if self.norm_bound_multiplier < 1.0:
            raise ValueError("norm_bound_multiplier must be >= 1")
        if self.enforce_budget and self.noise_multiplier <= 0:
            raise ValueError(
                "a budget can only be enforced when noise is added; set noise_multiplier > 0 "
                "or enforce_budget=False for the epsilon=infinity reference run"
            )


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """One round's outcome, from both the protocol's and the model's point of view."""

    round_id: int
    stage: RoundStage
    survivors: tuple[str, ...]
    dropouts: tuple[str, ...]
    accepted: bool
    aggregate_norm: float
    norm_bound: float
    epsilon_spent: float
    holdout_auc: float | None
    holdout_log_loss: float | None
    bytes_received: int
    duration_seconds: float
    reason: str | None = None


@dataclass(slots=True)
class RunSummary:
    """The result of a whole run."""

    rounds: tuple[RoundOutcome, ...]
    initial_params: FloatArray
    final_params: FloatArray
    holdout_auc: float
    holdout_log_loss: float
    epsilon_spent: float
    accepted_rounds: int
    rejected_rounds: int
    failed_rounds: int
    stopped_early: bool = False
    audit_head: str | None = None
    aggregate_norms: tuple[float, ...] = ()
    holdout_scores: FloatArray = field(default_factory=lambda: np.zeros(0))

    @property
    def model_delta(self) -> FloatArray:
        return self.final_params - self.initial_params


class FederatedRun:
    """Owns the participants, the coordinator, the ledger and the audit chain for one run."""

    def __init__(
        self,
        model: FederatedModel,
        participants: Sequence[Participant],
        holdout_features: FloatArray,
        holdout_labels: FloatArray,
        config: FederatedConfig,
        *,
        initial_params: FloatArray | None = None,
        metrics: RoundMetrics | None = None,
        audit_log: RoundAuditLog | None = None,
        ledger_path: Path | None = None,
    ) -> None:
        if len(participants) < config.threshold:
            raise ValueError(
                f"{len(participants)} participants is below the threshold {config.threshold}"
            )
        ids = [participant.participant_id for participant in participants]
        if len(set(ids)) != len(ids):
            raise ValueError("participant ids must be unique")
        self.model = model
        self.participants = {
            participant.participant_id: participant for participant in participants
        }
        self.config = config
        self.holdout_features = np.asarray(holdout_features, dtype=np.float64)
        self.holdout_labels = np.asarray(holdout_labels, dtype=np.float64)
        self.metrics = metrics
        self.audit_log = audit_log
        self.params = (
            model.initial_params() if initial_params is None else np.asarray(initial_params).copy()
        )
        self.initial_params = self.params.copy()
        self._rng = np.random.default_rng(derive_seed(config.seed, "central-noise"))
        self.ledger = (
            BudgetLedger(
                noise_multiplier=config.noise_multiplier,
                target_epsilon=config.target_epsilon,
                delta=config.delta,
                path=ledger_path,
            )
            if config.enforce_budget
            else None
        )
        self._clock = SimulationClock()
        self._coordinator: SecAggCoordinator | None = None
        self._clients: dict[str, SecAggClient] = {}
        if config.aggregation is AggregationMode.SECURE:
            self._build_protocol()

    # --- setup -------------------------------------------------------------------------

    def _build_protocol(self) -> None:
        identities = {name: IdentityKeyPair.generate() for name in sorted(self.participants)}
        public = {name: identity.public_bytes for name, identity in identities.items()}
        coordinator_identity = IdentityKeyPair.generate()
        self._coordinator = SecAggCoordinator(
            coordinator_identity,
            public,
            RoundConfig(
                dimension=self.model.dimension,
                threshold=self.config.threshold,
                clip_norm=self.config.clip_norm,
                max_clients=len(self.participants),
                stage_timeout_seconds=self.config.stage_timeout_seconds,
                norm_bound_slack=self.config.norm_bound_multiplier - 1.0,
            ),
            clock=self._clock,
        )
        self._clients = {
            name: SecAggClient(
                name,
                identities[name],
                peer_identity_keys=public,
                coordinator_identity_key=coordinator_identity.public_bytes,
            )
            for name in sorted(self.participants)
        }

    # --- running -----------------------------------------------------------------------

    def run(
        self,
        *,
        dropouts: Mapping[int, Sequence[str]] | None = None,
        transcript: TranscriptRecord | None = None,
    ) -> RunSummary:
        """Run every configured round, stopping early if the privacy budget runs out."""
        schedule = {int(key): tuple(value) for key, value in (dropouts or {}).items()}
        outcomes: list[RoundOutcome] = []
        norms: list[float] = []
        stopped_early = False
        for round_id in range(self.config.rounds):
            if self.ledger is not None and not self.ledger.can_run_round():
                stopped_early = True
                break
            try:
                outcome = self.run_one_round(
                    round_id, dropouts=schedule.get(round_id, ()), transcript=transcript
                )
            except BudgetExhaustedError:  # pragma: no cover - guarded by can_run_round
                stopped_early = True
                break
            outcomes.append(outcome)
            norms.append(outcome.aggregate_norm)
        scores = self._scores()
        return RunSummary(
            rounds=tuple(outcomes),
            initial_params=self.initial_params,
            final_params=self.params.copy(),
            holdout_auc=roc_auc(scores, self.holdout_labels),
            holdout_log_loss=log_loss(scores, self.holdout_labels),
            epsilon_spent=self.ledger.epsilon_spent() if self.ledger is not None else float("inf"),
            accepted_rounds=sum(1 for outcome in outcomes if outcome.accepted),
            rejected_rounds=sum(1 for outcome in outcomes if outcome.stage is RoundStage.REJECTED),
            failed_rounds=sum(1 for outcome in outcomes if outcome.stage is RoundStage.FAILED),
            stopped_early=stopped_early,
            audit_head=self.audit_log.head if self.audit_log is not None else None,
            aggregate_norms=tuple(norms),
            holdout_scores=scores,
        )

    def run_one_round(
        self,
        round_id: int,
        *,
        dropouts: Sequence[str] = (),
        transcript: TranscriptRecord | None = None,
    ) -> RoundOutcome:
        """Run a single round and apply it (or roll it back)."""
        updates: dict[str, FloatArray] = {}
        for name, participant in sorted(self.participants.items()):
            raw = participant.contribute(self.params, round_id)
            # Clipping happens exactly once, here, and only for clients running the reference
            # implementation - whose clipping step the configured ``variant`` may have broken.
            # A malicious client simply does not clip, and nothing in the protocol can make it.
            updates[name] = (
                clip_update(raw, self.config.clip_norm, self.config.variant)
                if participant.honest
                else np.asarray(raw, dtype=np.float64).ravel()
            )
        if self.config.aggregation is AggregationMode.SECURE:
            result = self._aggregate_secure(round_id, updates, dropouts, transcript)
        else:
            result = self._aggregate_plaintext(round_id, updates, dropouts)
        return self._apply(result)

    def _aggregate_plaintext(
        self,
        round_id: int,
        updates: Mapping[str, FloatArray],
        dropouts: Sequence[str],
    ) -> RoundResult:
        """Sum the same ring-encoded updates the protocol would, without masking them.

        The encode/decode round trip is kept so this path is numerically identical to the
        secure one rather than merely similar: the only thing removed is the masking.
        """
        dropped = set(dropouts)
        survivors = tuple(name for name in sorted(updates) if name not in dropped)
        if len(survivors) < self.config.threshold:
            raise QuorumError(
                f"only {len(survivors)} survivors, below threshold {self.config.threshold}"
            )
        started = self._clock()
        scheme = AggregationScheme.create(
            dimension=self.model.dimension,
            clip_norm=self.config.clip_norm,
            max_clients=len(self.participants),
        )
        ring = np.zeros(self.model.dimension, dtype=np.uint32)
        for name in survivors:
            ring = add_mod(ring, scheme.encode_raw(updates[name]))
        total = dequantize(ring, scheme.scale)
        norm = float(np.linalg.norm(total))
        bound = len(survivors) * self.config.clip_norm * self.config.norm_bound_multiplier
        rejected = norm > bound
        return RoundResult(
            round_id=round_id,
            stage=RoundStage.REJECTED if rejected else RoundStage.COMPLETED,
            survivors=survivors,
            dropouts=tuple(sorted(dropped & set(updates))),
            excluded=(),
            aggregate=None if rejected else np.asarray(total, dtype=np.float64),
            aggregate_norm=norm,
            norm_bound=bound,
            bytes_received=4 * self.model.dimension * len(survivors),
            duration_seconds=self._clock() - started,
            reason=(
                f"aggregate L2 norm {norm:.4f} exceeds the bound {bound:.4f}; round rolled back"
                if rejected
                else None
            ),
        )

    def _aggregate_secure(
        self,
        round_id: int,
        updates: Mapping[str, FloatArray],
        dropouts: Sequence[str],
        transcript: TranscriptRecord | None,
    ) -> RoundResult:
        if self._coordinator is None:  # pragma: no cover - set in __init__
            raise RuntimeError("the secure-aggregation protocol was not initialised")
        return run_round(
            self._coordinator,
            self._clients,
            updates,
            round_id,
            clock=self._clock,
            drop_before_masked_input=tuple(dropouts),
            clip_in_client=False,
            transcript=transcript,
        )

    def _apply(self, result: RoundResult) -> RoundOutcome:
        """Move the global model, charge the budget and record the round."""
        epsilon = self.ledger.epsilon_spent() if self.ledger is not None else float("inf")
        auc: float | None = None
        loss: float | None = None
        if result.accepted and result.aggregate is not None:
            if self.ledger is not None:
                epsilon = self.ledger.charge_round()
            survivors = max(len(result.survivors), 1)
            noised = add_central_noise(
                result.aggregate,
                self._rng,
                noise_multiplier=self.config.noise_multiplier,
                clip_norm=self.config.clip_norm,
                survivors=survivors,
                variant=self.config.variant,
            )
            self.params = self.params + self.config.server_learning_rate * (noised / survivors)
            scores = self._scores()
            auc = roc_auc(scores, self.holdout_labels)
            loss = log_loss(scores, self.holdout_labels)

        outcome = RoundOutcome(
            round_id=result.round_id,
            stage=result.stage,
            survivors=result.survivors,
            dropouts=result.dropouts,
            accepted=result.accepted,
            aggregate_norm=result.aggregate_norm,
            norm_bound=result.norm_bound,
            epsilon_spent=epsilon,
            holdout_auc=auc,
            holdout_log_loss=loss,
            bytes_received=result.bytes_received,
            duration_seconds=result.duration_seconds,
            reason=result.reason,
        )
        self._record(result, outcome)
        return outcome

    def _record(self, result: RoundResult, outcome: RoundOutcome) -> None:
        if self.metrics is not None:
            self.metrics.observe_round(
                duration_seconds=outcome.duration_seconds,
                participants=len(outcome.survivors),
                dropouts=len(outcome.dropouts),
                accepted=outcome.accepted,
                failed=outcome.stage is RoundStage.FAILED,
                epsilon_spent=0.0 if np.isinf(outcome.epsilon_spent) else outcome.epsilon_spent,
                aggregate_norm=outcome.aggregate_norm,
                holdout_auc=outcome.holdout_auc,
                bytes_received=outcome.bytes_received,
            )
        if self.audit_log is not None:
            self.audit_log.append(
                RoundRecord(
                    round_id=outcome.round_id,
                    stage=str(outcome.stage),
                    survivors=list(outcome.survivors),
                    dropouts=list(outcome.dropouts),
                    excluded=list(result.excluded),
                    accepted=outcome.accepted,
                    aggregate_digest=result.digest(),
                    aggregate_norm=outcome.aggregate_norm,
                    norm_bound=outcome.norm_bound,
                    epsilon_spent=(
                        -1.0 if np.isinf(outcome.epsilon_spent) else outcome.epsilon_spent
                    ),
                    holdout_auc=outcome.holdout_auc,
                    bytes_received=outcome.bytes_received,
                    duration_seconds=max(outcome.duration_seconds, 0.0),
                    reason=outcome.reason,
                    corrupt_share_holders=list(result.corrupt_share_holders),
                )
            )

    def _scores(self) -> FloatArray:
        return sigmoid(self.model.logits(self.params, self.holdout_features))
