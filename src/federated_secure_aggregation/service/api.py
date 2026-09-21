"""The coordinator as an HTTP service.

The routes are a thin shell: every protocol decision lives in
:class:`~federated_secure_aggregation.secagg.coordinator.SecAggCoordinator`, and the shell only
translates its exceptions into status codes. That separation is deliberate - it means the
property tests that hammer the state machine are testing the same code the service runs, and it
keeps the transport free of security-relevant logic.

Transport security comes from the baseline app factory (API key, rate limit, body size limit,
security headers, no-leak errors). Protocol security does *not* depend on it: every message is
signed by its sender's identity key and bound to the round nonce, so a caller who steals the API
key still cannot impersonate a silo, substitute a key or replay a message from another round.

Two route groups exist:

- **participant routes** carry the four client message kinds and read a client's share inbox;
- **operator routes** open a round and advance it through its stages. Advancing is explicit
  rather than implicit-on-timer so the state machine has no background tasks and no clock of its
  own, which is what makes the whole thing testable without sleeping.

Nothing here returns an individual masked vector to anyone. The only vector that leaves the
service is the noised aggregate, to the operator, after the round is accepted.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request
from pydantic import BaseModel, ConfigDict, Field

from federated_secure_aggregation.core.app import create_app
from federated_secure_aggregation.core.config import Settings
from federated_secure_aggregation.core.errors import AppError
from federated_secure_aggregation.core.logging import get_logger
from federated_secure_aggregation.core.security import (
    AuthenticatedPrincipal,
    OperatorPrincipal,
    SiloPrincipal,
)
from federated_secure_aggregation.monitoring.metrics import RoundMetrics, create_round_metrics
from federated_secure_aggregation.secagg.coordinator import RoundResult, SecAggCoordinator
from federated_secure_aggregation.secagg.errors import ProtocolStateError, QuorumError, SecAggError
from federated_secure_aggregation.secagg.messages import (
    KeyAdvertisement,
    MaskedInput,
    MessageKind,
    ProtocolMessageError,
    RoundStage,
    ShareBundle,
    SignedMessage,
    UnmaskShares,
)

__all__ = ["CoordinatorService", "create_coordinator_app"]

_log = get_logger("federated_secure_aggregation.service")


class StartRoundRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    round_id: int = Field(ge=0)


class RoundOpened(BaseModel):
    round_id: int
    round_nonce: str
    stage: RoundStage
    threshold: int
    dimension: int
    clip_norm: float
    scale: float


class Accepted(BaseModel):
    accepted: bool = True
    stage: RoundStage


class RoundStatus(BaseModel):
    round_id: int
    stage: RoundStage
    registered_clients: list[str]
    bytes_received: int
    # Published so a silo that starts late or restarts can join the round in progress. All of
    # these are agreement parameters, not secrets: the nonce is a salt, and the scheme has to
    # be identical across silos or the ring sum is meaningless.
    round_nonce: str
    threshold: int
    dimension: int
    clip_norm: float
    scale: float
    stage_answers: int


class Inbox(BaseModel):
    client_id: str
    ciphertexts: dict[str, str]


class RoundOutcomeResponse(BaseModel):
    round_id: int
    stage: RoundStage
    survivors: list[str]
    dropouts: list[str]
    accepted: bool
    aggregate_norm: float
    norm_bound: float
    bytes_received: int
    digest: str
    reason: str | None = None
    corrupt_share_holders: list[str] = Field(default_factory=list)


class CoordinatorService:
    """Holds the coordinator plus the per-round relay state the HTTP layer needs."""

    def __init__(self, coordinator: SecAggCoordinator, metrics: RoundMetrics | None = None) -> None:
        self.coordinator = coordinator
        self.metrics = metrics if metrics is not None else create_round_metrics()
        self._inbox: dict[str, dict[str, bytes]] = {}
        self._announcements: dict[MessageKind, dict[str, Any]] = {}
        self._last_result: RoundResult | None = None

    @property
    def last_result(self) -> RoundResult | None:
        return self._last_result

    def announcement(self, kind: MessageKind) -> dict[str, Any]:
        payload = self._announcements.get(kind)
        if payload is None:
            raise AppError(409, "not_available", f"{kind} has not been published yet")
        return payload

    def inbox_for(self, client_id: str) -> dict[str, str]:
        if client_id not in self._inbox:
            raise AppError(404, "not_found", "no shares are waiting for this client")
        return {
            dealer: base64.b64encode(blob).decode("ascii")
            for dealer, blob in sorted(self._inbox[client_id].items())
        }

    def start_round(self, round_id: int) -> RoundOpened:
        nonce = self.coordinator.start_round(round_id)
        self._inbox.clear()
        self._announcements.clear()
        self._last_result = None
        scheme = self.coordinator.scheme
        return RoundOpened(
            round_id=round_id,
            round_nonce=base64.b64encode(nonce).decode("ascii"),
            stage=self.coordinator.stage,
            threshold=self.coordinator.config.threshold,
            dimension=scheme.dimension,
            clip_norm=scheme.clip_norm,
            scale=scheme.scale,
        )

    def advance(self) -> dict[str, Any]:
        """Close the current stage and publish whatever that produces."""
        stage = self.coordinator.stage
        if stage is RoundStage.ADVERTISE_KEYS:
            pairing = self.coordinator.close_advertise_keys()
            payload = pairing.model_dump(mode="json")
            self._announcements[MessageKind.PAIRING_SET] = payload
            return payload
        if stage is RoundStage.SHARE_KEYS:
            masking, inbox = self.coordinator.close_share_keys()
            self._inbox = {client: dict(entries) for client, entries in inbox.items()}
            payload = masking.model_dump(mode="json")
            self._announcements[MessageKind.MASKING_SET] = payload
            return payload
        if stage is RoundStage.MASKED_INPUT:
            survivors = self.coordinator.close_masked_input()
            payload = survivors.model_dump(mode="json")
            self._announcements[MessageKind.SURVIVOR_SET] = payload
            return payload
        if stage is RoundStage.UNMASK:
            result = self.coordinator.finalize()
            self._last_result = result
            self.metrics.observe_round(
                duration_seconds=result.duration_seconds,
                participants=len(result.survivors),
                dropouts=len(result.dropouts),
                accepted=result.accepted,
                failed=result.stage is RoundStage.FAILED,
                epsilon_spent=0.0,
                aggregate_norm=result.aggregate_norm,
                holdout_auc=None,
                bytes_received=result.bytes_received,
            )
            return RoundOutcomeResponse(
                round_id=result.round_id,
                stage=result.stage,
                survivors=list(result.survivors),
                dropouts=list(result.dropouts),
                accepted=result.accepted,
                aggregate_norm=result.aggregate_norm,
                norm_bound=result.norm_bound,
                bytes_received=result.bytes_received,
                digest=result.digest(),
                reason=result.reason,
                corrupt_share_holders=list(result.corrupt_share_holders),
            ).model_dump(mode="json")
        raise AppError(409, "wrong_stage", f"round stage {stage} cannot be advanced")


def _translate(error: SecAggError | ProtocolMessageError) -> AppError:
    """Map a protocol failure to a status code without echoing internals back."""
    if isinstance(error, ProtocolStateError):
        return AppError(409, "wrong_stage", str(error))
    if isinstance(error, QuorumError):
        return AppError(409, "quorum_not_met", str(error))
    return AppError(400, "invalid_message", str(error))


def create_coordinator_app(
    service: CoordinatorService, settings: Settings | None = None
) -> FastAPI:
    """Build the hardened FastAPI app with the coordinator routes mounted on it.

    The HTTP metrics are registered into the *service's* collector registry rather than a fresh
    one, so ``/metrics`` renders the round metrics (duration, participants, dropouts, rejected
    rounds, epsilon spent, holdout AUC) alongside them. Two registries would leave the protocol
    metrics collected but never scraped, which is the failure mode this wiring exists to avoid.
    """
    app = create_app(
        settings,
        readiness_checks={"round_machine": lambda: service.coordinator.stage is not None},
        registry=service.metrics.registry,
    )
    app.state.coordinator_service = service

    def _guard[T](action: Callable[[], T]) -> T:
        """Run a protocol call and translate its exceptions into HTTP errors."""
        try:
            return action()
        except (SecAggError, ProtocolMessageError) as exc:
            _log.info("protocol_rejected", error_type=type(exc).__name__)
            raise _translate(exc) from None

    @app.post("/v1/rounds", response_model=RoundOpened)
    async def start_round(body: StartRoundRequest, principal: OperatorPrincipal) -> RoundOpened:
        """Operator route: open a round and publish its nonce."""
        _log.info("round_started", key_id=principal.key_id, round_id=body.round_id)
        return _guard(lambda: service.start_round(body.round_id))

    @app.get("/v1/rounds/current", response_model=RoundStatus)
    async def round_status(principal: AuthenticatedPrincipal) -> RoundStatus:
        del principal
        coordinator = service.coordinator
        result = service.last_result
        return RoundStatus(
            round_id=_guard(lambda: coordinator.round_id),
            stage=coordinator.stage,
            registered_clients=list(coordinator.registered_clients),
            bytes_received=result.bytes_received if result is not None else 0,
            round_nonce=base64.b64encode(_guard(lambda: coordinator.round_nonce)).decode("ascii"),
            threshold=coordinator.config.threshold,
            dimension=coordinator.scheme.dimension,
            clip_norm=coordinator.scheme.clip_norm,
            scale=coordinator.scheme.scale,
            stage_answers=coordinator.stage_answers,
        )

    @app.post("/v1/rounds/advance")
    async def advance(request: Request, principal: OperatorPrincipal) -> dict[str, Any]:
        """Operator route: close the current stage and return what it published."""
        del request
        _log.info("stage_advance", key_id=principal.key_id, stage=str(service.coordinator.stage))
        return _guard(service.advance)

    @app.post("/v1/rounds/key-advertisements", response_model=Accepted)
    async def key_advertisement(
        message: SignedMessage[KeyAdvertisement], principal: SiloPrincipal
    ) -> Accepted:
        del principal
        _guard(lambda: service.coordinator.submit_key_advertisement(message))
        return Accepted(stage=service.coordinator.stage)

    @app.get("/v1/rounds/pairing-set")
    async def pairing_set(principal: AuthenticatedPrincipal) -> dict[str, Any]:
        del principal
        return service.announcement(MessageKind.PAIRING_SET)

    @app.post("/v1/rounds/share-bundles", response_model=Accepted)
    async def share_bundle(
        message: SignedMessage[ShareBundle], principal: SiloPrincipal
    ) -> Accepted:
        del principal
        _guard(lambda: service.coordinator.submit_share_bundle(message))
        return Accepted(stage=service.coordinator.stage)

    @app.get("/v1/rounds/masking-set")
    async def masking_set(principal: AuthenticatedPrincipal) -> dict[str, Any]:
        del principal
        return service.announcement(MessageKind.MASKING_SET)

    @app.get("/v1/rounds/inbox/{client_id}", response_model=Inbox)
    async def inbox(client_id: str, principal: SiloPrincipal) -> Inbox:
        del principal
        return Inbox(client_id=client_id, ciphertexts=service.inbox_for(client_id))

    @app.post("/v1/rounds/masked-inputs", response_model=Accepted)
    async def masked_input(
        message: SignedMessage[MaskedInput], principal: SiloPrincipal
    ) -> Accepted:
        del principal
        _guard(lambda: service.coordinator.submit_masked_input(message))
        return Accepted(stage=service.coordinator.stage)

    @app.get("/v1/rounds/survivor-set")
    async def survivor_set(principal: AuthenticatedPrincipal) -> dict[str, Any]:
        del principal
        return service.announcement(MessageKind.SURVIVOR_SET)

    @app.post("/v1/rounds/unmask-shares", response_model=Accepted)
    async def unmask_shares(
        message: SignedMessage[UnmaskShares], principal: SiloPrincipal
    ) -> Accepted:
        del principal
        _guard(lambda: service.coordinator.submit_unmask_shares(message))
        return Accepted(stage=service.coordinator.stage)

    return app
