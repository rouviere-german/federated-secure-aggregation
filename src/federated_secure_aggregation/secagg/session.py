"""Driving one full round in process: the reference orchestration of the four stages.

The HTTP service and the simulation both go through this module, so there is a single place
where stage order, timeout handling and dropout injection live. Nothing here trusts the
participants: the coordinator verifies every client message and every client verifies the
coordinator's announcements.

Dropouts are injected by simply *not* calling a client. The coordinator then sees a stage that
cannot complete, waits for the stage timeout and closes with whoever answered, which is exactly
what would happen if a silo's process died.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import RoundResult, SecAggCoordinator
from federated_secure_aggregation.secagg.errors import ProtocolStateError

__all__ = ["SimulationClock", "TranscriptRecord", "run_round"]

FloatArray = npt.NDArray[np.float64]


class SimulationClock:
    """Monotonic clock for in-process runs: real elapsed time plus explicit timeout jumps.

    Round durations therefore stay meaningful (they include the real cost of the cryptography)
    while stage timeouts fire instantly instead of blocking the test suite for 30 seconds.
    """

    def __init__(self) -> None:
        self._origin = time.monotonic()
        self._skew = 0.0

    def __call__(self) -> float:
        return time.monotonic() - self._origin + self._skew

    def skip_timeout(self, seconds: float) -> None:
        if seconds <= 0:
            raise ValueError("timeout skip must be positive")
        self._skew += seconds


@dataclass(slots=True)
class TranscriptRecord:
    """Everything the coordinator received this round, for the payload-inspection tests."""

    masked_vectors: dict[str, FloatArray] = field(default_factory=dict)
    message_json: list[str] = field(default_factory=list)

    def record(self, payload: str) -> None:
        self.message_json.append(payload)


def run_round(
    coordinator: SecAggCoordinator,
    clients: Mapping[str, SecAggClient],
    updates: Mapping[str, FloatArray],
    round_id: int,
    *,
    clock: SimulationClock,
    drop_before_shares: Collection[str] = (),
    drop_before_masked_input: Collection[str] = (),
    drop_before_unmask: Collection[str] = (),
    clip_in_client: bool = True,
    transcript: TranscriptRecord | None = None,
) -> RoundResult:
    """Run ADVERTISE_KEYS -> SHARE_KEYS -> MASKED_INPUT -> UNMASK -> AGGREGATE once.

    ``clip_in_client=False`` says the caller already applied the reference client's L2 clip (or
    deliberately did not), so the client should encode the update as given. Callers that hand
    over raw updates leave it at ``True``.
    """
    silent_shares = set(drop_before_shares)
    silent_inputs = set(drop_before_masked_input) | silent_shares
    silent_unmask = set(drop_before_unmask) | silent_inputs
    timeout = coordinator.config.stage_timeout_seconds

    nonce = coordinator.start_round(round_id)
    for client_id in sorted(clients):
        advertisement = clients[client_id].advertise_keys(round_id, nonce)
        _record(transcript, advertisement.model_dump_json())
        coordinator.submit_key_advertisement(advertisement)
    pairing_message = _close(coordinator.close_advertise_keys, clock, timeout)

    for client_id in sorted(clients):
        bundle = clients[client_id].build_share_bundle(pairing_message)
        if client_id in silent_shares:
            continue
        _record(transcript, bundle.model_dump_json())
        coordinator.submit_share_bundle(bundle)
    masking_message, inbox = _close(coordinator.close_share_keys, clock, timeout)

    for client_id in sorted(inbox):
        clients[client_id].receive_shares(inbox[client_id])
        clients[client_id].accept_masking_set(masking_message)

    for client_id in sorted(inbox):
        if client_id in silent_inputs:
            continue
        message = clients[client_id].masked_input(
            updates[client_id], coordinator.scheme, clip=clip_in_client
        )
        _record(transcript, message.model_dump_json())
        if transcript is not None:
            transcript.masked_vectors[client_id] = np.frombuffer(
                message.body.vector, dtype="<u4"
            ).astype(np.float64)
        coordinator.submit_masked_input(message)
    survivor_message = _close(coordinator.close_masked_input, clock, timeout)

    for client_id in survivor_message.body.survivors:
        if client_id in silent_unmask:
            continue
        shares = clients[client_id].unmask(survivor_message)
        _record(transcript, shares.model_dump_json())
        coordinator.submit_unmask_shares(shares)
    return _close(coordinator.finalize, clock, timeout)


def _record(transcript: TranscriptRecord | None, payload: str) -> None:
    if transcript is not None:
        transcript.record(payload)


def _close[T](action: Callable[[], T], clock: SimulationClock, timeout: float) -> T:
    """Close a stage, waiting out the stage timeout if not everyone answered."""
    try:
        return action()
    except ProtocolStateError:
        clock.skip_timeout(timeout)
        return action()
