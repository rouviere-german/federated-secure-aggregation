"""What a silo container runs, and what drives the rounds in the compose stack.

Two loops live here, both thin:

``run_silo``
    A silo's loop. It builds *its own* partition locally from the shared seed and its id -- no
    shared volume, no dataset download, no row ever leaving the container -- and then follows
    the coordinator's published stage through
    :class:`~federated_secure_aggregation.service.client_http.SiloDriver`. Every protocol
    decision stays in ``SecAggClient``; this loop only decides *when* to act, by polling.

``run_operator``
    The round driver. In compose it is a separate container because it is a separate role: it
    holds an operator API key, opens rounds and advances stages, and never sees key material or
    silo data. A silo cannot open a round, which is what stops a compromised silo from choosing
    the pairing set.

Both loops are restartable by design: they read the coordinator's current stage rather than
keeping their own idea of it. A silo that dies mid-round and comes back simply missed a stage,
which makes it a dropout -- exactly the case the protocol exists to survive. Nothing here
retries a stage it already answered, because the coordinator accepts one message per client per
stage and a retry would be refused anyway.

Neither loop is exercised by the compose stack in CI's default path; both are covered in
process against the ASGI app in ``tests/integration/test_silo_runner.py``, which is what keeps
the container entrypoints from rotting.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.core.logging import get_logger
from federated_secure_aggregation.fl.experiment import build_experiment, silo_ids
from federated_secure_aggregation.fl.local import LocalTrainingConfig, train_local
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.errors import SecAggError
from federated_secure_aggregation.secagg.messages import ProtocolMessageError, RoundStage
from federated_secure_aggregation.secagg.quantize import clip_l2
from federated_secure_aggregation.service.client_http import HttpProtocolError, SiloDriver

__all__ = ["SiloConfig", "run_operator", "run_silo"]

FloatArray = npt.NDArray[np.float64]
_log = get_logger("federated_secure_aggregation.silo")


@dataclass(frozen=True, slots=True)
class SiloConfig:
    """Everything a silo container needs that is not key material."""

    silo_id: str
    n_silos: int = 5
    n_rows: int = 6_000
    alpha: float = 1.0
    seed: int = 20260920
    rounds: int = 20
    epochs: int = 2
    batch_size: int = 64
    learning_rate: float = 0.5
    l2: float = 1e-4
    poll_seconds: float = 0.25
    max_polls: int = 2_000

    def __post_init__(self) -> None:
        if self.silo_id not in silo_ids(self.n_silos):
            raise ValueError(f"{self.silo_id} is not one of {silo_ids(self.n_silos)}")
        if self.rounds < 1:
            raise ValueError("rounds must be >= 1")
        if self.poll_seconds <= 0:
            raise ValueError("poll_seconds must be > 0")
        if self.max_polls < 1:
            raise ValueError("max_polls must be >= 1")


def run_silo(
    config: SiloConfig,
    client: SecAggClient,
    http: httpx.Client,
    *,
    api_key: str,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Join rounds until ``config.rounds`` distinct ones have been seen.

    Returns a summary of what this silo did. Raw rows never leave this function: only the masked
    vector built inside ``SecAggClient`` is handed to the driver.
    """
    experiment = build_experiment(
        n_rows=config.n_rows, n_silos=config.n_silos, alpha=config.alpha, seed=config.seed
    )
    features, labels = experiment.silo_data()[config.silo_id]
    model = experiment.logistic_model()
    global_params = model.initial_params()
    driver = SiloDriver(client, http, api_key=api_key)
    training = LocalTrainingConfig(
        epochs=config.epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        l2=config.l2,
    )

    seen: set[int] = set()
    completed = 0
    for _ in range(config.max_polls):
        if len(seen) >= config.rounds:
            break
        opened = _poll(driver, sleep, config.poll_seconds)
        if opened is None:
            continue
        round_id, stage, nonce = opened
        if round_id in seen or stage is not RoundStage.ADVERTISE_KEYS:
            # Either already played, or the round moved past the only stage a silo can join at.
            sleep(config.poll_seconds)
            continue
        seen.add(round_id)
        try:
            _play_round(
                driver,
                config,
                model,
                training,
                global_params=global_params,
                features=features,
                labels=labels,
                round_id=round_id,
                nonce=nonce,
            )
        except (HttpProtocolError, SecAggError, ProtocolMessageError) as exc:
            # Missing a stage makes this silo a dropout for the round, which the protocol
            # recovers from. Losing a round is never a reason to lose the process.
            _log.warning(
                "silo_round_failed",
                silo=config.silo_id,
                round_id=round_id,
                error_type=type(exc).__name__,
            )
        else:
            completed += 1
            _log.info("silo_round_done", silo=config.silo_id, round_id=round_id)
    return {
        "silo_id": config.silo_id,
        "rounds_seen": len(seen),
        "rounds_completed": completed,
        "rows": int(features.shape[0]),
    }


def _poll(
    driver: SiloDriver, sleep: Callable[[float], None], poll_seconds: float
) -> tuple[int, RoundStage, bytes] | None:
    """Ask the coordinator what round is open. ``None`` means "nothing to do yet"."""
    try:
        opened = driver.current_round()
    except HttpProtocolError:
        opened = None
    if opened is None:
        sleep(poll_seconds)
    return opened


def _play_round(
    driver: SiloDriver,
    config: SiloConfig,
    model: Any,
    training: LocalTrainingConfig,
    *,
    global_params: FloatArray,
    features: FloatArray,
    labels: FloatArray,
    round_id: int,
    nonce: bytes,
) -> None:
    """Walk one round's stages, waiting for the coordinator to open each one."""
    scheme = driver.scheme()
    # Per-round seed: reproducible across restarts, different every round.
    trained = train_local(
        model, global_params, features, labels, training, seed=config.seed + round_id
    )
    update, _ = clip_l2(trained - global_params, scheme.clip_norm)

    driver.advertise_keys(round_id, nonce)
    if not driver.wait_for_stage(RoundStage.SHARE_KEYS, poll_seconds=config.poll_seconds):
        raise HttpProtocolError("share-keys", 409, "stage_missed")
    driver.send_shares()
    if not driver.wait_for_stage(RoundStage.MASKED_INPUT, poll_seconds=config.poll_seconds):
        raise HttpProtocolError("masked-input", 409, "stage_missed")
    driver.collect_shares_and_mask(update, scheme)
    if not driver.wait_for_stage(RoundStage.UNMASK, poll_seconds=config.poll_seconds):
        raise HttpProtocolError("unmask", 409, "stage_missed")
    driver.reveal_shares()


def run_operator(
    http: httpx.Client,
    *,
    api_key: str,
    rounds: int,
    expected_clients: int = 0,
    poll_seconds: float = 0.25,
    join_polls: int = 1_000,
    sleep: Callable[[float], None] = time.sleep,
) -> list[dict[str, Any]]:
    """Open ``rounds`` rounds and ask the coordinator to close each stage. One outcome per round.

    The operator does not decide when a stage closes -- it only *asks*. The coordinator refuses
    (409) until every registered silo has answered or the stage has timed out, which is what
    stops an operator from closing a stage early to exclude silos it dislikes. Excluding a silo
    therefore costs a full stage timeout and is visible in the round's dropout list, instead of
    being a free, silent choice.

    So the loop here is "retry the advance until it is allowed". ``expected_clients`` is accepted
    for symmetry with the silo config and is deliberately unused: the operator is not the one who
    counts answers.
    """
    if rounds < 1:
        raise ValueError("rounds must be >= 1")
    del expected_clients
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    outcomes: list[dict[str, Any]] = []

    for round_id in range(rounds):
        opened = http.post("/v1/rounds", json={"round_id": round_id}, headers=headers)
        if opened.status_code >= 400:
            raise HttpProtocolError("start-round", opened.status_code)
        outcome = _drive_round(http, headers, poll_seconds, join_polls, sleep)
        outcomes.append(outcome or {"round_id": round_id, "stage": "FAILED", "accepted": False})
        _log.info("operator_round_done", round_id=round_id)
    return outcomes


def _drive_round(
    http: httpx.Client,
    headers: dict[str, str],
    poll_seconds: float,
    join_polls: int,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Advance a round through its four stages, waiting whenever the coordinator says "not yet"."""
    for _ in range(4):
        payload = _advance_when_allowed(http, headers, poll_seconds, join_polls, sleep)
        if payload is None:
            return None
        if "accepted" in payload:
            return payload
    return None


def _advance_when_allowed(
    http: httpx.Client,
    headers: dict[str, str],
    poll_seconds: float,
    join_polls: int,
    sleep: Callable[[float], None],
) -> dict[str, Any] | None:
    """Retry ``advance`` until the coordinator permits the stage to close, or give up."""
    for _ in range(join_polls):
        response = http.post("/v1/rounds/advance", headers=headers)
        if response.status_code < 400:
            body = response.json()
            return dict(body) if isinstance(body, dict) else {}
        if response.status_code != 409:
            return None
        sleep(poll_seconds)
    return None


def main_silo(
    config: SiloConfig, client: SecAggClient, base_url: str, *, api_key: str
) -> dict[str, Any]:  # pragma: no cover - needs a socket
    """Container entrypoint: open a real HTTP client and run the silo loop."""
    with httpx.Client(base_url=base_url, timeout=30.0) as http:
        return run_silo(config, client, http, api_key=api_key)


def main_operator(
    base_url: str, *, api_key: str, rounds: int, expected_clients: int
) -> list[dict[str, Any]]:  # pragma: no cover - needs a socket
    """Container entrypoint: open a real HTTP client and drive the rounds."""
    with httpx.Client(base_url=base_url, timeout=30.0) as http:
        return run_operator(http, api_key=api_key, rounds=rounds, expected_clients=expected_clients)
