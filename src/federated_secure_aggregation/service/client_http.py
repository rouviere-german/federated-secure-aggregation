"""Driving a silo through one round over HTTP.

This is the piece a real silo container runs. It owns an
:class:`~federated_secure_aggregation.secagg.client.SecAggClient` and does nothing but move its
messages over ``httpx``: no protocol decision is taken here, so a bug in the transport cannot
weaken the protocol, and the same client state machine is exercised by the in-process tests.

The driver is deliberately paranoid about the coordinator's replies: announcements are parsed
into the same signed types the client expects and are verified against the coordinator's
identity key inside the client, so a coordinator that tampers with a pairing set is caught
there and not here.

``httpx.Client`` is injected rather than constructed, which is how the integration tests point
it at the ASGI app without opening a socket - the whole suite stays offline.

Transient answers are retried here rather than being allowed to cost a round. The coordinator's
own rate limiter is the failure a healthy federation meets first: six pollers against one
service produce 429s under load, and the limiter answers them with a ``Retry-After``. Treating
that as fatal turns a silo into a dropout, or kills the operator container outright, for a
condition the server explicitly said was temporary. 4xx answers other than 408 and 429 are
*decisions* (a bad key, a wrong stage, a forbidden role) and are never retried.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.core.logging import get_logger
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.messages import (
    MaskingSet,
    PairingSet,
    RoundStage,
    SignedMessage,
    SurvivorSet,
)
from federated_secure_aggregation.secagg.quantize import AggregationScheme

__all__ = [
    "RETRYABLE_STATUSES",
    "HttpProtocolError",
    "RetryPolicy",
    "SiloDriver",
    "request_with_retry",
]

FloatArray = npt.NDArray[np.float64]
_log = get_logger("federated_secure_aggregation.transport")

# 408 and 429 are the two 4xx answers that mean "later", not "no".
RETRYABLE_STATUSES = frozenset({408, 429, 500, 502, 503, 504})

# Stages in protocol order, so a driver can tell "not yet" from "already missed".
_STAGE_ORDER = {
    RoundStage.ADVERTISE_KEYS: 0,
    RoundStage.SHARE_KEYS: 1,
    RoundStage.MASKED_INPUT: 2,
    RoundStage.UNMASK: 3,
    RoundStage.AGGREGATE: 4,
    RoundStage.COMPLETED: 5,
    RoundStage.REJECTED: 5,
    RoundStage.FAILED: 5,
}
_TERMINAL_STAGES = frozenset({RoundStage.COMPLETED, RoundStage.REJECTED, RoundStage.FAILED})


class HttpProtocolError(RuntimeError):
    """The coordinator answered with an error status or an unusable body."""

    def __init__(self, action: str, status_code: int, code: str | None = None) -> None:
        super().__init__(
            f"{action} failed with status {status_code}" + (f" ({code})" if code else "")
        )
        self.action = action
        self.status_code = status_code
        self.code = code


@dataclass(frozen=True, slots=True)
class RetryPolicy:
    """How hard to try again after a transient answer.

    Bounded on purpose: a silo that retries for ever would hold its round open past every
    stage timeout, which is worse than becoming a dropout the protocol already recovers from.
    """

    attempts: int = 4
    backoff_seconds: float = 0.1
    max_backoff_seconds: float = 5.0

    def __post_init__(self) -> None:
        if self.attempts < 1:
            raise ValueError("attempts must be >= 1")
        if self.backoff_seconds <= 0:
            raise ValueError("backoff_seconds must be > 0")
        if self.max_backoff_seconds < self.backoff_seconds:
            raise ValueError("max_backoff_seconds must be >= backoff_seconds")

    def delay(self, attempt: int, retry_after: str | None) -> float:
        """Honour ``Retry-After`` when it is a plain number of seconds, else back off."""
        if retry_after is not None:
            text = retry_after.strip()
            # Only the delta-seconds form; the HTTP-date form needs a clock we do not want here.
            if text.isascii() and text.isdigit():
                return min(float(text), self.max_backoff_seconds)
        backoff: float = self.backoff_seconds * float(2**attempt)
        return min(backoff, self.max_backoff_seconds)


def request_with_retry(
    http: httpx.Client,
    method: str,
    path: str,
    *,
    headers: Mapping[str, str],
    content: str | None = None,
    json: Any = None,
    retry: RetryPolicy,
    sleep: Callable[[float], None] = time.sleep,
) -> httpx.Response:
    """Send a request, retrying transient answers. Returns the last response either way.

    The caller decides what a final failure means, which differs by role: a silo becomes a
    dropout, the operator retries the stage.
    """
    response = http.request(method, path, headers=dict(headers), content=content, json=json)
    for attempt in range(retry.attempts - 1):
        if response.status_code not in RETRYABLE_STATUSES:
            return response
        delay = retry.delay(attempt, response.headers.get("Retry-After"))
        _log.info("transport_retry", path=path, status=response.status_code, delay=delay)
        sleep(delay)
        response = http.request(method, path, headers=dict(headers), content=content, json=json)
    return response


class SiloDriver:
    """One silo's HTTP conversation with the coordinator for one round."""

    def __init__(
        self,
        client: SecAggClient,
        http: httpx.Client,
        *,
        api_key: str,
        retry: RetryPolicy | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.client = client
        self._http = http
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
        self._retry = retry if retry is not None else RetryPolicy()
        self._sleep = sleep

    # --- transport ---------------------------------------------------------------------

    def _post(self, path: str, payload: str, action: str) -> dict[str, Any]:
        return self._body(self._send("POST", path, content=payload), action)

    def _get(self, path: str, action: str) -> dict[str, Any]:
        return self._body(self._send("GET", path), action)

    def _send(self, method: str, path: str, *, content: str | None = None) -> httpx.Response:
        return request_with_retry(
            self._http,
            method,
            path,
            headers=self._headers,
            content=content,
            retry=self._retry,
            sleep=self._sleep,
        )

    @staticmethod
    def _body(response: httpx.Response, action: str) -> dict[str, Any]:
        if response.status_code >= 400:
            code: str | None = None
            try:
                payload = response.json()
                code = str(payload.get("error", {}).get("code"))
            except ValueError:  # pragma: no cover - non-JSON error bodies
                code = None
            raise HttpProtocolError(action, response.status_code, code)
        data = response.json()
        if not isinstance(data, dict):  # pragma: no cover - the API always returns objects
            raise HttpProtocolError(action, response.status_code, "unexpected_body")
        return data

    # --- round discovery ---------------------------------------------------------------

    def current_round(self) -> tuple[int, RoundStage, bytes] | None:
        """The open round's id, stage and nonce, or ``None`` when no round is open.

        A silo polls this instead of being told, so a container that restarts mid-run rejoins
        on its own. "No round open" is an ordinary answer, not an error.
        """
        try:
            payload = self._get("/v1/rounds/current", "round-status")
        except HttpProtocolError as exc:
            if exc.status_code == 409:
                return None
            raise
        stage = RoundStage(str(payload["stage"]))
        if stage in _TERMINAL_STAGES:
            return None
        nonce = base64.b64decode(str(payload["round_nonce"]), validate=True)
        return int(payload["round_id"]), stage, nonce

    def scheme(self) -> AggregationScheme:
        """Rebuild the coordinator's quantization scheme from the parameters it publishes.

        Every silo must encode with identical parameters or the ring sum is meaningless, so
        they are taken from the coordinator rather than configured per silo.
        """
        payload = self._get("/v1/rounds/current", "round-status")
        return AggregationScheme(
            dimension=int(payload["dimension"]),
            clip_norm=float(payload["clip_norm"]),
            scale=float(payload["scale"]),
            max_clients=len(list(payload["registered_clients"])),
        )

    def wait_for_stage(
        self,
        stage: RoundStage,
        *,
        poll_seconds: float = 0.25,
        attempts: int = 400,
        sleep: Callable[[float], None] | None = None,
    ) -> bool:
        """Block until the coordinator reaches ``stage``, or the round ends. ``False`` on end.

        Bounded by ``attempts`` so a stalled coordinator cannot wedge a silo forever; the silo
        then simply misses the stage and becomes a dropout, which the protocol recovers from.
        """
        wait = sleep if sleep is not None else self._sleep
        for _ in range(attempts):
            payload = self._get("/v1/rounds/current", "round-status")
            current = RoundStage(str(payload["stage"]))
            if current is stage:
                return True
            if current in _TERMINAL_STAGES or _STAGE_ORDER[current] > _STAGE_ORDER[stage]:
                return False
            wait(poll_seconds)
        return False

    # --- stages ------------------------------------------------------------------------

    def advertise_keys(self, round_id: int, round_nonce: bytes) -> None:
        message = self.client.advertise_keys(round_id, round_nonce)
        self._post("/v1/rounds/key-advertisements", message.model_dump_json(), "advertise-keys")

    def send_shares(self) -> None:
        pairing = SignedMessage[PairingSet].model_validate(
            self._get("/v1/rounds/pairing-set", "pairing-set")
        )
        bundle = self.client.build_share_bundle(pairing)
        self._post("/v1/rounds/share-bundles", bundle.model_dump_json(), "share-bundle")

    def collect_shares_and_mask(self, update: FloatArray, scheme: AggregationScheme) -> None:
        inbox = self._get(f"/v1/rounds/inbox/{self.client.client_id}", "inbox")
        ciphertexts = {
            str(dealer): base64.b64decode(str(blob), validate=True)
            for dealer, blob in dict(inbox["ciphertexts"]).items()
        }
        self.client.receive_shares(ciphertexts)
        masking = SignedMessage[MaskingSet].model_validate(
            self._get("/v1/rounds/masking-set", "masking-set")
        )
        self.client.accept_masking_set(masking)
        message = self.client.masked_input(update, scheme)
        self._post("/v1/rounds/masked-inputs", message.model_dump_json(), "masked-input")

    def reveal_shares(self) -> None:
        survivors = SignedMessage[SurvivorSet].model_validate(
            self._get("/v1/rounds/survivor-set", "survivor-set")
        )
        message = self.client.unmask(survivors)
        self._post("/v1/rounds/unmask-shares", message.model_dump_json(), "unmask-shares")
