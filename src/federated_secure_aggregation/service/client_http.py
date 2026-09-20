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
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable
from typing import Any

import httpx
import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.messages import (
    MaskingSet,
    PairingSet,
    RoundStage,
    SignedMessage,
    SurvivorSet,
)
from federated_secure_aggregation.secagg.quantize import AggregationScheme

__all__ = ["HttpProtocolError", "SiloDriver"]

FloatArray = npt.NDArray[np.float64]

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


class SiloDriver:
    """One silo's HTTP conversation with the coordinator for one round."""

    def __init__(self, client: SecAggClient, http: httpx.Client, *, api_key: str) -> None:
        self.client = client
        self._http = http
        self._headers = {"X-API-Key": api_key, "Content-Type": "application/json"}

    # --- transport ---------------------------------------------------------------------

    def _post(self, path: str, payload: str, action: str) -> dict[str, Any]:
        response = self._http.post(path, content=payload, headers=self._headers)
        return self._body(response, action)

    def _get(self, path: str, action: str) -> dict[str, Any]:
        response = self._http.get(path, headers=self._headers)
        return self._body(response, action)

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
        sleep: Callable[[float], None] = time.sleep,
    ) -> bool:
        """Block until the coordinator reaches ``stage``, or the round ends. ``False`` on end.

        Bounded by ``attempts`` so a stalled coordinator cannot wedge a silo forever; the silo
        then simply misses the stage and becomes a dropout, which the protocol recovers from.
        """
        for _ in range(attempts):
            payload = self._get("/v1/rounds/current", "round-status")
            current = RoundStage(str(payload["stage"]))
            if current is stage:
                return True
            if current in _TERMINAL_STAGES or _STAGE_ORDER[current] > _STAGE_ORDER[stage]:
                return False
            sleep(poll_seconds)
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
