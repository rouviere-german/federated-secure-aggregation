"""The coordinator over HTTP: a full round through the ASGI app, plus what crosses the wire.

The whole round runs through Starlette's ``TestClient`` against the real FastAPI application,
so every request inspected here is the actual protocol payload, routed by the real router and
checked by the real auth and body-size middleware. No socket is opened, which keeps the suite
offline.

Bodies are captured by :class:`RecordingASGI`, which wraps the app and copies each request body
as the server receives it -- the server's side of the wire, after any client-side encoding. That
is the evidence for the "only masked vectors cross the wire" claim, so it must be the real
bytes and not a re-serialization of what the test meant to send.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastapi.testclient import TestClient
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from federated_secure_aggregation.core.config import Settings
from federated_secure_aggregation.core.security import hash_api_key
from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import RoundConfig, SecAggCoordinator
from federated_secure_aggregation.secagg.masking import plaintext_sum
from federated_secure_aggregation.secagg.session import SimulationClock
from federated_secure_aggregation.service.api import CoordinatorService, create_coordinator_app
from federated_secure_aggregation.service.client_http import HttpProtocolError, SiloDriver
from federated_secure_aggregation.service.runtime import (
    RosterError,
    build_service_from_env,
    load_identity,
    load_roster,
)

pytestmark = pytest.mark.integration

API_KEY = "coordinator-test-key-" + "y" * 24
AUTH = {"X-API-Key": API_KEY}
# Raw protocol posts send a pre-serialized body, so they must declare the media type the
# SiloDriver sends; without it FastAPI will not parse the body at all.
JSON_AUTH = {**AUTH, "Content-Type": "application/json"}
DIMENSION = 24
CLIENTS = 5
THRESHOLD = 3


class RecordingASGI:
    """Wraps an ASGI app and keeps every HTTP request body exactly as it arrives."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app
        self.bodies: list[bytes] = []

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":  # lifespan and websockets pass straight through
            await self._app(scope, receive, send)
            return

        chunks: list[bytes] = []

        async def recording_receive() -> Message:
            message = await receive()
            if message["type"] == "http.request":
                chunks.append(bytes(message.get("body", b"")))
                if not message.get("more_body", False):
                    body = b"".join(chunks)
                    chunks.clear()
                    if body:
                        self.bodies.append(body)
            return message

        await self._app(scope, recording_receive, send)


class Harness:
    def __init__(self) -> None:
        ids = [f"silo_{index:02d}" for index in range(CLIENTS)]
        self.identities = {name: IdentityKeyPair.generate() for name in ids}
        public = {name: identity.public_bytes for name, identity in self.identities.items()}
        self.coordinator_identity = IdentityKeyPair.generate()
        self.clock = SimulationClock()
        self.coordinator = SecAggCoordinator(
            self.coordinator_identity,
            public,
            RoundConfig(
                dimension=DIMENSION,
                threshold=THRESHOLD,
                clip_norm=1.0,
                max_clients=CLIENTS,
                stage_timeout_seconds=2.0,
            ),
            clock=self.clock,
        )
        self.service = CoordinatorService(self.coordinator)
        settings = Settings(
            _env_file=None,
            api_key_sha256=[hash_api_key(API_KEY)],
            environment="test",
            # A single round is a burst by design: every silo posts once per stage, and all
            # silos share one source address in-process. The limiter itself is exercised in
            # tests/integration/test_app.py; here it must not stand in for the protocol.
            rate_limit_per_second=100_000.0,
            rate_limit_burst=100_000,
        )
        self.app = create_coordinator_app(self.service, settings)
        self.transport = RecordingASGI(self.app)
        self.http = TestClient(self.transport, base_url="http://coordinator")
        self.clients = {
            name: SecAggClient(
                name,
                self.identities[name],
                peer_identity_keys=public,
                coordinator_identity_key=self.coordinator_identity.public_bytes,
            )
            for name in ids
        }
        self.drivers = {
            name: SiloDriver(client, self.http, api_key=API_KEY)
            for name, client in self.clients.items()
        }

    @property
    def ids(self) -> list[str]:
        return sorted(self.clients)

    def close(self) -> None:
        self.http.close()

    def advance(self) -> dict[str, Any]:
        response = self.http.post("/v1/rounds/advance", headers=AUTH)
        assert response.status_code == 200, response.text
        payload: dict[str, Any] = response.json()
        return payload

    def run_round(
        self, round_id: int, *, drop: tuple[str, ...] = ()
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        opened = self.http.post("/v1/rounds", json={"round_id": round_id}, headers=AUTH)
        assert opened.status_code == 200, opened.text
        nonce = base64.b64decode(opened.json()["round_nonce"])
        rng = np.random.default_rng(round_id)
        updates = {name: rng.normal(size=DIMENSION) * 0.05 for name in self.ids}

        for name in self.ids:
            self.drivers[name].advertise_keys(round_id, nonce)
        self.advance()
        for name in self.ids:
            self.drivers[name].send_shares()
        self.advance()
        for name in self.ids:
            if name in drop:
                continue
            self.drivers[name].collect_shares_and_mask(updates[name], self.coordinator.scheme)
        if drop:
            self.clock.skip_timeout(self.coordinator.config.stage_timeout_seconds + 1)
        self.advance()
        for name in self.ids:
            if name in drop:
                continue
            self.drivers[name].reveal_shares()
        return self.advance(), updates


@pytest.fixture
def harness() -> Iterator[Harness]:
    instance = Harness()
    try:
        yield instance
    finally:
        instance.close()


def test_a_full_round_over_http_reconstructs_the_plaintext_sum(harness: Harness) -> None:
    outcome, updates = harness.run_round(0)
    assert outcome["stage"] == "COMPLETED"
    assert outcome["accepted"] is True
    assert sorted(outcome["survivors"]) == harness.ids
    assert outcome["dropouts"] == []
    assert len(outcome["digest"]) == 64

    result = harness.service.last_result
    assert result is not None
    assert result.aggregate is not None
    expected = harness.coordinator.scheme.decode(
        plaintext_sum(harness.coordinator.scheme.encode(updates[name]) for name in harness.ids),
        count=CLIENTS,
    )
    assert np.array_equal(result.aggregate, expected)


def test_dropouts_are_recovered_over_http(harness: Harness) -> None:
    outcome, updates = harness.run_round(0, drop=("silo_00", "silo_01"))
    assert outcome["stage"] == "COMPLETED"
    assert sorted(outcome["dropouts"]) == ["silo_00", "silo_01"]
    result = harness.service.last_result
    assert result is not None
    assert result.aggregate is not None
    survivors = [name for name in harness.ids if name not in {"silo_00", "silo_01"}]
    expected = harness.coordinator.scheme.decode(
        plaintext_sum(harness.coordinator.scheme.encode(updates[name]) for name in survivors),
        count=len(survivors),
    )
    assert np.array_equal(result.aggregate, expected)


def test_no_request_body_ever_carries_a_plaintext_update(harness: Harness) -> None:
    """The payload-inspection control: only masked vectors cross the wire."""
    _, updates = harness.run_round(0)
    wire = b"\n".join(harness.transport.bodies)
    assert wire
    for update in updates.values():
        encoded = harness.coordinator.scheme.encode(update).astype("<u4").tobytes()
        blob = base64.b64encode(encoded)
        assert blob not in wire
        # The plain float bytes are not there either.
        assert np.ascontiguousarray(update, dtype="<f8").tobytes() not in wire

    vectors = [
        json.loads(body).get("body", {}).get("vector")
        for body in harness.transport.bodies
        if b'"masked_input"' in body
    ]
    assert len(vectors) == CLIENTS
    masked = [np.frombuffer(base64.b64decode(item), dtype="<u4") for item in vectors if item]
    for name, vector in zip(harness.ids, masked, strict=True):
        plaintext = harness.coordinator.scheme.encode(updates[name])
        assert not np.array_equal(vector, plaintext)


def test_every_route_requires_the_api_key(harness: Harness) -> None:
    for method, path, payload in [
        ("post", "/v1/rounds", {"round_id": 0}),
        ("post", "/v1/rounds/advance", None),
        ("get", "/v1/rounds/current", None),
        ("get", "/v1/rounds/pairing-set", None),
        ("get", "/v1/rounds/masking-set", None),
        ("get", "/v1/rounds/survivor-set", None),
        ("get", "/v1/rounds/inbox/silo_00", None),
        ("post", "/v1/rounds/key-advertisements", {}),
        ("post", "/v1/rounds/share-bundles", {}),
        ("post", "/v1/rounds/masked-inputs", {}),
        ("post", "/v1/rounds/unmask-shares", {}),
    ]:
        response = harness.http.request(method.upper(), path, json=payload)
        assert response.status_code == 401, (method, path, response.status_code)
        # Auth runs before validation, so an unauthenticated caller learns nothing about
        # whether its body would have been accepted.
        assert response.json()["error"]["code"] == "unauthorized"


def test_protocol_failures_map_to_status_codes_without_leaking(harness: Harness) -> None:
    early = harness.http.get("/v1/rounds/pairing-set", headers=AUTH)
    assert early.status_code == 409
    assert early.json()["error"]["code"] == "not_available"

    opened = harness.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    nonce = base64.b64decode(opened.json()["round_nonce"])
    advertisement = harness.clients["silo_00"].advertise_keys(0, nonce)
    first = harness.http.post(
        "/v1/rounds/key-advertisements", content=advertisement.model_dump_json(), headers=JSON_AUTH
    )
    assert first.status_code == 200
    duplicate = harness.http.post(
        "/v1/rounds/key-advertisements", content=advertisement.model_dump_json(), headers=JSON_AUTH
    )
    assert duplicate.status_code == 409
    assert duplicate.json()["error"]["code"] == "wrong_stage"

    wrong_stage = harness.http.post(
        "/v1/rounds/masked-inputs", content=advertisement.model_dump_json(), headers=JSON_AUTH
    )
    assert wrong_stage.status_code in {409, 422}

    missing_inbox = harness.http.get("/v1/rounds/inbox/silo_00", headers=AUTH)
    assert missing_inbox.status_code == 404


def test_round_status_and_repeated_advance(harness: Harness) -> None:
    harness.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    status = harness.http.get("/v1/rounds/current", headers=AUTH)
    assert status.status_code == 200
    assert status.json()["stage"] == "ADVERTISE_KEYS"
    assert status.json()["registered_clients"] == harness.ids

    # Nobody advertised a key, so each advance closes a stage below quorum and the round
    # fails. Advancing past the terminal stage is a 409, not a crash or a silent no-op.
    seen: list[int] = []
    for _ in range(6):
        seen.append(harness.http.post("/v1/rounds/advance", headers=AUTH).status_code)
    assert 409 in seen
    final = harness.http.post("/v1/rounds/advance", headers=AUTH)
    assert final.status_code == 409
    assert final.json()["error"]["code"] in {"not_available", "wrong_stage"}


def test_metrics_endpoint_requires_auth_and_reports_rounds(harness: Harness) -> None:
    harness.run_round(0)
    assert harness.http.get("/metrics").status_code == 401
    metrics = harness.http.get("/metrics", headers=AUTH)
    assert metrics.status_code == 200
    # Round metrics and HTTP metrics must come from the same registry, or the protocol
    # metrics the runbook relies on are collected but never scraped.
    assert "fedsecagg_rounds_total 1.0" in metrics.text
    for name in (
        "fedsecagg_round_duration_seconds",
        "fedsecagg_participants",
        "fedsecagg_dropouts_total",
        "fedsecagg_rounds_rejected_total",
        "fedsecagg_epsilon_spent",
    ):
        assert name in metrics.text, name
    assert "app_http_requests_total" in metrics.text


def test_health_and_readiness(harness: Harness) -> None:
    assert harness.http.get("/healthz").json() == {"status": "ok"}
    ready = harness.http.get("/readyz")
    assert ready.status_code == 200
    assert ready.json()["checks"]["round_machine"] == "ok"


def test_the_driver_reports_http_failures(harness: Harness) -> None:
    driver = SiloDriver(harness.clients["silo_00"], harness.http, api_key="wrong-key")
    with pytest.raises(HttpProtocolError) as info:
        driver.advertise_keys(0, bytes(16))
    assert info.value.status_code == 401
    assert info.value.action == "advertise-keys"


def test_build_service_from_env_reads_mounted_key_material(tmp_path: Path) -> None:
    identity = IdentityKeyPair.generate()
    roster = {
        f"silo_{index:02d}": base64.b64encode(IdentityKeyPair.generate().public_bytes).decode(
            "ascii"
        )
        for index in range(4)
    }
    identity_path = tmp_path / "coordinator_identity.json"
    roster_path = tmp_path / "roster.json"
    identity_path.write_text(
        json.dumps({"private_key_b64": base64.b64encode(identity.private_bytes).decode("ascii")}),
        encoding="utf-8",
    )
    roster_path.write_text(json.dumps(roster), encoding="utf-8")

    settings = Settings(
        _env_file=None,
        api_key_sha256=[hash_api_key(API_KEY)],
        environment="test",
        coordinator_identity_path=identity_path,
        roster_path=roster_path,
        model_dimension=DIMENSION,
        secagg_threshold=3,
    )
    app = build_service_from_env(settings)
    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 200
    assert load_identity(identity_path).public_bytes == identity.public_bytes
    assert set(load_roster(roster_path)) == set(roster)


def test_roster_and_identity_files_are_validated(tmp_path: Path) -> None:
    missing = tmp_path / "nope.json"
    with pytest.raises(RosterError, match="does not exist"):
        load_roster(missing)
    with pytest.raises(RosterError, match="does not exist"):
        load_identity(missing)

    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{", encoding="utf-8")
    with pytest.raises(RosterError, match="not valid JSON"):
        load_roster(bad_json)
    with pytest.raises(RosterError, match="not valid JSON"):
        load_identity(bad_json)

    empty = tmp_path / "empty.json"
    empty.write_text("{}", encoding="utf-8")
    with pytest.raises(RosterError, match="non-empty object"):
        load_roster(empty)

    bad_id = tmp_path / "bad_id.json"
    bad_id.write_text(json.dumps({"Silo A": "AAAA"}), encoding="utf-8")
    with pytest.raises(RosterError, match="invalid client id"):
        load_roster(bad_id)

    short_key = tmp_path / "short.json"
    short_key.write_text(
        json.dumps({"silo_00": base64.b64encode(b"short").decode("ascii")}), encoding="utf-8"
    )
    with pytest.raises(RosterError, match="must decode to 32 bytes"):
        load_roster(short_key)

    not_base64 = tmp_path / "not_b64.json"
    not_base64.write_text(json.dumps({"silo_00": "!!!"}), encoding="utf-8")
    with pytest.raises(RosterError, match="not valid base64"):
        load_roster(not_base64)

    duplicate = tmp_path / "dupe.json"
    shared = base64.b64encode(IdentityKeyPair.generate().public_bytes).decode("ascii")
    duplicate.write_text(json.dumps({"silo_00": shared, "silo_01": shared}), encoding="utf-8")
    with pytest.raises(RosterError, match="share an identity key"):
        load_roster(duplicate)

    not_object = tmp_path / "list.json"
    not_object.write_text(json.dumps([1, 2]), encoding="utf-8")
    with pytest.raises(RosterError, match="must be a JSON object"):
        load_identity(not_object)
