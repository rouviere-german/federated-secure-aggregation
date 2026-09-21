"""The container entrypoints: five silo loops and an operator driving real rounds.

This is the compose stack without Docker. Each silo runs its own ``run_silo`` loop in a thread
against the real ASGI app, exactly as a container would against a real socket, while the main
thread plays the operator. Nothing is stubbed below the HTTP boundary, so the polling, the stage
waits and the dropout path are the ones that ship.

Threads are the point, not an accident: the protocol only works because silos act concurrently
inside a stage, and a sequential test would deadlock at the first ``wait_for_stage``. The suite
still opens no socket -- Starlette's ``TestClient`` runs the app in-process.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Any, cast

import httpx
import pytest
from fastapi.testclient import TestClient

from federated_secure_aggregation.core.config import Settings
from federated_secure_aggregation.core.security import hash_api_key
from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import RoundConfig, SecAggCoordinator
from federated_secure_aggregation.secagg.messages import RoundStage
from federated_secure_aggregation.service.api import CoordinatorService, create_coordinator_app
from federated_secure_aggregation.service.client_http import (
    HttpProtocolError,
    RetryPolicy,
    SiloDriver,
)
from federated_secure_aggregation.service.silo_runner import (
    SiloConfig,
    run_operator,
    run_silo,
)

pytestmark = pytest.mark.integration

# One key per identity, as the compose stack issues them.
OPERATOR_KEY = "silo-runner-test-operator-" + "z" * 24
AUTH = {"X-API-Key": OPERATOR_KEY}
SILOS = 5
THRESHOLD = 3
ROUNDS = 2
# The model has one weight per feature plus a bias; the coordinator must agree on the width.
DIMENSION = 13
# What docker-compose.yml sets on the coordinator: sized per identity, for six pollers at the
# shipped 0.25 s poll interval.
SHIPPED_RATE = 50.0
SHIPPED_BURST = 200


def silo_key(silo_id: str) -> str:
    return f"silo-runner-test-{silo_id}-" + "z" * 24


class Stack:
    """A coordinator app plus five silo clients, all in this process."""

    def __init__(
        self,
        *,
        n_silos: int = SILOS,
        threshold: int = THRESHOLD,
        stage_timeout_seconds: float = 120.0,
        rate_limit_per_second: float = SHIPPED_RATE * 25,
        rate_limit_burst: int = SHIPPED_BURST * 25,
    ) -> None:
        self.ids = [f"silo_{index:02d}" for index in range(n_silos)]
        identities = {name: IdentityKeyPair.generate() for name in self.ids}
        public = {name: identity.public_bytes for name, identity in identities.items()}
        coordinator_identity = IdentityKeyPair.generate()
        self.coordinator = SecAggCoordinator(
            coordinator_identity,
            public,
            RoundConfig(
                dimension=DIMENSION,
                threshold=threshold,
                clip_norm=1.0,
                max_clients=n_silos,
                stage_timeout_seconds=stage_timeout_seconds,
            ),
        )
        self.service = CoordinatorService(self.coordinator)
        settings = Settings(
            _env_file=None,
            api_key_sha256=[hash_api_key(silo_key(name)) for name in self.ids],
            operator_api_key_sha256=[hash_api_key(OPERATOR_KEY)],
            environment="test",
            # These tests poll 25x faster than the shipped 0.25 s interval so the suite stays
            # quick, so the default here is the shipped limit scaled by the same 25x. That
            # keeps the requests-per-poll-interval headroom identical to production instead of
            # switching the limiter off. ``test_the_stack_completes_at_the_shipped_limits``
            # runs the same stack at the unscaled compose values.
            rate_limit_per_second=rate_limit_per_second,
            rate_limit_burst=rate_limit_burst,
        )
        self.app = create_coordinator_app(self.service, settings)
        self.http = TestClient(self.app, base_url="http://coordinator")
        self.clients = {
            name: SecAggClient(
                name,
                identities[name],
                peer_identity_keys=public,
                coordinator_identity_key=coordinator_identity.public_bytes,
            )
            for name in self.ids
        }

    def config(self, silo_id: str, **overrides: Any) -> SiloConfig:
        base: dict[str, Any] = {
            "silo_id": silo_id,
            "n_silos": len(self.ids),
            "n_rows": 1_200,
            "rounds": ROUNDS,
            "epochs": 1,
            "poll_seconds": 0.01,
            "max_polls": 4_000,
        }
        base.update(overrides)
        return SiloConfig(**base)

    def close(self) -> None:
        self.http.close()


@pytest.fixture
def stack() -> Any:
    built = Stack()
    yield built
    built.close()


@pytest.fixture
def short_timeout_stack() -> Any:
    """A stack whose stages time out quickly, so absent silos become dropouts in test time."""
    built = Stack(stage_timeout_seconds=1.0)
    yield built
    built.close()


def _sleep(seconds: float) -> None:
    """A real sleep that releases the GIL, so the silo threads actually get to run."""
    threading.Event().wait(seconds)


def _drive(
    stack: Stack,
    joining: list[str],
    *,
    rounds: int = ROUNDS,
    poll_seconds: float = 0.01,
) -> dict[str, Any]:
    """Start every joining silo in a thread and play the operator until the rounds are done."""
    results: dict[str, Any] = {}
    started = threading.Event()

    def silo(name: str) -> dict[str, Any]:
        started.set()
        return run_silo(
            stack.config(name, rounds=rounds, poll_seconds=poll_seconds),
            stack.clients[name],
            stack.http,
            api_key=silo_key(name),
            sleep=_sleep,
        )

    with ThreadPoolExecutor(max_workers=len(joining) + 1) as pool:
        futures = {name: pool.submit(silo, name) for name in joining}
        started.wait(timeout=10)
        outcomes = run_operator(
            stack.http,
            api_key=OPERATOR_KEY,
            rounds=rounds,
            poll_seconds=poll_seconds,
            join_polls=3_000,
            sleep=_sleep,
        )
        for name, future in futures.items():
            results[name] = future.result(timeout=120)
    results["_outcomes"] = outcomes
    return results


def test_five_silo_loops_and_an_operator_complete_real_rounds(stack: Stack) -> None:
    """The compose stack's happy path: every silo joins, every round is accepted."""
    results = _drive(stack, stack.ids)
    outcomes = results.pop("_outcomes")

    assert len(outcomes) == ROUNDS
    for outcome in outcomes:
        assert outcome["stage"] == RoundStage.COMPLETED
        assert outcome["accepted"] is True
        assert sorted(outcome["survivors"]) == stack.ids
        assert outcome["dropouts"] == []

    for name in stack.ids:
        assert results[name]["rounds_completed"] == ROUNDS
        assert results[name]["silo_id"] == name
        # Each silo trained on its own partition and nothing else.
        assert 0 < results[name]["rows"] < 1_200


def test_the_stack_completes_at_the_shipped_rate_limits() -> None:
    """The whole federation, at the exact limits docker-compose.yml configures.

    This is the test that would have caught the deployment defect: with one API key shared by
    the operator and all five silos, every process drew on one token bucket, and at the 5/s
    default each silo's first round failed with ``HttpProtocolError`` while the operator died
    on a 429 from ``POST /v1/rounds``. The fix is two-sided -- one key per identity so the
    buckets are really per identity, a limit on the coordinator sized for six pollers, and a
    transport that honours ``Retry-After`` instead of losing the round -- so the test runs the
    real loops at the real numbers.

    Polling five times faster than the shipped interval (0.05 s, not 0.25 s) keeps the test
    quick and makes it strictly harder than production: six identities at 20 req/s each is
    120 req/s in total, well past the 50/s ceiling that one shared bucket would have imposed,
    while each individual bucket stays inside it. That is exactly the difference the fix makes,
    so the test asserts *zero* throttled answers rather than only that the round completed --
    the retrying transport would otherwise paper over a collapsed bucket and pass anyway.

    It takes several rounds for that to bite, because a burst of 200 absorbs a single round's
    ~90 requests whatever the bucket topology. Five rounds is where the sustained rate decides
    it: measured here, per-identity buckets give 516 requests and 0 throttled in 3.2 s, while
    collapsing every principal onto one bucket gives 27 throttled and takes 9.2 s.
    """
    rounds = 5
    built = Stack(rate_limit_per_second=SHIPPED_RATE, rate_limit_burst=SHIPPED_BURST)
    counted = _CountThrottled(built.http)
    built.http = cast("Any", counted)
    try:
        results = _drive(built, built.ids, rounds=rounds, poll_seconds=0.05)
        outcomes = results.pop("_outcomes")
        assert len(outcomes) == rounds
        for outcome in outcomes:
            assert outcome["stage"] == RoundStage.COMPLETED
            assert outcome["accepted"] is True
            assert sorted(outcome["survivors"]) == built.ids
        for name in built.ids:
            assert results[name]["rounds_completed"] == rounds
        assert counted.throttled == 0, (
            f"{counted.throttled} of {counted.total} requests were rate limited: the six "
            "identities are sharing a token bucket instead of holding one each"
        )
    finally:
        built.close()


class _CountThrottled:
    """Passes every request through, counting the 429s the coordinator hands back."""

    def __init__(self, inner: TestClient) -> None:
        self._inner = inner
        self._lock = threading.Lock()
        self.total = 0
        self.throttled = 0

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        response = self._inner.request(method, path, **kwargs)
        with self._lock:
            self.total += 1
            if response.status_code == 429:
                self.throttled += 1
        return response

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def test_a_rate_limited_silo_retries_instead_of_dropping_out(stack: Stack) -> None:
    """One 429 must cost a wait, not the round.

    Before the fix ``SiloDriver`` raised on any status >= 400, so a single burst of rate
    limiting turned a healthy silo into a dropout for that round.
    """
    stack.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    throttled = _ThrottleOnce(stack.http)
    driver = SiloDriver(
        stack.clients["silo_00"],
        throttled.client,
        api_key=silo_key("silo_00"),
        retry=RetryPolicy(attempts=3, backoff_seconds=0.01),
        sleep=lambda _: None,
    )
    driver.advertise_keys(0, stack.coordinator.round_nonce)
    assert throttled.injected == 1
    assert stack.coordinator.stage_answers == 1


class _ThrottleOnce:
    """Wraps a client so the first request comes back 429 with a ``Retry-After``."""

    def __init__(self, inner: TestClient) -> None:
        self.injected = 0
        self._inner = inner
        self.client = cast("Any", self)

    def request(self, method: str, path: str, **kwargs: Any) -> Any:
        if self.injected == 0:
            self.injected += 1
            return httpx.Response(
                429, headers={"Retry-After": "1"}, json={"error": {"code": "rate_limited"}}
            )
        return self._inner.request(method, path, **kwargs)


def test_rounds_still_complete_when_two_silos_never_join(short_timeout_stack: Stack) -> None:
    """Two containers stay down. Survivors are above the threshold, so the round completes.

    The stage cannot close until it times out, so the missing silos cost a real timeout -- see
    :func:`test_the_operator_cannot_close_a_stage_early_to_exclude_silos` for why that matters.
    """
    stack = short_timeout_stack
    joining = stack.ids[2:]
    assert len(joining) == THRESHOLD
    results = _drive(stack, joining, rounds=1)
    outcomes = results.pop("_outcomes")

    assert len(outcomes) == 1
    outcome = outcomes[0]
    assert outcome["stage"] == RoundStage.COMPLETED
    assert outcome["accepted"] is True
    assert sorted(outcome["survivors"]) == joining
    # Absent silos never advertised a key, so they never entered the pairing set at all.
    assert outcome["dropouts"] == []
    for name in joining:
        assert results[name]["rounds_completed"] == 1


def _stage(stack: Stack) -> str:
    """The coordinator's current stage, read fresh (it changes between assertions)."""
    return str(stack.coordinator.stage)


def test_the_operator_cannot_close_a_stage_early_to_exclude_silos(stack: Stack) -> None:
    """The isolation attack, refused.

    If an operator could close ADVERTISE_KEYS as soon as it liked, it could run a round with a
    single silo of its choosing and learn that silo's update exactly. The coordinator refuses to
    close a stage until every registered silo has answered or the stage has timed out, so
    excluding a silo costs a full timeout and shows up in the round's dropout list.
    """
    stack.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    for name in stack.ids[:2]:
        SiloDriver(stack.clients[name], stack.http, api_key=silo_key(name)).advertise_keys(
            0, stack.coordinator.round_nonce
        )

    refused = stack.http.post("/v1/rounds/advance", headers=AUTH)
    assert refused.status_code == 409
    assert refused.json()["error"]["code"] == "wrong_stage"
    assert _stage(stack) == RoundStage.ADVERTISE_KEYS

    for name in stack.ids[2:]:
        SiloDriver(stack.clients[name], stack.http, api_key=silo_key(name)).advertise_keys(
            0, stack.coordinator.round_nonce
        )
    assert stack.http.post("/v1/rounds/advance", headers=AUTH).status_code == 200
    assert _stage(stack) == RoundStage.SHARE_KEYS


def test_a_silo_config_must_name_a_real_silo() -> None:
    with pytest.raises(ValueError, match="not one of"):
        SiloConfig(silo_id="silo_99", n_silos=5)
    with pytest.raises(ValueError, match="rounds"):
        SiloConfig(silo_id="silo_00", rounds=0)
    with pytest.raises(ValueError, match="poll_seconds"):
        SiloConfig(silo_id="silo_00", poll_seconds=0.0)
    with pytest.raises(ValueError, match="max_polls"):
        SiloConfig(silo_id="silo_00", max_polls=0)


def test_the_operator_refuses_a_run_with_no_rounds(stack: Stack) -> None:
    with pytest.raises(ValueError, match="rounds"):
        run_operator(stack.http, api_key=OPERATOR_KEY, rounds=0)


def test_a_silo_finds_no_round_before_the_operator_opens_one(stack: Stack) -> None:
    driver = SiloDriver(stack.clients["silo_00"], stack.http, api_key=silo_key("silo_00"))
    assert driver.current_round() is None


def test_a_silo_reads_the_scheme_and_nonce_the_coordinator_published(stack: Stack) -> None:
    """Every silo must quantize with identical parameters or the ring sum is meaningless."""
    opened = stack.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    assert opened.status_code == 200
    driver = SiloDriver(stack.clients["silo_00"], stack.http, api_key=silo_key("silo_00"))

    current = driver.current_round()
    assert current is not None
    round_id, stage, nonce = current
    assert round_id == 0
    assert stage is RoundStage.ADVERTISE_KEYS
    assert nonce == stack.coordinator.round_nonce

    scheme = driver.scheme()
    assert scheme.dimension == stack.coordinator.scheme.dimension
    assert scheme.scale == stack.coordinator.scheme.scale
    assert scheme.clip_norm == stack.coordinator.scheme.clip_norm


def test_waiting_for_a_stage_that_already_passed_returns_false(stack: Stack) -> None:
    """A silo that overslept must learn it missed the stage instead of blocking forever."""
    stack.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    for name in stack.ids:
        driver = SiloDriver(stack.clients[name], stack.http, api_key=silo_key(name))
        driver.advertise_keys(0, stack.coordinator.round_nonce)
    stack.http.post("/v1/rounds/advance", headers=AUTH)  # now at SHARE_KEYS

    late = SiloDriver(stack.clients["silo_00"], stack.http, api_key=silo_key("silo_00"))
    assert late.wait_for_stage(RoundStage.ADVERTISE_KEYS, poll_seconds=0.0, attempts=2) is False
    assert late.wait_for_stage(RoundStage.SHARE_KEYS, poll_seconds=0.0, attempts=2) is True


def test_wait_for_stage_gives_up_instead_of_blocking_forever(stack: Stack) -> None:
    stack.http.post("/v1/rounds", json={"round_id": 0}, headers=AUTH)
    driver = SiloDriver(stack.clients["silo_00"], stack.http, api_key=silo_key("silo_00"))
    calls: list[float] = []
    assert (
        driver.wait_for_stage(RoundStage.UNMASK, poll_seconds=0.0, attempts=3, sleep=calls.append)
        is False
    )
    assert len(calls) == 3


def test_a_bad_api_key_is_reported_not_swallowed(stack: Stack) -> None:
    driver = SiloDriver(stack.clients["silo_00"], stack.http, api_key="wrong-key")
    with pytest.raises(HttpProtocolError) as info:
        driver.scheme()
    assert info.value.status_code == 401
