"""Transient HTTP failures must cost a retry, not a round.

A silo that treats a 429 or a 503 as fatal becomes a dropout for that round, and an operator
that treats one as fatal kills its container mid-run. Both are self-inflicted: the coordinator's
own rate limiter answers 429 with a ``Retry-After``, and it is the one the federation's own
polling will hit first.

The transport is driven here against a stub that returns a scripted sequence of responses, so
the retry policy is tested without a clock, a socket or a coordinator.
"""

from __future__ import annotations

from collections.abc import Iterator

import httpx
import pytest

from federated_secure_aggregation.service.client_http import (
    RetryPolicy,
    request_with_retry,
)

pytestmark = pytest.mark.unit

HEADERS = {"X-API-Key": "k"}


def _responses(*scripted: httpx.Response) -> Iterator[httpx.Response]:
    return iter(scripted)


def _client(script: list[httpx.Response]) -> httpx.Client:
    remaining = iter(script)

    def handler(request: httpx.Request) -> httpx.Response:
        return next(remaining)

    return httpx.Client(transport=httpx.MockTransport(handler), base_url="http://coordinator")


def test_a_rate_limited_request_is_retried_after_the_advertised_delay() -> None:
    slept: list[float] = []
    script = [
        httpx.Response(429, headers={"Retry-After": "2"}, json={}),
        httpx.Response(200, json={"ok": True}),
    ]
    with _client(script) as http:
        response = request_with_retry(
            http,
            "POST",
            "/v1/rounds/advance",
            headers=HEADERS,
            retry=RetryPolicy(attempts=3),
            sleep=slept.append,
        )
    assert response.status_code == 200
    assert slept == [2.0]


def test_server_errors_back_off_exponentially_and_then_give_up() -> None:
    slept: list[float] = []
    script = [httpx.Response(503, json={}) for _ in range(4)]
    with _client(script) as http:
        response = request_with_retry(
            http,
            "GET",
            "/v1/rounds/current",
            headers=HEADERS,
            retry=RetryPolicy(attempts=4, backoff_seconds=0.1, max_backoff_seconds=1.0),
            sleep=slept.append,
        )
    # Four attempts means three waits; the last failure is returned rather than raised, so the
    # caller decides whether the round is lost.
    assert response.status_code == 503
    assert slept == [0.1, 0.2, 0.4]


def test_the_backoff_is_capped() -> None:
    slept: list[float] = []
    script = [httpx.Response(429, json={}) for _ in range(5)]
    with _client(script) as http:
        request_with_retry(
            http,
            "GET",
            "/v1/rounds/current",
            headers=HEADERS,
            retry=RetryPolicy(attempts=5, backoff_seconds=1.0, max_backoff_seconds=2.0),
            sleep=slept.append,
        )
    assert slept == [1.0, 2.0, 2.0, 2.0]


def test_a_permanent_failure_is_not_retried() -> None:
    """403 is an answer, not weather. Retrying it only wastes the round's remaining time."""
    slept: list[float] = []
    script = [httpx.Response(403, json={}), httpx.Response(200, json={})]
    with _client(script) as http:
        response = request_with_retry(
            http,
            "POST",
            "/v1/rounds",
            headers=HEADERS,
            retry=RetryPolicy(attempts=3),
            sleep=slept.append,
        )
    assert response.status_code == 403
    assert slept == []


def test_an_unparsable_retry_after_falls_back_to_the_backoff() -> None:
    slept: list[float] = []
    script = [
        httpx.Response(429, headers={"Retry-After": "Wed, 21 Oct 2026 07:28:00 GMT"}, json={}),
        httpx.Response(200, json={}),
    ]
    with _client(script) as http:
        request_with_retry(
            http,
            "GET",
            "/v1/rounds/current",
            headers=HEADERS,
            retry=RetryPolicy(attempts=2, backoff_seconds=0.25),
            sleep=slept.append,
        )
    assert slept == [0.25]


def test_a_retry_policy_must_allow_at_least_one_attempt() -> None:
    with pytest.raises(ValueError, match="attempts"):
        RetryPolicy(attempts=0)
    with pytest.raises(ValueError, match="backoff"):
        RetryPolicy(backoff_seconds=0.0)
    with pytest.raises(ValueError, match="max_backoff"):
        RetryPolicy(backoff_seconds=2.0, max_backoff_seconds=1.0)
