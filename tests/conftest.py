"""Shared fixtures. Every test runs offline: non-loopback sockets are blocked."""

from __future__ import annotations

import logging
import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import structlog
from hypothesis import HealthCheck, settings

from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import RoundConfig, SecAggCoordinator
from federated_secure_aggregation.secagg.session import SimulationClock

settings.register_profile("dev", max_examples=60, deadline=None)
settings.register_profile(
    "ci",
    max_examples=200,
    deadline=None,
    derandomize=True,
    suppress_health_check=[HealthCheck.too_slow],
)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "dev"))

REPO_ROOT = Path(__file__).resolve().parents[1]
_LOOPBACK = frozenset({"127.0.0.1", "::1", "localhost"})


@pytest.fixture(autouse=True)
def _offline_and_clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    for name in list(os.environ):
        if name.startswith("APP_"):
            monkeypatch.delenv(name, raising=False)

    real_connect = socket.socket.connect

    def guarded_connect(self: socket.socket, address: Any) -> None:
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in _LOOPBACK:
            raise RuntimeError(f"tests must not use the network (attempted {host})")
        real_connect(self, address)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)


@pytest.fixture
def restore_logging() -> Iterator[None]:
    root = logging.getLogger()
    handlers, level = root.handlers[:], root.level
    yield
    structlog.reset_defaults()
    root.handlers = handlers
    root.setLevel(level)
    logging.getLogger("uvicorn.access").disabled = False


@pytest.fixture
def repo_root() -> Path:
    return REPO_ROOT


@dataclass(slots=True)
class Federation:
    """A coordinator, its clients and the shared clock, ready to run rounds in process."""

    coordinator: SecAggCoordinator
    clients: dict[str, SecAggClient]
    clock: SimulationClock
    identities: dict[str, IdentityKeyPair]
    coordinator_identity: IdentityKeyPair
    config: RoundConfig

    @property
    def ids(self) -> list[str]:
        return sorted(self.clients)

    def updates(self, seed: int = 0, scale: float = 0.05) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(seed)
        return {
            name: rng.normal(size=self.config.dimension) * scale for name in sorted(self.clients)
        }


def build_federation(
    *,
    n_clients: int = 5,
    threshold: int = 3,
    dimension: int = 16,
    clip_norm: float = 1.0,
    stage_timeout_seconds: float = 5.0,
    max_round_bytes: int = 64 << 20,
) -> Federation:
    """Wire a small federation with real keys; used by most protocol tests."""
    ids = [f"silo_{index:02d}" for index in range(n_clients)]
    identities = {name: IdentityKeyPair.generate() for name in ids}
    public = {name: identity.public_bytes for name, identity in identities.items()}
    coordinator_identity = IdentityKeyPair.generate()
    config = RoundConfig(
        dimension=dimension,
        threshold=threshold,
        clip_norm=clip_norm,
        max_clients=n_clients,
        stage_timeout_seconds=stage_timeout_seconds,
        max_round_bytes=max_round_bytes,
    )
    clock = SimulationClock()
    coordinator = SecAggCoordinator(coordinator_identity, public, config, clock=clock)
    clients = {
        name: SecAggClient(
            name,
            identities[name],
            peer_identity_keys=public,
            coordinator_identity_key=coordinator_identity.public_bytes,
        )
        for name in ids
    }
    return Federation(
        coordinator=coordinator,
        clients=clients,
        clock=clock,
        identities=identities,
        coordinator_identity=coordinator_identity,
        config=config,
    )


@pytest.fixture
def federation() -> Federation:
    return build_federation()
