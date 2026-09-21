"""Wiring the coordinator service from configuration and mounted key material.

Identity keys are never generated at start-up and never committed: the coordinator reads its
own Ed25519 private key from a file, and the silo roster (id -> public key) from another. In
compose both are bind-mounted read-only; ``scripts/generate_identities.py`` creates them once
outside the repository. A coordinator that cannot read either file refuses to start, rather
than silently accepting unauthenticated participants.
"""

from __future__ import annotations

import base64
import binascii
import json
from pathlib import Path

from fastapi import FastAPI

from federated_secure_aggregation.core.config import Settings, get_settings
from federated_secure_aggregation.crypto.keys import KEY_BYTES, IdentityKeyPair
from federated_secure_aggregation.monitoring.metrics import create_round_metrics
from federated_secure_aggregation.privacy.ledger import BudgetLedger
from federated_secure_aggregation.secagg.coordinator import RoundConfig, SecAggCoordinator
from federated_secure_aggregation.secagg.messages import CLIENT_ID_PATTERN
from federated_secure_aggregation.service.api import (
    CentralPrivacy,
    CoordinatorService,
    create_coordinator_app,
)

__all__ = ["RosterError", "build_service_from_env", "load_identity", "load_roster"]


class RosterError(ValueError):
    """The roster or identity file is missing, malformed or holds an unusable key."""


def _decode_key(value: object, label: str) -> bytes:
    if not isinstance(value, str):
        raise RosterError(f"{label} must be a base64 string")
    try:
        raw = base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise RosterError(f"{label} is not valid base64") from exc
    if len(raw) != KEY_BYTES:
        raise RosterError(f"{label} must decode to {KEY_BYTES} bytes, got {len(raw)}")
    return raw


def load_roster(path: Path) -> dict[str, bytes]:
    """Read ``{"silo_00": "<base64 ed25519 public key>", ...}``."""
    if not path.is_file():
        raise RosterError(f"roster file {path} does not exist")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RosterError(f"roster file {path} is not valid JSON") from exc
    if not isinstance(data, dict) or not data:
        raise RosterError("roster must be a non-empty object of client id -> public key")
    import re  # noqa: PLC0415 - only needed to validate ids once at start-up

    roster: dict[str, bytes] = {}
    for client_id, value in data.items():
        if not isinstance(client_id, str) or not re.fullmatch(CLIENT_ID_PATTERN, client_id):
            raise RosterError(f"invalid client id in roster: {client_id!r}")
        roster[client_id] = _decode_key(value, f"public key for {client_id}")
    if len(set(roster.values())) != len(roster):
        raise RosterError("two silos share an identity key")
    return roster


def load_identity(path: Path) -> IdentityKeyPair:
    """Read ``{"private_key_b64": "..."}`` and rebuild the coordinator's identity key."""
    if not path.is_file():
        raise RosterError(f"identity file {path} does not exist")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RosterError(f"identity file {path} is not valid JSON") from exc
    if not isinstance(data, dict):
        raise RosterError("identity file must be a JSON object")
    return IdentityKeyPair.from_private_bytes(
        _decode_key(data.get("private_key_b64"), "coordinator private key")
    )


def build_service_from_env(settings: Settings | None = None) -> FastAPI:
    """Build the coordinator app from ``APP_*`` settings and the mounted key files."""
    cfg = settings if settings is not None else get_settings()
    roster = load_roster(cfg.roster_path)
    identity = load_identity(cfg.coordinator_identity_path)
    coordinator = SecAggCoordinator(
        identity,
        roster,
        RoundConfig(
            dimension=cfg.model_dimension,
            threshold=cfg.secagg_threshold,
            clip_norm=cfg.clip_norm,
            max_clients=max(len(roster), cfg.secagg_threshold),
            stage_timeout_seconds=cfg.stage_timeout_seconds,
        ),
    )
    # The ledger is persisted next to the rest of the run's state, so restarting the container
    # does not reset the privacy budget to zero -- which would make the claimed epsilon a
    # statement about uptime rather than about the data.
    cfg.data_dir.mkdir(parents=True, exist_ok=True)
    privacy = CentralPrivacy(
        ledger=BudgetLedger(
            noise_multiplier=cfg.noise_multiplier,
            target_epsilon=cfg.target_epsilon,
            delta=cfg.target_delta,
            path=cfg.data_dir / "budget.json",
        ),
        clip_norm=cfg.clip_norm,
    )
    service = CoordinatorService(coordinator, create_round_metrics(), privacy)
    return create_coordinator_app(service, cfg)
