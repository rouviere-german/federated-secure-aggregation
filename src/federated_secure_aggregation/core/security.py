"""API-key authentication and role-based route authorization.

Keys are high-entropy random tokens, so a plain SHA-256 digest is an adequate at-rest form (a
slow password hash buys nothing against a 256-bit secret). The presented key is hashed first and
the digest is compared against *every* configured digest with ``hmac.compare_digest`` and no
early exit, so response timing reveals neither key prefixes nor which configured key matched.

Each configured digest carries a **role**. One key shared by every caller makes "operator
routes" a label rather than a control: whoever can post a masked input can also open rounds and
advance stages. It also collapses the rate limit, because the bucket is keyed on the presented
key, so every caller sharing one key shares one bucket and a normal number of pollers looks
like a flood. Issuing one key per identity fixes both at once, and :func:`require_role` is what
makes the role mean something at the route.
"""

from __future__ import annotations

import hashlib
import hmac
import secrets
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Annotated

from fastapi import Depends, Request

from federated_secure_aggregation.core.errors import AppError
from federated_secure_aggregation.core.logging import get_logger

__all__ = [
    "API_KEY_HEADER",
    "DEFAULT_ROLE",
    "ROLE_OPERATOR",
    "ROLE_SILO",
    "ApiKeyVerifier",
    "AuthenticatedPrincipal",
    "OperatorPrincipal",
    "Principal",
    "SiloPrincipal",
    "extract_api_key",
    "generate_api_key",
    "hash_api_key",
    "require_api_key",
    "require_role",
]

API_KEY_HEADER = "X-API-Key"
_MAX_KEY_LENGTH = 512

# Opens rounds and asks the coordinator to advance stages. Holds no key material.
ROLE_OPERATOR = "operator"
# Carries the four protocol message kinds and reads its own share inbox.
ROLE_SILO = "silo"
# What a key configured without an explicit role gets: the least-privileged one.
DEFAULT_ROLE = ROLE_SILO

_log = get_logger("federated_secure_aggregation.security")


def hash_api_key(raw_key: str) -> str:
    """Lowercase hex SHA-256 of an API key, as stored in ``APP_API_KEY_SHA256``."""
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def generate_api_key() -> tuple[str, str]:
    """Return a new (raw_key, sha256_hex) pair. Show the raw key once; store only the digest."""
    raw = secrets.token_urlsafe(32)
    return raw, hash_api_key(raw)


@dataclass(frozen=True, slots=True)
class Principal:
    """An authenticated caller: a non-secret prefix of its key digest, plus its role."""

    key_id: str
    role: str = DEFAULT_ROLE


class ApiKeyVerifier:
    """Verifies presented keys against configured digests and reports each key's role.

    ``keys`` is either an iterable of hex digests (all taking :data:`DEFAULT_ROLE`) or a
    mapping of hex digest -> role.
    """

    def __init__(self, keys: Iterable[str] | Mapping[str, str]) -> None:
        items = (
            keys.items()
            if isinstance(keys, Mapping)
            else ((digest, DEFAULT_ROLE) for digest in keys)
        )
        self._entries = tuple((bytes.fromhex(digest), role) for digest, role in items)

    @property
    def configured(self) -> bool:
        return bool(self._entries)

    def verify(self, presented: str | None) -> Principal | None:
        """Return the principal for a valid key, else None. Fails closed with no keys set."""
        if not presented or len(presented) > _MAX_KEY_LENGTH:
            return None
        digest = hashlib.sha256(presented.encode("utf-8")).digest()
        matched: tuple[bytes, str] | None = None
        for candidate, role in self._entries:
            if hmac.compare_digest(digest, candidate):
                matched = (candidate, role)
        if matched is None:
            return None
        return Principal(key_id=matched[0].hex()[:12], role=matched[1])


def extract_api_key(headers: Mapping[str, str]) -> str | None:
    """Read the key from ``X-API-Key`` or ``Authorization: Bearer <key>``.

    Starlette's ``Headers`` is case-insensitive; plain mappings must use lowercase names.
    """
    value = headers.get("x-api-key")
    if value:
        return value
    auth = headers.get("authorization")
    if auth and auth[:7].lower() == "bearer ":
        return auth[7:].strip() or None
    return None


def require_api_key(request: Request) -> Principal:
    """FastAPI dependency: authenticate the caller or raise a uniform 401."""
    verifier: ApiKeyVerifier = request.app.state.api_key_verifier
    principal = verifier.verify(extract_api_key(request.headers))
    if principal is None:
        request.app.state.metrics.auth_failures_total.inc()
        raise AppError(
            401,
            "unauthorized",
            "A valid API key is required.",
            headers={"WWW-Authenticate": "ApiKey"},
        )
    return principal


def require_role(*allowed: str) -> Callable[[Request], Principal]:
    """Dependency factory: authenticate, then require one of ``allowed`` roles.

    A 403 (not 401) and no hint about what the route wanted: the caller proved who it is and
    the answer is that this identity may not do this.
    """
    permitted = frozenset(allowed)

    def dependency(request: Request) -> Principal:
        principal = require_api_key(request)
        if principal.role not in permitted:
            _log.info(
                "role_refused",
                key_id=principal.key_id,
                role=principal.role,
                path=request.url.path,
            )
            raise AppError(403, "forbidden", "This API key may not use this route.")
        return principal

    return dependency


AuthenticatedPrincipal = Annotated[Principal, Depends(require_api_key)]
OperatorPrincipal = Annotated[Principal, Depends(require_role(ROLE_OPERATOR))]
SiloPrincipal = Annotated[Principal, Depends(require_role(ROLE_SILO))]
