from __future__ import annotations

import hmac

import pytest

from federated_secure_aggregation.core.security import (
    DEFAULT_ROLE,
    ROLE_OPERATOR,
    ROLE_SILO,
    ApiKeyVerifier,
    extract_api_key,
    generate_api_key,
    hash_api_key,
)

pytestmark = pytest.mark.unit


def test_generated_keys_verify_against_their_digest() -> None:
    raw, digest = generate_api_key()
    assert digest == hash_api_key(raw)
    verifier = ApiKeyVerifier([hash_api_key("other-key"), digest])
    principal = verifier.verify(raw)
    assert principal is not None
    assert principal.key_id == digest[:12]
    assert principal.role == DEFAULT_ROLE
    assert verifier.verify("wrong-key") is None
    assert verifier.verify(None) is None
    assert verifier.verify("") is None
    assert verifier.verify("x" * 600) is None


def test_a_verifier_reports_the_role_its_key_was_configured_with() -> None:
    """Roles are what stop a silo key from opening rounds; the verifier carries them."""
    operator, silo = "operator-key-value", "silo-key-value"
    verifier = ApiKeyVerifier(
        {hash_api_key(operator): ROLE_OPERATOR, hash_api_key(silo): ROLE_SILO}
    )
    operator_principal = verifier.verify(operator)
    silo_principal = verifier.verify(silo)
    assert operator_principal is not None
    assert silo_principal is not None
    assert operator_principal.role == ROLE_OPERATOR
    assert silo_principal.role == ROLE_SILO
    # Distinct keys mean distinct key ids, which is what gives each identity its own rate bucket.
    assert operator_principal.key_id != silo_principal.key_id


def test_no_configured_keys_fails_closed() -> None:
    verifier = ApiKeyVerifier([])
    assert not verifier.configured
    assert verifier.verify("anything") is None


def test_every_configured_digest_is_compared_without_early_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[int] = []
    real = hmac.compare_digest

    def counting(a: bytes, b: bytes) -> bool:
        calls.append(1)
        return real(a, b)

    monkeypatch.setattr(hmac, "compare_digest", counting)
    keys = [f"key-{i}" for i in range(5)]
    verifier = ApiKeyVerifier(hash_api_key(k) for k in keys)
    assert verifier.verify(keys[0]) is not None
    assert len(calls) == 5


@pytest.mark.parametrize(
    ("headers", "expected"),
    [
        ({"x-api-key": "k1"}, "k1"),
        ({"authorization": "Bearer abc"}, "abc"),
        ({"authorization": "bearer  abc "}, "abc"),
        ({"authorization": "Basic abc"}, None),
        ({"authorization": "Bearer "}, None),
        ({}, None),
    ],
)
def test_extract_api_key(headers: dict[str, str], expected: str | None) -> None:
    assert extract_api_key(headers) == expected
