from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from federated_secure_aggregation.core.config import Settings, get_settings

pytestmark = pytest.mark.unit


def test_defaults_fail_closed() -> None:
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == []
    assert settings.cors_allow_origins == []
    assert settings.enable_docs is False
    assert settings.metrics_require_auth is True
    assert settings.host == "127.0.0.1"
    # Protocol defaults must also fail closed: no roster path that exists by accident, and a
    # threshold above 1 so a single silo can never reconstruct a round on its own.
    assert settings.secagg_threshold >= 2
    assert settings.target_epsilon > 0


def test_environment_variables_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("APP_API_KEY_SHA256", f"{'a' * 64}, {'b' * 64}")
    monkeypatch.setenv("APP_CORS_ALLOW_ORIGINS", "https://app.example.com,http://localhost:3000")
    monkeypatch.setenv("APP_SECAGG_THRESHOLD", "4")
    monkeypatch.setenv("APP_NOISE_MULTIPLIER", "1.5")
    monkeypatch.setenv("APP_ROSTER_PATH", "/run/secrets/roster.json")
    settings = Settings(_env_file=None)
    assert settings.api_key_sha256 == ["a" * 64, "b" * 64]
    assert settings.cors_allow_origins == ["https://app.example.com", "http://localhost:3000"]
    assert settings.secagg_threshold == 4
    assert settings.noise_multiplier == 1.5
    assert settings.roster_path == Path("/run/secrets/roster.json")


@pytest.mark.parametrize("origins", [["*"], ["https://example.com/"], ["example.com"]])
def test_wildcard_or_malformed_origins_are_rejected(origins: list[str]) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, cors_allow_origins=origins)


@pytest.mark.parametrize("digest", ["A" * 64, "abc", "g" * 64])
def test_non_sha256_key_hashes_are_rejected(digest: str) -> None:
    with pytest.raises(ValidationError):
        Settings(_env_file=None, api_key_sha256=[digest])


def test_env_example_documents_every_setting(repo_root: Path) -> None:
    lines = (repo_root / ".env.example").read_text(encoding="utf-8").splitlines()
    documented = {
        line.split("=", 1)[0].strip() for line in lines if line.strip() and not line.startswith("#")
    }
    expected = {f"APP_{name.upper()}" for name in Settings.model_fields}
    assert documented == expected


def test_a_key_digest_cannot_hold_two_roles() -> None:
    """One key per identity is the point; a digest in both lists would silently be an operator."""
    digest = "a" * 64
    with pytest.raises(ValidationError, match="cannot be both"):
        Settings(_env_file=None, api_key_sha256=[digest], operator_api_key_sha256=[digest])


def test_api_key_roles_maps_each_digest_to_its_role() -> None:
    settings = Settings(
        _env_file=None, api_key_sha256=["a" * 64, "b" * 64], operator_api_key_sha256=["c" * 64]
    )
    assert settings.api_key_roles() == {
        "a" * 64: "silo",
        "b" * 64: "silo",
        "c" * 64: "operator",
    }


def test_get_settings_is_cached() -> None:
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
