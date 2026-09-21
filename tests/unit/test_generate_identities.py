"""Key material generated for compose must be usable by compose.

The script is the only supported way to create a federation's identities and transport keys,
and its output is consumed by a POSIX shell (`set -a && . ./secrets/compose.env && set +a`) and
by the container at start-up. Two properties therefore matter as much as the key material
itself: one key per identity, and LF line endings -- a CRLF `compose.env` sourced in bash gives
every value a trailing carriage return, which turns a digest into one that matches nothing and
a port into a parse error, with no error message that points at the cause.
"""

from __future__ import annotations

import base64
import importlib.util
import json
from pathlib import Path
from types import ModuleType

import pytest

from federated_secure_aggregation.core.security import hash_api_key
from federated_secure_aggregation.crypto.keys import KEY_BYTES

pytestmark = pytest.mark.unit


@pytest.fixture
def generator(repo_root: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "generate_identities", repo_root / "scripts" / "generate_identities.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def generated(generator: ModuleType, tmp_path: Path) -> Path:
    generator.generate(tmp_path, 5)
    return tmp_path


def _env(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line and not line.startswith("#"):
            key, _, value = line.partition("=")
            values[key] = value
    return values


def test_every_generated_text_file_uses_lf(generated: Path) -> None:
    """The bug this pins: `. ./secrets/compose.env` in bash must not import stray \\r."""
    for path in sorted(generated.iterdir()):
        raw = path.read_bytes()
        assert b"\r\n" not in raw, f"{path.name} has CRLF line endings"


def test_compose_env_carries_one_key_per_identity(generated: Path) -> None:
    values = _env(generated / "compose.env")
    silo_keys = [values[f"APP_SILO_{index:02d}_API_KEY"] for index in range(5)]
    operator_key = values["APP_OPERATOR_API_KEY"]
    assert len({*silo_keys, operator_key}) == 6, "keys must be distinct per identity"

    digests = values["APP_SILO_API_KEY_SHA256"].split(",")
    assert digests == [hash_api_key(key) for key in silo_keys]
    assert values["APP_OPERATOR_API_KEY_SHA256"] == hash_api_key(operator_key)
    # A digest in both lists would make the operator routes reachable by a silo.
    assert values["APP_OPERATOR_API_KEY_SHA256"] not in digests


def test_each_silo_gets_its_own_identity_and_the_roster_holds_only_public_keys(
    generated: Path,
) -> None:
    roster = json.loads((generated / "roster.json").read_text(encoding="utf-8"))
    assert sorted(roster) == [f"silo_{index:02d}" for index in range(5)]
    privates = set()
    for silo_id, public_b64 in roster.items():
        assert len(base64.b64decode(public_b64, validate=True)) == KEY_BYTES
        identity = json.loads((generated / f"{silo_id}_identity.json").read_text(encoding="utf-8"))
        assert identity["public_key_b64"] == public_b64
        assert identity["private_key_b64"] not in public_b64
        privates.add(identity["private_key_b64"])
    assert len(privates) == 5
    # The roster is published to every silo, so a private key leaking into it would hand each
    # container every other silo's identity.
    assert not privates & set(roster.values())


def test_a_federation_needs_at_least_two_silos(generator: ModuleType) -> None:
    with pytest.raises(SystemExit):
        generator.main(["--silos", "1"])
