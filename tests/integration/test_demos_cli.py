"""The offline demo and every CLI subcommand, exercised end to end.

The demo is the repo's headline claim: twenty rounds of secure federated training in which two
silos drop out mid-run and one silo poisons its update. These tests assert the outcomes the
README advertises rather than just "it exited zero", so a regression in dropout recovery, the
norm bound or the budget ledger fails here.
"""

from __future__ import annotations

import json
import runpy
import sys
from pathlib import Path
from typing import Any

import pytest
from opentelemetry import trace as otel_trace

from federated_secure_aggregation import cli
from federated_secure_aggregation.core import tracing
from federated_secure_aggregation.core.security import hash_api_key
from federated_secure_aggregation.demo import (
    ATTACK_ROUND,
    DROPOUT_ROUND,
    ROUNDS,
    TARGET_EPSILON,
    run_demo,
)

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def demo_summary(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    """One demo run shared by the assertions below; it takes a few seconds."""
    return dict(run_demo(tmp_path_factory.mktemp("demo")))


def test_demo_runs_every_round_and_keeps_one_rejection(demo_summary: dict[str, Any]) -> None:
    assert demo_summary["rounds"] == ROUNDS
    assert demo_summary["accepted_rounds"] == ROUNDS - 1
    assert demo_summary["rejected_rounds"] == 1


def test_demo_rejects_the_poisoned_round_at_the_aggregate_level(
    demo_summary: dict[str, Any],
) -> None:
    """The coordinator cannot see which silo cheated, only that the aggregate is too long."""
    assert demo_summary["rejected_round_ids"] == [ATTACK_ROUND]


def test_demo_completes_the_round_in_which_two_silos_drop_out(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["dropout_round_ids"] == [DROPOUT_ROUND]
    assert len(demo_summary["dropped_silos"]) == 2
    survivors = demo_summary["survivors_in_dropout_round"]
    assert len(survivors) == 3
    assert not set(survivors) & set(demo_summary["dropped_silos"])


def test_demo_stays_inside_the_privacy_budget(demo_summary: dict[str, Any]) -> None:
    assert demo_summary["target_epsilon"] == TARGET_EPSILON
    assert 0.0 < demo_summary["epsilon_spent"] <= TARGET_EPSILON
    assert demo_summary["noise_multiplier"] > 1.0


def test_demo_reports_the_utility_cost_of_the_guarantee(demo_summary: dict[str, Any]) -> None:
    """Noise costs AUC. The demo must report that cost rather than hide it."""
    assert 0.5 < demo_summary["federated_holdout_auc"] <= 1.0
    assert demo_summary["federated_holdout_auc_no_dp"] > demo_summary["federated_holdout_auc"]
    assert demo_summary["dp_utility_cost_auc"] == pytest.approx(
        demo_summary["federated_holdout_auc_no_dp"] - demo_summary["federated_holdout_auc"],
        abs=1e-4,
    )
    assert 0.5 < demo_summary["centralized_holdout_auc"] <= 1.0
    assert 0.0 < demo_summary["local_only_holdout_auc"] <= 1.0


def test_demo_verifies_its_audit_chain_and_registry_roundtrip(
    demo_summary: dict[str, Any],
) -> None:
    assert demo_summary["audit_chain_entries"] == ROUNDS
    assert len(demo_summary["audit_chain_head"]) == 64
    assert demo_summary["registry_roundtrip_ok"] is True
    assert isinstance(demo_summary["model_version"], str)


def test_demo_learning_is_reproducible_bit_for_bit(tmp_path: Path) -> None:
    """Same seed, same machine, same model.

    Two fresh runs must agree on the partition, every per-round aggregate digest and every
    published number. They deliberately do *not* agree on the audit chain hashes: each round
    record commits to its wall-clock ``duration_seconds``, so the chain is a tamper-evident log
    of one execution rather than a reproducible artifact. ``model_version`` differs for the same
    reason -- the run manifest carries a timestamp. The test pins both facts so neither the
    reproducibility claim nor its exception can drift silently.
    """
    first = run_demo(tmp_path / "a")
    second = run_demo(tmp_path / "b")

    volatile = {"audit_chain_head", "model_version"}
    assert {key: value for key, value in first.items() if key not in volatile} == {
        key: value for key, value in second.items() if key not in volatile
    }
    assert first["audit_chain_head"] != second["audit_chain_head"]

    def digests(workdir: Path) -> list[object]:
        records = (tmp_path / workdir / "round_audit.jsonl").read_text(encoding="utf-8")
        return [
            json.loads(line)["record"]["aggregate_digest"]
            for line in records.splitlines()
            if line.strip()
        ]

    assert digests(Path("a")) == digests(Path("b"))
    assert any(digest is not None for digest in digests(Path("a")))


def test_demo_repeated_in_the_same_workdir_does_not_inherit_a_spent_budget(
    tmp_path: Path,
) -> None:
    """Regression: the ledger and audit chain persist on purpose, so a second run in the same
    directory used to start at the target epsilon and refuse all twenty rounds."""
    first = run_demo(tmp_path)
    second = run_demo(tmp_path)
    assert first["accepted_rounds"] == second["accepted_rounds"] == ROUNDS - 1
    assert second["audit_chain_entries"] == ROUNDS
    assert first["epsilon_spent"] == second["epsilon_spent"]


@pytest.mark.usefixtures("restore_logging")
def test_cli_demo_prints_one_json_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert cli.main(["demo", "--workdir", str(tmp_path)]) == 0
    output = capsys.readouterr().out
    # The per-round markdown table is printed first, then exactly one JSON object.
    payload = json.loads(output[output.index("{") :])
    assert payload["rounds"] == ROUNDS
    assert payload["rejected_round_ids"] == [ATTACK_ROUND]
    assert "| round | survivors |" in output


def test_cli_budget_reports_the_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_demo(tmp_path)
    capsys.readouterr()  # discard the demo's per-round table
    assert cli.main(["budget", str(tmp_path / "budget.json")]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["rounds_spent"] == ROUNDS - 1
    assert payload["target_epsilon"] == TARGET_EPSILON
    assert 0.0 < payload["epsilon_spent"] <= TARGET_EPSILON
    assert payload["rounds_remaining"] >= 0


def test_cli_audit_verify_accepts_the_real_head_and_rejects_a_forged_one(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = run_demo(tmp_path)
    path = tmp_path / "round_audit.jsonl"
    head = str(summary["audit_chain_head"])
    assert cli.main(["audit", "verify", str(path), "--expect-head", head]) == 0
    assert cli.main(["audit", "verify", str(path), "--expect-head", "0" * 64]) == 1
    capsys.readouterr()


def test_cli_generate_api_key(capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["generate-api-key"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["api_key_sha256"] == hash_api_key(payload["api_key"])


def test_cli_generate_identity_prints_a_usable_ed25519_pair(
    capsys: pytest.CaptureFixture[str],
) -> None:
    import base64

    from federated_secure_aggregation.crypto.keys import IdentityKeyPair

    assert cli.main(["generate-identity"]) == 0
    payload = json.loads(capsys.readouterr().out)
    private = base64.b64decode(payload["private_key_b64"])
    public = base64.b64decode(payload["public_key_b64"])
    identity = IdentityKeyPair.from_private_bytes(private)
    assert identity.public_bytes == public


def test_cli_serve_refuses_to_start_without_mounted_key_material(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """No roster, no coordinator: it must fail loudly, not accept unauthenticated silos."""
    from federated_secure_aggregation.service.runtime import RosterError

    monkeypatch.chdir(tmp_path)
    with pytest.raises(RosterError, match="roster file"):
        cli.main(["serve"])


@pytest.mark.usefixtures("restore_logging")
def test_cli_serve_configures_uvicorn_safely(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    import base64

    import uvicorn

    from federated_secure_aggregation.crypto.keys import IdentityKeyPair

    secrets_dir = tmp_path / "secrets"
    secrets_dir.mkdir()
    coordinator = IdentityKeyPair.generate()
    (secrets_dir / "coordinator_identity.json").write_text(
        json.dumps(
            {"private_key_b64": base64.b64encode(coordinator.private_bytes).decode("ascii")}
        ),
        encoding="utf-8",
    )
    (secrets_dir / "roster.json").write_text(
        json.dumps(
            {
                f"silo_{index:02d}": base64.b64encode(
                    IdentityKeyPair.generate().public_bytes
                ).decode("ascii")
                for index in range(3)
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)

    captured: dict[str, Any] = {}

    def fake_run(app: object, **kwargs: Any) -> None:
        captured.update(kwargs, app=app)

    monkeypatch.setattr(uvicorn, "run", fake_run)
    monkeypatch.setattr(tracing, "_installed", [])
    monkeypatch.setattr(otel_trace, "set_tracer_provider", lambda provider: None)
    assert cli.main(["serve", "--port", "8123"]) == 0
    assert captured["host"] == "127.0.0.1"
    assert captured["port"] == 8123
    assert captured["server_header"] is False
    assert captured["access_log"] is False


def test_module_entry_point(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["federated_secure_aggregation", "generate-api-key"])
    with pytest.raises(SystemExit) as exit_info:
        runpy.run_module("federated_secure_aggregation", run_name="__main__")
    assert exit_info.value.code == 0
    assert "api_key_sha256" in capsys.readouterr().out
