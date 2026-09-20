"""The hash-chained round log and the Prometheus round metrics."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from prometheus_client import CollectorRegistry, generate_latest

from federated_secure_aggregation.monitoring.chain import (
    GENESIS,
    AuditChainError,
    RoundAuditLog,
    RoundRecord,
)
from federated_secure_aggregation.monitoring.chain import main as chain_main
from federated_secure_aggregation.monitoring.metrics import create_round_metrics

pytestmark = pytest.mark.unit


def _record(round_id: int, *, accepted: bool = True) -> RoundRecord:
    return RoundRecord(
        round_id=round_id,
        stage="COMPLETED" if accepted else "REJECTED",
        survivors=["silo_00", "silo_01", "silo_02"],
        dropouts=["silo_03"],
        excluded=[],
        accepted=accepted,
        aggregate_digest="a" * 64,
        aggregate_norm=1.5,
        norm_bound=3.0,
        epsilon_spent=0.5 * round_id,
        holdout_auc=0.8,
        bytes_received=1234,
        duration_seconds=0.25,
    )


def test_an_empty_log_has_the_genesis_head(tmp_path: Path) -> None:
    log = RoundAuditLog(tmp_path / "audit.jsonl")
    assert log.head == GENESIS
    assert log.verify() == 0
    assert list(log.entries()) == []


def test_appending_links_entries_and_verify_accepts_the_head(tmp_path: Path) -> None:
    log = RoundAuditLog(tmp_path / "audit.jsonl")
    heads = [log.append(_record(index)) for index in range(4)]
    assert len(set(heads)) == 4
    assert log.head == heads[-1]
    assert log.verify(heads[-1]) == 4
    entries = list(log.entries())
    assert [entry.index for entry in entries] == [0, 1, 2, 3]
    assert entries[0].prev_hash == GENESIS
    assert entries[3].prev_hash == entries[2].entry_hash


def test_editing_a_record_breaks_the_chain(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = RoundAuditLog(path)
    for index in range(3):
        log.append(_record(index))
    lines = path.read_text(encoding="utf-8").splitlines()
    payload = json.loads(lines[1])
    payload["record"]["epsilon_spent"] = 0.0
    lines[1] = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="entry 1 has been modified"):
        log.verify()


def test_removing_the_tail_is_only_detected_with_an_expected_head(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = RoundAuditLog(path)
    for index in range(3):
        log.append(_record(index))
    head = log.head
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join(lines[:2]) + "\n", encoding="utf-8")
    assert log.verify() == 2  # the truncated chain is internally consistent
    with pytest.raises(AuditChainError, match="entries may be missing"):
        log.verify(head)


def test_reordering_entries_is_detected(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = RoundAuditLog(path)
    for index in range(3):
        log.append(_record(index))
    lines = path.read_text(encoding="utf-8").splitlines()
    path.write_text("\n".join([lines[0], lines[2], lines[1]]) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="claims index"):
        log.verify()


def test_malformed_lines_are_reported(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_text("not json\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="not valid JSON"):
        RoundAuditLog(path).verify()
    path.write_text(json.dumps({"index": 0}) + "\n", encoding="utf-8")
    with pytest.raises(AuditChainError, match="not a valid chain entry"):
        RoundAuditLog(path).verify()


def test_blank_lines_are_ignored(tmp_path: Path) -> None:
    path = tmp_path / "audit.jsonl"
    log = RoundAuditLog(path)
    log.append(_record(0))
    path.write_text(path.read_text(encoding="utf-8") + "\n\n", encoding="utf-8")
    assert log.verify() == 1


def test_the_verify_cli_reports_success_and_failure(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "audit.jsonl"
    log = RoundAuditLog(path)
    log.append(_record(0))
    assert chain_main(["verify", str(path), "--expect-head", log.head]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["entries"] == 1

    assert chain_main(["verify", str(path), "--expect-head", "0" * 64]) == 1
    failure = json.loads(capsys.readouterr().out)
    assert failure["ok"] is False


def test_round_metrics_record_every_outcome() -> None:
    registry = CollectorRegistry()
    metrics = create_round_metrics(registry)
    metrics.observe_round(
        duration_seconds=0.5,
        participants=5,
        dropouts=0,
        accepted=True,
        failed=False,
        epsilon_spent=1.5,
        aggregate_norm=2.0,
        holdout_auc=0.81,
        bytes_received=1000,
    )
    metrics.observe_round(
        duration_seconds=0.2,
        participants=3,
        dropouts=2,
        accepted=False,
        failed=False,
        epsilon_spent=1.5,
        aggregate_norm=9.0,
        holdout_auc=None,
        bytes_received=900,
    )
    metrics.observe_round(
        duration_seconds=0.1,
        participants=0,
        dropouts=0,
        accepted=False,
        failed=True,
        epsilon_spent=1.5,
        aggregate_norm=0.0,
        holdout_auc=None,
        bytes_received=10,
    )
    text = generate_latest(registry).decode("utf-8")
    assert "fedsecagg_rounds_total 3.0" in text
    assert "fedsecagg_rounds_rejected_total 1.0" in text
    assert "fedsecagg_rounds_failed_total 1.0" in text
    assert "fedsecagg_dropouts_total 2.0" in text
    assert "fedsecagg_epsilon_spent 1.5" in text
    assert "fedsecagg_global_holdout_auc 0.81" in text


def test_metrics_carry_no_per_client_labels() -> None:
    """Per-client labels would turn /metrics into a participation log."""
    registry = CollectorRegistry()
    metrics = create_round_metrics(registry)
    metrics.observe_round(
        duration_seconds=0.1,
        participants=2,
        dropouts=0,
        accepted=True,
        failed=False,
        epsilon_spent=0.0,
        aggregate_norm=0.0,
        holdout_auc=None,
        bytes_received=0,
    )
    text = generate_latest(registry).decode("utf-8")
    assert "silo_" not in text
    assert "client=" not in text
