"""A hash-chained, append-only record of what each round did.

One line of JSON per round, each carrying the hash of the line before it::

    entry_hash = sha256(prev_hash || canonical_json(payload))

The payload holds only protocol metadata - participants, dropouts, the hash of the accepted
aggregate, the epsilon spent, the round outcome - never a model update, a borrower row or a
secret. Anyone with the log and its head hash can tell whether a record was altered, reordered
or removed from the middle, because every later hash depends on it.

The honest limits, which the threat model repeats: this is tamper-*evident*, not tamper-*proof*.
Whoever can write the file can rewrite the whole chain from any point and recompute the
following hashes; detecting that requires the head hash to be published somewhere the writer
cannot reach. Truncating the tail is invisible unless the expected head is known, which is why
:meth:`RoundAuditLog.verify` takes an ``expect_head`` argument and the CLI exposes it.

Writes take the repository's cross-process file lock and go through an atomic rewrite, so a
crash mid-append leaves the previous chain intact rather than a half-written line.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from federated_secure_aggregation.core.canonical import canonical_json, sha256_hex
from federated_secure_aggregation.core.fileio import atomic_write_text, file_lock

__all__ = ["AuditChainError", "RoundAuditLog", "RoundRecord"]

GENESIS = "0" * 64


class AuditChainError(ValueError):
    """The chain does not verify: a broken link, a bad index or an unexpected head."""


class RoundRecord(BaseModel):
    """One round's audit payload. Everything here is safe to publish."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    round_id: int = Field(ge=0)
    stage: str
    survivors: list[str]
    dropouts: list[str]
    excluded: list[str]
    accepted: bool
    aggregate_digest: str | None = None
    aggregate_norm: float
    norm_bound: float
    epsilon_spent: float
    holdout_auc: float | None = None
    bytes_received: int = Field(ge=0)
    duration_seconds: float = Field(ge=0)
    reason: str | None = None

    def payload(self) -> dict[str, JsonValue]:
        return dict(self.model_dump(mode="json"))


@dataclass(frozen=True, slots=True)
class ChainEntry:
    """A record as stored: its index, the previous hash and its own hash."""

    index: int
    prev_hash: str
    entry_hash: str
    record: RoundRecord


def _entry_hash(prev_hash: str, payload: Mapping[str, JsonValue]) -> str:
    return sha256_hex(prev_hash + canonical_json(dict(payload)))


class RoundAuditLog:
    """Append-only hash chain of :class:`RoundRecord` entries stored as JSON lines."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @property
    def head(self) -> str:
        """Hash of the last entry, or the all-zero genesis hash for an empty log."""
        entries = list(self.entries())
        return entries[-1].entry_hash if entries else GENESIS

    def append(self, record: RoundRecord) -> str:
        """Append ``record`` and return the new head hash."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with file_lock(self.path):
            existing = self.path.read_text(encoding="utf-8") if self.path.exists() else ""
            entries = list(_parse(existing))
            prev_hash = entries[-1].entry_hash if entries else GENESIS
            payload = record.payload()
            entry_hash = _entry_hash(prev_hash, payload)
            line = json.dumps(
                {
                    "index": len(entries),
                    "prev_hash": prev_hash,
                    "entry_hash": entry_hash,
                    "record": payload,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            atomic_write_text(self.path, existing + line + "\n")
        return entry_hash

    def entries(self) -> Iterator[ChainEntry]:
        if not self.path.exists():
            return iter(())
        return _parse(self.path.read_text(encoding="utf-8"))

    def verify(self, expect_head: str | None = None) -> int:
        """Re-hash the whole chain; return the number of entries or raise."""
        prev_hash = GENESIS
        count = 0
        for entry in self.entries():
            if entry.index != count:
                raise AuditChainError(f"entry {count} claims index {entry.index}")
            if entry.prev_hash != prev_hash:
                raise AuditChainError(f"entry {count} does not link to the previous entry")
            expected = _entry_hash(entry.prev_hash, entry.record.payload())
            if entry.entry_hash != expected:
                raise AuditChainError(f"entry {count} has been modified")
            prev_hash = entry.entry_hash
            count += 1
        if expect_head is not None and prev_hash != expect_head:
            raise AuditChainError(
                "chain head does not match the expected value (entries may be missing)"
            )
        return count


def _parse(text: str) -> Iterator[ChainEntry]:
    for number, line in enumerate(text.splitlines()):
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError as exc:
            raise AuditChainError(f"line {number + 1} is not valid JSON") from exc
        try:
            yield ChainEntry(
                index=int(data["index"]),
                prev_hash=str(data["prev_hash"]),
                entry_hash=str(data["entry_hash"]),
                record=RoundRecord.model_validate(data["record"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise AuditChainError(f"line {number + 1} is not a valid chain entry") from exc


def main(argv: list[str] | None = None) -> int:
    """``verify <path> [--expect-head HASH]`` for the CLI."""
    import argparse  # noqa: PLC0415 - only needed on the CLI path

    parser = argparse.ArgumentParser(prog="round-audit")
    sub = parser.add_subparsers(dest="command", required=True)
    verify = sub.add_parser("verify")
    verify.add_argument("path", type=Path)
    verify.add_argument("--expect-head", default=None)
    args = parser.parse_args(argv)

    log = RoundAuditLog(args.path)
    try:
        count = log.verify(args.expect_head)
    except AuditChainError as exc:
        print(json.dumps({"ok": False, "error": str(exc)}, indent=2))  # noqa: T201
        return 1
    print(json.dumps({"ok": True, "entries": count, "head": log.head}, indent=2))  # noqa: T201
    return 0
