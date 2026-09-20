#!/usr/bin/env python3
"""What secure aggregation costs and whether it is exact, measured across client counts.

Usage:
  uv run python bench/bench_secagg.py           # run, write bench/results/secagg.json
  uv run python bench/bench_secagg.py --check   # also enforce bench/thresholds.json

Three questions, answered by running the real protocol end to end (X25519, AES-CTR masks,
AES-GCM share wrapping, Ed25519 signatures, Shamir reconstruction):

1. **Exactness.** For each client count, with no dropouts and with the largest dropout set the
   threshold allows, is the unmasked aggregate bit-for-bit equal to the plaintext ring sum? The
   answer has to be yes every time; anything else is a correctness bug, not a tolerance.
2. **Overhead.** How many bytes does the coordinator receive per round, and how long does a
   round take, as the number of clients grows? Pairwise masking is quadratic in the client
   count, so this is where that shows up.
3. **Dropout cost.** How much slower is a round that has to reconstruct dropped clients' mask
   keys from Shamir shares?

The reported ``bytes_per_client`` makes the quadratic term visible: it grows with the client
count because each client's share bundle carries one ciphertext per peer.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from federated_secure_aggregation.crypto.keys import IdentityKeyPair
from federated_secure_aggregation.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from federated_secure_aggregation.secagg.client import SecAggClient
from federated_secure_aggregation.secagg.coordinator import RoundConfig, SecAggCoordinator
from federated_secure_aggregation.secagg.masking import plaintext_sum
from federated_secure_aggregation.secagg.messages import RoundStage
from federated_secure_aggregation.secagg.session import SimulationClock, run_round

BENCH = "secagg"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

DIMENSION = 512
CLIENT_COUNTS = (3, 5, 10, 20)
CLIP_NORM = 1.0


def _setup(
    n_clients: int, threshold: int
) -> tuple[SecAggCoordinator, dict[str, SecAggClient], SimulationClock]:
    ids = [f"silo_{index:02d}" for index in range(n_clients)]
    identities = {name: IdentityKeyPair.generate() for name in ids}
    public = {name: identity.public_bytes for name, identity in identities.items()}
    coordinator_identity = IdentityKeyPair.generate()
    clock = SimulationClock()
    coordinator = SecAggCoordinator(
        coordinator_identity,
        public,
        RoundConfig(
            dimension=DIMENSION,
            threshold=threshold,
            clip_norm=CLIP_NORM,
            max_clients=n_clients,
            stage_timeout_seconds=1.0,
        ),
        clock=clock,
    )
    clients = {
        name: SecAggClient(
            name,
            identities[name],
            peer_identity_keys=public,
            coordinator_identity_key=coordinator_identity.public_bytes,
        )
        for name in ids
    }
    return coordinator, clients, clock


def _updates(ids: list[str], seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {name: rng.normal(size=DIMENSION) * 0.02 for name in ids}


def measure(n_clients: int, dropouts: int) -> dict[str, float]:
    """Run one round and report exactness, wall time and bytes received."""
    threshold = max(2, n_clients // 2 + 1)
    coordinator, clients, clock = _setup(n_clients, threshold)
    ids = sorted(clients)
    updates = _updates(ids, seed=n_clients * 100 + dropouts)
    dropped = tuple(ids[:dropouts])

    start = time.perf_counter()
    result = run_round(
        coordinator, clients, updates, 0, clock=clock, drop_before_masked_input=dropped
    )
    wall = time.perf_counter() - start

    if result.stage is not RoundStage.COMPLETED or result.aggregate is None:
        raise SystemExit(f"round failed with {n_clients} clients and {dropouts} dropouts")
    survivors = [name for name in ids if name not in dropped]
    reference = coordinator.scheme.decode(
        plaintext_sum(coordinator.scheme.encode(updates[name]) for name in survivors),
        count=len(survivors),
    )
    exact = bool(np.array_equal(result.aggregate, reference))
    return {
        "clients": float(n_clients),
        "dropouts": float(dropouts),
        "threshold": float(threshold),
        "survivors": float(len(survivors)),
        "exact": float(exact),
        "seconds": round(wall, 4),
        "bytes": float(result.bytes_received),
        "bytes_per_client": round(result.bytes_received / n_clients, 1),
    }


def run() -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    for n_clients in CLIENT_COUNTS:
        threshold = max(2, n_clients // 2 + 1)
        rows.append(measure(n_clients, 0))
        rows.append(measure(n_clients, n_clients - threshold))
    metrics: dict[str, float] = {
        "exact_fraction": float(np.mean([row["exact"] for row in rows])),
        "max_round_seconds": max(row["seconds"] for row in rows),
        "seconds_5_clients": next(
            row["seconds"] for row in rows if row["clients"] == 5 and row["dropouts"] == 0
        ),
        "seconds_20_clients": next(
            row["seconds"] for row in rows if row["clients"] == 20 and row["dropouts"] == 0
        ),
        "bytes_per_client_5": next(
            row["bytes_per_client"] for row in rows if row["clients"] == 5 and row["dropouts"] == 0
        ),
        "bytes_per_client_20": next(
            row["bytes_per_client"] for row in rows if row["clients"] == 20 and row["dropouts"] == 0
        ),
    }
    dropout_rows = [row for row in rows if row["dropouts"] > 0]
    clean_rows = [row for row in rows if row["dropouts"] == 0]
    metrics["dropout_overhead_ratio"] = round(
        float(np.mean([row["seconds"] for row in dropout_rows]))
        / float(np.mean([row["seconds"] for row in clean_rows])),
        4,
    )
    return rows, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="secure-aggregation exactness and overhead")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows, metrics = run()
    metrics["wall_seconds"] = round(time.perf_counter() - started, 3)
    write_results(
        RESULTS, {"bench": BENCH, "machine": machine_info(), "metrics": metrics, "rows": rows}
    )
    print(
        markdown_table(
            ["clients", "threshold", "dropouts", "survivors", "exact", "seconds", "bytes/client"],
            [
                [
                    int(row["clients"]),
                    int(row["threshold"]),
                    int(row["dropouts"]),
                    int(row["survivors"]),
                    "yes" if row["exact"] else "NO",
                    row["seconds"],
                    row["bytes_per_client"],
                ]
                for row in rows
            ],
        )
    )
    if args.check:
        problems = check_thresholds(metrics, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
