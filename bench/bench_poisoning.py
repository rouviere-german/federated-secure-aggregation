#!/usr/bin/env python3
"""What the aggregate norm bound catches, and what it does not.

Usage:
  uv run python bench/bench_poisoning.py           # run, write bench/results/poisoning.json
  uv run python bench/bench_poisoning.py --check   # also enforce bench/thresholds.json

Secure aggregation hides individual updates from the coordinator, which is the point - and also
means the coordinator cannot inspect them for abuse. The only integrity check it can apply is on
the sum: reject and roll back any round whose unmasked aggregate has L2 norm above
``survivors * clip_norm``. This bench measures exactly how far that gets.

**Caught.** One malicious silo multiplies its update by ``scale``. For each scale the bench runs
several rounds through the real protocol and reports the fraction rejected. Two effects set the
detection floor: the honest sum already uses part of the norm budget, and the ring encoding caps
one client's contribution at ``max_clients * clip_norm`` per coordinate, so beyond a certain
scale the attack saturates rather than growing.

**Not caught.** A sign-flip attacker negates its honest update and rescales it to sit exactly on
the clipping bound. Every check passes - the vector is indistinguishable from a legitimate
update of a silo whose data points the other way - and the round is accepted. The bench reports
the AUC the model reaches with such an attacker present against the clean run. This is a
limitation of aggregate-level defence, not a bug, and the README and threat model say so:
per-client attribution would need zero-knowledge range proofs, which this repo does not build.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from federated_secure_aggregation.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from federated_secure_aggregation.fl.experiment import build_experiment
from federated_secure_aggregation.fl.local import LocalTrainingConfig
from federated_secure_aggregation.fl.participants import (
    Participant,
    ScaledAttacker,
    SignFlipAttacker,
)
from federated_secure_aggregation.fl.simulation import (
    AggregationMode,
    FederatedConfig,
    FederatedRun,
)

BENCH = "poisoning"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

SEED = 20260920
SILOS = 5
THRESHOLD = 3
CLIP_NORM = 1.0
ROUNDS = 10
SCALES = (1.5, 2.0, 3.0, 5.0, 10.0, 40.0)
SIGN_FLIP_COUNTS = (1, 2)


def _config(**overrides: object) -> FederatedConfig:
    base: dict[str, object] = {
        "rounds": ROUNDS,
        "clip_norm": CLIP_NORM,
        "noise_multiplier": 0.0,
        "threshold": THRESHOLD,
        "aggregation": AggregationMode.SECURE,
        "seed": SEED,
        "enforce_budget": False,
    }
    base.update(overrides)
    return FederatedConfig(**base)  # type: ignore[arg-type]


def _run(participants: list[Participant], experiment: object) -> tuple[float, float, list[float]]:
    from federated_secure_aggregation.fl.experiment import Experiment  # noqa: PLC0415

    assert isinstance(experiment, Experiment)  # noqa: S101 - narrows the type for mypy
    model = experiment.logistic_model()
    run = FederatedRun(
        model,
        participants,
        experiment.holdout_features,
        experiment.holdout_labels,
        _config(),
    )
    summary = run.run()
    rejected = summary.rejected_rounds / len(summary.rounds)
    return rejected, summary.holdout_auc, list(summary.aggregate_norms)


def run() -> tuple[list[dict[str, float | str]], dict[str, float]]:
    experiment = build_experiment(n_rows=6_000, n_silos=SILOS, alpha=1.0, seed=SEED)
    local = LocalTrainingConfig(epochs=2, batch_size=64, learning_rate=0.5, l2=1e-4)
    honest = experiment.participants(experiment.logistic_model(), local)

    clean_rejected, clean_auc, clean_norms = _run(list(honest), experiment)
    rows: list[dict[str, float | str]] = [
        {
            "attacker": "none",
            "scale": 0.0,
            "rejection_rate": round(clean_rejected, 4),
            "holdout_auc": round(clean_auc, 4),
            "mean_aggregate_norm": round(float(np.mean(clean_norms)), 4),
        }
    ]
    metrics: dict[str, float] = {
        "clean_rejection_rate": round(clean_rejected, 6),
        "clean_auc": round(clean_auc, 6),
        "clean_mean_aggregate_norm": round(float(np.mean(clean_norms)), 4),
        "norm_bound": float(SILOS) * CLIP_NORM,
    }

    for scale in SCALES:
        participants: list[Participant] = [*honest[:-1]]
        participants.append(
            ScaledAttacker(silo_id=honest[-1].silo_id, inner=honest[-1], scale=scale)
        )
        rejected, auc, norms = _run(participants, experiment)
        rows.append(
            {
                "attacker": "scaled",
                "scale": scale,
                "rejection_rate": round(rejected, 4),
                "holdout_auc": round(auc, 4),
                "mean_aggregate_norm": round(float(np.mean(norms)), 4),
            }
        )
        metrics[f"rejection_rate_scale{str(scale).replace('.', 'p')}"] = round(rejected, 6)

    for count in SIGN_FLIP_COUNTS:
        flip: list[Participant] = [*honest[:-count]]
        flip.extend(
            SignFlipAttacker(silo_id=silo.silo_id, inner=silo, clip_norm=CLIP_NORM)
            for silo in honest[-count:]
        )
        flip_rejected, flip_auc, flip_norms = _run(flip, experiment)
        rows.append(
            {
                "attacker": f"sign-flip x{count}",
                "scale": 1.0,
                "rejection_rate": round(flip_rejected, 4),
                "holdout_auc": round(flip_auc, 4),
                "mean_aggregate_norm": round(float(np.mean(flip_norms)), 4),
            }
        )
        metrics[f"sign_flip{count}_rejection_rate"] = round(flip_rejected, 6)
        metrics[f"sign_flip{count}_auc"] = round(flip_auc, 6)
        metrics[f"sign_flip{count}_auc_drop"] = round(clean_auc - flip_auc, 6)
    return rows, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="aggregate norm bound: reach and limits")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    started = time.perf_counter()
    rows, metrics = run()
    metrics["wall_seconds"] = round(time.perf_counter() - started, 2)
    write_results(
        RESULTS, {"bench": BENCH, "machine": machine_info(), "metrics": metrics, "rows": rows}
    )
    print(
        markdown_table(
            ["attacker", "scale", "rounds rejected", "holdout AUC", "mean aggregate norm"],
            [
                [
                    row["attacker"],
                    row["scale"],
                    row["rejection_rate"],
                    row["holdout_auc"],
                    row["mean_aggregate_norm"],
                ]
                for row in rows
            ],
        )
    )
    print(
        f"\nnorm bound = survivors * clip_norm = {metrics['norm_bound']}, "
        f"wall seconds: {metrics['wall_seconds']}"
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
