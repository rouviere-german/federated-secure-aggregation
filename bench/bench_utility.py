#!/usr/bin/env python3
"""What the federation buys, and what differential privacy costs, across non-IID regimes.

Usage:
  uv run python bench/bench_utility.py           # run, write bench/results/utility.json
  uv run python bench/bench_utility.py --check   # also enforce bench/thresholds.json

The grid is Dirichlet ``alpha`` in {0.1, 1, 10} (extreme, moderate and mild label skew) against
``epsilon`` in {1, 4, 8, infinity}. Each cell trains the same logistic model for the same number
of rounds with the same seeds; only the noise multiplier changes, calibrated by the in-repo RDP
accountant so that the whole run lands exactly on the target epsilon.

Two reference points accompany every row: the **centralized** model trained on the pooled data
(the ceiling the law usually forbids) and the **local-only** average (the floor each lender gets
alone). A federated number that beats neither is not worth having.

The rows for ``epsilon = infinity`` are also run through the *full secure-aggregation protocol*,
not only through plaintext summation, and the AUC gap between the two is reported. It should be
zero up to quantization: the protocol reconstructs the same sum, so any visible gap would be a
bug in the masking or the ring encoding rather than a property of the method.

The honest headline is in the epsilon columns: with this few silos, client-level central DP is
expensive. Each silo is one unit of privacy, so the noise needed to hide a silo is a large
fraction of the signal the silos jointly produce, and the table shows it rather than hiding it
behind a favourable configuration.
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
from federated_secure_aggregation.fl.baselines import centralized_baseline, local_only_baseline
from federated_secure_aggregation.fl.experiment import build_experiment
from federated_secure_aggregation.fl.local import LocalTrainingConfig
from federated_secure_aggregation.fl.simulation import (
    AggregationMode,
    FederatedConfig,
    FederatedRun,
)
from federated_secure_aggregation.privacy.accountant import calibrate_noise_multiplier

BENCH = "utility"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

ALPHAS = (0.1, 1.0, 10.0)
EPSILONS: tuple[float, ...] = (1.0, 4.0, 8.0)
ROUNDS = 20
SILOS = 10
THRESHOLD = 6
DELTA = 1e-5
CLIP_NORM = 1.0
SEED = 20260920
ROWS = 8_000


def _train(
    experiment: object, alpha: float, epsilon: float | None, mode: AggregationMode
) -> dict[str, float]:
    from federated_secure_aggregation.fl.experiment import Experiment  # noqa: PLC0415

    assert isinstance(experiment, Experiment)  # noqa: S101 - narrows the type for mypy
    model = experiment.logistic_model()
    local = LocalTrainingConfig(epochs=2, batch_size=64, learning_rate=0.5, l2=1e-4)
    if epsilon is None:
        noise_multiplier = 0.0
    else:
        noise_multiplier = calibrate_noise_multiplier(
            target_epsilon=epsilon, delta=DELTA, steps=ROUNDS, sample_rate=1.0
        )
    run = FederatedRun(
        model,
        experiment.participants(model, local),
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(
            rounds=ROUNDS,
            clip_norm=CLIP_NORM,
            noise_multiplier=noise_multiplier,
            target_epsilon=epsilon if epsilon is not None else 1.0,
            delta=DELTA,
            threshold=THRESHOLD,
            aggregation=mode,
            seed=SEED,
            enforce_budget=epsilon is not None,
        ),
    )
    summary = run.run()
    return {
        "alpha": alpha,
        "epsilon": float("inf") if epsilon is None else epsilon,
        "noise_multiplier": round(noise_multiplier, 4),
        "auc": round(summary.holdout_auc, 4),
        "log_loss": round(summary.holdout_log_loss, 4),
        "accepted_rounds": float(summary.accepted_rounds),
        "epsilon_spent": (float("inf") if epsilon is None else round(summary.epsilon_spent, 4)),
    }


def run() -> tuple[list[dict[str, float]], dict[str, float]]:
    rows: list[dict[str, float]] = []
    metrics: dict[str, float] = {}
    for alpha in ALPHAS:
        experiment = build_experiment(n_rows=ROWS, n_silos=SILOS, alpha=alpha, seed=SEED)
        model = experiment.logistic_model()
        pooled_features, pooled_labels = experiment.pooled()
        centralized = centralized_baseline(
            model,
            pooled_features,
            pooled_labels,
            experiment.holdout_features,
            experiment.holdout_labels,
        )
        local_only = local_only_baseline(
            model, experiment.silo_data(), experiment.holdout_features, experiment.holdout_labels
        )
        rows.append(
            {
                "alpha": alpha,
                "epsilon": float("nan"),
                "noise_multiplier": 0.0,
                "auc": round(centralized.auc, 4),
                "log_loss": round(centralized.log_loss, 4),
                "accepted_rounds": 0.0,
                "epsilon_spent": float("nan"),
                "kind": 0.0,  # 0 = centralized baseline
            }
        )
        rows.append(
            {
                "alpha": alpha,
                "epsilon": float("nan"),
                "noise_multiplier": 0.0,
                "auc": round(local_only.auc, 4),
                "log_loss": round(local_only.log_loss, 4),
                "accepted_rounds": 0.0,
                "epsilon_spent": float("nan"),
                "kind": 1.0,  # 1 = local-only baseline
            }
        )
        tag = _tag(alpha)
        metrics[f"centralized_auc_{tag}"] = centralized.auc
        metrics[f"local_only_auc_{tag}"] = local_only.auc

        plaintext_inf = _train(experiment, alpha, None, AggregationMode.PLAINTEXT)
        secure_inf = _train(experiment, alpha, None, AggregationMode.SECURE)
        metrics[f"secure_vs_plaintext_auc_gap_{tag}"] = round(
            abs(secure_inf["auc"] - plaintext_inf["auc"]), 6
        )
        metrics[f"federated_auc_eps_inf_{tag}"] = plaintext_inf["auc"]
        metrics[f"federated_minus_local_eps_inf_{tag}"] = round(
            plaintext_inf["auc"] - local_only.auc, 6
        )
        metrics[f"centralized_minus_federated_eps_inf_{tag}"] = round(
            centralized.auc - plaintext_inf["auc"], 6
        )
        rows.append({**plaintext_inf, "kind": 2.0})  # 2 = federated
        rows.append({**secure_inf, "kind": 3.0})  # 3 = federated through the protocol

        for epsilon in EPSILONS:
            result = _train(experiment, alpha, epsilon, AggregationMode.PLAINTEXT)
            rows.append({**result, "kind": 2.0})
            metrics[f"federated_auc_eps{int(epsilon)}_{tag}"] = result["auc"]
            metrics[f"epsilon_spent_eps{int(epsilon)}_{tag}"] = result["epsilon_spent"]
    metrics["max_secure_vs_plaintext_auc_gap"] = max(
        metrics[f"secure_vs_plaintext_auc_gap_{_tag(alpha)}"] for alpha in ALPHAS
    )
    return rows, metrics


def _tag(alpha: float) -> str:
    return f"alpha{str(alpha).replace('.', 'p')}"


_KINDS = {0.0: "centralized", 1.0: "local-only", 2.0: "federated", 3.0: "federated (protocol)"}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="utility of the federation and cost of DP")
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
            ["alpha", "model", "epsilon", "noise z", "AUC", "log loss"],
            [
                [
                    row["alpha"],
                    _KINDS[row["kind"]],
                    "-" if np.isnan(row["epsilon"]) else row["epsilon"],
                    row["noise_multiplier"],
                    row["auc"],
                    row["log_loss"],
                ]
                for row in rows
            ],
        )
    )
    print(f"\nwall seconds: {metrics['wall_seconds']}")
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
