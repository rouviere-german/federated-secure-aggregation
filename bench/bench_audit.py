#!/usr/bin/env python3
"""Does the privacy audit have power, and does it have specificity?

Usage:
  uv run python bench/bench_audit.py           # run, write bench/results/audit.json
  uv run python bench/bench_audit.py --check   # also enforce bench/thresholds.json

A privacy audit that never flags anything is worthless, and one that flags everything is worse.
This bench measures both halves on the same pipeline, with the same canaries and seeds:

**Power.** Three implementations that genuinely void the claimed guarantee must be flagged, that
is, the lower end of the audited epsilon's confidence interval must sit above the epsilon the
accountant reports:

  - ``no_clipping``          the per-client L2 clip is skipped
  - ``weak_noise``           the noise multiplier is out by a factor of ten
  - ``average_sensitivity``  noise calibrated to ``C / survivors`` instead of ``C``

**Specificity.** Two implementations that do *not* void the guarantee must not be flagged:

  - ``correct``             the reference pipeline
  - ``clip_after_noise``    the aggregate norm bound applied to the *noised* sum

The second one is the interesting case. Clipping an already-private output is post-processing,
so the guarantee survives it; a cosine-based audit is invariant to the rescaling it performs and
correctly reports the same epsilon as ``correct``. It is included as a deliberate negative
control, and the threshold file asserts that it stays unflagged.

**Known ceiling.** The estimator normalizes by the spread of the reference cosines, which with
``k`` canaries in ``d`` dimensions cannot resolve a separation beyond roughly ``sqrt(d / k)``
(about 5.3 in this configuration). Leakage worse than that registers as "at least this bad", so
the reported epsilon for a badly broken variant is a lower bound and not a measurement of how
bad it really is. ``audit_ceiling_mu`` is reported so the table can be read with that in mind.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from federated_secure_aggregation.evaluation.report import (
    check_thresholds,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from federated_secure_aggregation.fl.audit_run import AuditRunConfig, run_privacy_audit
from federated_secure_aggregation.fl.experiment import build_experiment
from federated_secure_aggregation.privacy.noise import MechanismVariant

BENCH = "audit"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"

SEED = 20260920
VARIANTS = (
    MechanismVariant.CORRECT,
    MechanismVariant.NO_CLIPPING,
    MechanismVariant.WEAK_NOISE,
    MechanismVariant.AVERAGE_SENSITIVITY,
    MechanismVariant.CLIP_AFTER_NOISE,
)


def run() -> tuple[list[dict[str, float | str | bool]], dict[str, float]]:
    experiment = build_experiment(n_rows=6_000, n_silos=5, alpha=1.0, seed=SEED)
    config = AuditRunConfig(seed=SEED)
    rows: list[dict[str, float | str | bool]] = []
    metrics: dict[str, float] = {}
    broken_flagged = 0
    sound_flagged = 0
    for variant in VARIANTS:
        outcome = run_privacy_audit(experiment, config, variant)
        rows.append(outcome.row())
        name = str(variant)
        metrics[f"epsilon_low_{name}"] = round(outcome.result.epsilon_low, 4)
        metrics[f"mu_hat_{name}"] = round(outcome.result.mu, 4)
        metrics[f"flagged_{name}"] = float(outcome.flagged)
        if variant.is_broken:
            broken_flagged += int(outcome.flagged)
        else:
            sound_flagged += int(outcome.flagged)
        if variant is MechanismVariant.CORRECT:
            metrics["claimed_epsilon"] = round(outcome.accountant_epsilon, 4)
            metrics["theoretical_mu"] = round(outcome.theoretical_mu, 4)
            metrics["model_dimension"] = float(outcome.dimension)
            metrics["audit_ceiling_mu"] = round(math.sqrt(outcome.dimension / config.n_canaries), 4)
            metrics["mu_recovery_ratio"] = round(outcome.result.mu / outcome.theoretical_mu, 4)
    metrics["broken_variants_flagged"] = float(broken_flagged)
    metrics["sound_variants_flagged"] = float(sound_flagged)
    metrics["n_canaries"] = float(config.n_canaries)
    metrics["n_reference"] = float(config.n_reference)
    metrics["rounds"] = float(config.rounds)
    return rows, metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="power and specificity of the privacy audit")
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
            [
                "variant",
                "breaks DP?",
                "mu hat",
                "mu low",
                "eps hat",
                "eps low (95%)",
                "claimed eps",
                "flagged",
            ],
            [
                [
                    row["variant"],
                    "yes" if row["expected_broken"] else "no",
                    row["mu_hat"],
                    row["mu_low"],
                    row["epsilon_hat"],
                    row["epsilon_low"],
                    row["claimed_epsilon"],
                    "yes" if row["flagged"] else "no",
                ]
                for row in rows
            ],
        )
    )
    print(
        f"\ntheoretical mu = sqrt(T)/z = {metrics['theoretical_mu']}, "
        f"audit ceiling sqrt(d/k) = {metrics['audit_ceiling_mu']}, "
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
