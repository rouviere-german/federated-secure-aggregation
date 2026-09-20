#!/usr/bin/env python3
"""Micro-benchmarks for the primitives every round runs, with CI regression thresholds.

Usage:
  uv run python bench/bench_baseline.py          # run, write bench/results/baseline.json
  uv run python bench/bench_baseline.py --check  # also exit 1 if bench/thresholds.json is exceeded

These are per-call latencies of the pieces that dominate a round: deriving one pairwise mask
(an X25519 exchange, an HKDF and an AES-CTR expansion), splitting and combining Shamir shares,
encoding and summing in the ring, signing and verifying a protocol message, and one step of the
RDP accountant. Thresholds are several times the numbers measured on the reference laptop,
because shared CI runners are noisy: they catch order-of-magnitude regressions, not small drifts.
"""

from __future__ import annotations

import argparse
import os
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np

from federated_secure_aggregation.core.ratelimit import RateLimiter
from federated_secure_aggregation.crypto import shamir
from federated_secure_aggregation.crypto.keys import IdentityKeyPair, KeyAgreementKeyPair
from federated_secure_aggregation.crypto.prg import prg_uint32
from federated_secure_aggregation.evaluation.report import (
    check_thresholds,
    flatten,
    load_thresholds,
    machine_info,
    markdown_table,
    write_results,
)
from federated_secure_aggregation.ml.drift import DriftMonitor
from federated_secure_aggregation.privacy.accountant import DEFAULT_ORDERS, RDPAccountant
from federated_secure_aggregation.secagg.masking import (
    MaskDeriver,
    mask_vector,
    self_seed_commitment,
)
from federated_secure_aggregation.secagg.messages import (
    KeyAdvertisement,
    MessageKind,
    sign_message,
    verify_message,
)
from federated_secure_aggregation.secagg.quantize import AggregationScheme, add_mod

BENCH = "baseline"
ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "bench" / "results" / f"{BENCH}.json"
THRESHOLDS = ROOT / "bench" / "thresholds.json"
DIMENSION = 1_024
NONCE = bytes(range(16))
SEED = bytes(range(32, 64))


def _timed(function: Callable[[], object], runs: int, warmup: int = 5) -> dict[str, float]:
    for _ in range(warmup):
        function()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        function()
        samples.append(time.perf_counter_ns() - start)
    micros = np.asarray(samples, dtype=np.float64) / 1_000.0
    return {
        "runs": float(runs),
        "p50_us": round(float(np.percentile(micros, 50)), 1),
        "p95_us": round(float(np.percentile(micros, 95)), 1),
    }


def bench_pairwise_mask() -> dict[str, float]:
    deriver = MaskDeriver(round_nonce=NONCE, dimension=DIMENSION)
    own = KeyAgreementKeyPair.generate()
    peer = KeyAgreementKeyPair.generate()
    return _timed(lambda: deriver.pairwise_mask(own.private_bytes, peer.public_bytes), runs=500)


def bench_prg_expand() -> dict[str, float]:
    seed = os.urandom(32)
    return _timed(lambda: prg_uint32(seed, DIMENSION), runs=2_000)


def bench_mask_vector_10_peers() -> dict[str, float]:
    deriver = MaskDeriver(round_nonce=NONCE, dimension=DIMENSION)
    own = KeyAgreementKeyPair.generate()
    peers = {
        f"silo_{index:02d}": KeyAgreementKeyPair.generate().public_bytes for index in range(10)
    }
    scheme = AggregationScheme.create(dimension=DIMENSION, clip_norm=1.0, max_clients=11)
    update = scheme.encode(np.random.default_rng(0).normal(size=DIMENSION) * 0.01)
    seed = os.urandom(32)
    return _timed(
        lambda: mask_vector(
            deriver,
            update,
            own_id="silo_99",
            own_mask_private=own.private_bytes,
            self_mask_seed=seed,
            peer_mask_public=peers,
        ),
        runs=100,
        warmup=2,
    )


def bench_shamir_split() -> dict[str, float]:
    secret = os.urandom(32)
    return _timed(lambda: shamir.split(secret, threshold=5, shares=10), runs=2_000)


def bench_shamir_combine() -> dict[str, float]:
    shares = shamir.split(os.urandom(32), threshold=5, shares=10)
    return _timed(lambda: shamir.combine(shares[:5]), runs=2_000)


def bench_ring_add() -> dict[str, float]:
    rng = np.random.default_rng(1)
    left = rng.integers(0, 2**32, size=DIMENSION, dtype=np.uint64).astype(np.uint32)
    right = rng.integers(0, 2**32, size=DIMENSION, dtype=np.uint64).astype(np.uint32)
    return _timed(lambda: add_mod(left, right), runs=5_000)


def bench_quantize_encode() -> dict[str, float]:
    scheme = AggregationScheme.create(dimension=DIMENSION, clip_norm=1.0, max_clients=10)
    vector = np.random.default_rng(2).normal(size=DIMENSION) * 0.01
    return _timed(lambda: scheme.encode(vector), runs=5_000)


def _advertisement() -> tuple[object, bytes]:
    identity = IdentityKeyPair.generate()
    keys = KeyAgreementKeyPair.generate()
    body = KeyAdvertisement(
        mask_public_key=keys.public_bytes,
        wrap_public_key=keys.public_bytes,
        self_seed_commitment=self_seed_commitment(SEED, NONCE),
    )
    message = sign_message(
        body,
        kind=MessageKind.KEY_ADVERTISEMENT,
        round_id=1,
        round_nonce=NONCE,
        sender="silo_00",
        identity=identity,
    )
    return message, identity.public_bytes


def bench_message_sign() -> dict[str, float]:
    identity = IdentityKeyPair.generate()
    keys = KeyAgreementKeyPair.generate()
    body = KeyAdvertisement(
        mask_public_key=keys.public_bytes,
        wrap_public_key=keys.public_bytes,
        self_seed_commitment=self_seed_commitment(SEED, NONCE),
    )
    return _timed(
        lambda: sign_message(
            body,
            kind=MessageKind.KEY_ADVERTISEMENT,
            round_id=1,
            round_nonce=NONCE,
            sender="silo_00",
            identity=identity,
        ),
        runs=2_000,
    )


def bench_message_verify() -> dict[str, float]:
    message, public = _advertisement()
    return _timed(
        lambda: verify_message(
            message,  # type: ignore[arg-type]
            public_key=public,
            expect_kind=MessageKind.KEY_ADVERTISEMENT,
            expect_round_id=1,
            expect_round_nonce=NONCE,
            expect_sender="silo_00",
        ),
        runs=2_000,
    )


def bench_rdp_step() -> dict[str, float]:
    accountant = RDPAccountant(noise_multiplier=2.0, sample_rate=0.5)
    accountant.step(20)
    return _timed(lambda: accountant.epsilon(1e-5), runs=2_000)


def bench_rdp_curve() -> dict[str, float]:
    from federated_secure_aggregation.privacy.accountant import (  # noqa: PLC0415
        rdp_subsampled_gaussian,
    )

    return _timed(
        lambda: rdp_subsampled_gaussian(
            sample_rate=0.5, noise_multiplier=2.0, orders=DEFAULT_ORDERS
        ),
        runs=50,
        warmup=2,
    )


def bench_drift() -> dict[str, float]:
    rng = np.random.default_rng(6)
    monitor = DriftMonitor({"aggregate_norm": rng.normal(size=10_000)})
    current = {"aggregate_norm": rng.normal(loc=0.1, size=10_000)}
    return _timed(lambda: monitor.check(current), runs=30, warmup=2)


def bench_rate_limiter() -> dict[str, float]:
    limiter = RateLimiter(rate=1e9, burst=1_000_000_000)
    return _timed(lambda: limiter.check("silo_00"), runs=20_000)


def run() -> dict[str, dict[str, float]]:
    return {
        "pairwise_mask_1k": bench_pairwise_mask(),
        "prg_expand_1k": bench_prg_expand(),
        "mask_vector_1k_10_peers": bench_mask_vector_10_peers(),
        "shamir_split_32b_5of10": bench_shamir_split(),
        "shamir_combine_32b_5of10": bench_shamir_combine(),
        "ring_add_1k": bench_ring_add(),
        "quantize_encode_1k": bench_quantize_encode(),
        "message_sign": bench_message_sign(),
        "message_verify": bench_message_verify(),
        "rdp_epsilon_20_steps": bench_rdp_step(),
        "rdp_curve_all_orders": bench_rdp_curve(),
        "drift_check_10k_vs_10k": bench_drift(),
        "rate_limiter_check": bench_rate_limiter(),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="micro-benchmarks for the protocol primitives")
    parser.add_argument("--check", action="store_true", help="enforce bench/thresholds.json")
    args = parser.parse_args(argv)

    results = run()
    flat = flatten(results)
    write_results(RESULTS, {"bench": BENCH, "machine": machine_info(), "metrics": flat})
    print(
        markdown_table(
            ["benchmark", "runs", "p50 (us)", "p95 (us)"],
            [[name, s["runs"], s["p50_us"], s["p95_us"]] for name, s in results.items()],
        )
    )
    if args.check:
        problems = check_thresholds(flat, load_thresholds(THRESHOLDS, BENCH))
        for problem in problems:
            print(f"REGRESSION: {problem}")
        if problems:
            return 1
        print("all benchmarks within thresholds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
