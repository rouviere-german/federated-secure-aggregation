"""The offline demo: 20 rounds of secure federated training with dropouts and an attacker.

It runs entirely in this process, needs no network, no API key and no downloads, and prints a
JSON summary plus a per-round table. What it shows, in one run:

1. five silos with a non-IID Dirichlet split train a credit-default model through the full
   secure-aggregation protocol (X25519, pairwise masks, Shamir shares, Ed25519 signatures);
2. two silos are killed mid-run and the round still completes, because the survivors hand back
   shares of the dropped silos' mask keys;
3. one round contains a malicious silo whose update is scaled 40x: the coordinator cannot see
   which silo it was, but the aggregate norm bound rejects the round and rolls it back;
4. Gaussian noise is added to every accepted aggregate and the RDP ledger reports the epsilon
   spent after each round, refusing to continue past the target;
5. the round audit chain is written and verified, and the final model is registered in the
   content-addressed registry with its run manifest.

Everything is seeded, so the same command prints the same numbers on the same machine.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import JsonValue

from federated_secure_aggregation.evaluation.report import markdown_table
from federated_secure_aggregation.fl.baselines import centralized_baseline, local_only_baseline
from federated_secure_aggregation.fl.experiment import build_experiment
from federated_secure_aggregation.fl.local import LocalTrainingConfig
from federated_secure_aggregation.fl.participants import Participant, ScaledAttacker
from federated_secure_aggregation.fl.simulation import (
    AggregationMode,
    FederatedConfig,
    FederatedRun,
)
from federated_secure_aggregation.ml.registry import ModelRegistry
from federated_secure_aggregation.ml.serialization import dumps_npz
from federated_secure_aggregation.monitoring.chain import RoundAuditLog
from federated_secure_aggregation.monitoring.metrics import create_round_metrics
from federated_secure_aggregation.privacy.accountant import calibrate_noise_multiplier
from federated_secure_aggregation.secagg.quantize import clip_l2

__all__ = ["run_demo"]

FloatArray = npt.NDArray[np.float64]

ROUNDS = 20
SILOS = 5
THRESHOLD = 3
TARGET_EPSILON = 8.0
DELTA = 1e-5
CLIP_NORM = 1.0
ATTACK_ROUND = 12
ATTACK_SCALE = 40.0
DROPOUT_ROUND = 7
SEED = 20260920


def _reset_run_state(workdir: Path) -> None:
    """Clear the per-run state so the demo is idempotent.

    The budget ledger and the round audit chain are deliberately *persistent*: that is what makes
    the ledger refuse rounds after a restart and the chain tamper-evident across processes. A demo
    that inherits them is therefore a demo that reports nothing on its second run -- the ledger is
    already at the target epsilon, so every round is refused. Resetting here keeps the persistence
    honest (``tests/unit/test_ledger_and_noise.py`` covers reload-and-refuse) while making
    ``tasks.py demo`` reproducible when run twice in a row.

    The registry is *not* cleared: it is content-addressed, so re-registering the same model is a
    no-op and keeping it proves the digest is stable across runs.
    """
    for name in ("budget.json", "round_audit.jsonl"):
        for path in (workdir / name, workdir / f"{name}.lock"):
            path.unlink(missing_ok=True)


def run_demo(workdir: Path) -> dict[str, JsonValue]:
    """Run the demo and return a JSON-serializable summary."""
    workdir.mkdir(parents=True, exist_ok=True)
    _reset_run_state(workdir)
    experiment = build_experiment(n_rows=6_000, n_silos=SILOS, alpha=1.0, seed=SEED)
    model = experiment.logistic_model()
    local = LocalTrainingConfig(epochs=2, batch_size=64, learning_rate=0.5, l2=1e-4)

    noise_multiplier = calibrate_noise_multiplier(
        target_epsilon=TARGET_EPSILON, delta=DELTA, steps=ROUNDS, sample_rate=1.0
    )

    honest = experiment.participants(model, local)
    attacker_id = honest[-1].silo_id
    participants: list[Participant] = [*honest[:-1]]
    participants.append(
        _RoundScopedAttacker(
            silo_id=attacker_id,
            honest=honest[-1],
            attack_round=ATTACK_ROUND,
            scale=ATTACK_SCALE,
            clip_norm=CLIP_NORM,
        )
    )

    audit_log = RoundAuditLog(workdir / "round_audit.jsonl")
    metrics = create_round_metrics()
    run = FederatedRun(
        model,
        participants,
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(
            rounds=ROUNDS,
            clip_norm=CLIP_NORM,
            noise_multiplier=noise_multiplier,
            target_epsilon=TARGET_EPSILON,
            delta=DELTA,
            threshold=THRESHOLD,
            aggregation=AggregationMode.SECURE,
            seed=SEED,
        ),
        metrics=metrics,
        audit_log=audit_log,
        ledger_path=workdir / "budget.json",
    )
    dropped = tuple(silo.silo_id for silo in honest[:2])
    summary = run.run(dropouts={DROPOUT_ROUND: dropped})

    # Reference run: identical in every way except that no noise is added. It is the
    # "epsilon = infinity" row, and the gap to the run above is the price of the guarantee.
    reference = FederatedRun(
        model,
        experiment.participants(model, local),
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(
            rounds=ROUNDS,
            clip_norm=CLIP_NORM,
            noise_multiplier=0.0,
            threshold=THRESHOLD,
            aggregation=AggregationMode.SECURE,
            seed=SEED,
            enforce_budget=False,
        ),
    ).run()

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

    manifest = experiment.manifest(
        params={
            "rounds": ROUNDS,
            "threshold": THRESHOLD,
            "noise_multiplier": round(noise_multiplier, 6),
            "target_epsilon": TARGET_EPSILON,
            "delta": DELTA,
            "attack_round": ATTACK_ROUND,
            "dropout_round": DROPOUT_ROUND,
        },
        metrics={
            "holdout_auc": summary.holdout_auc,
            "holdout_log_loss": summary.holdout_log_loss,
            "epsilon_spent": summary.epsilon_spent,
        },
    )
    registry = ModelRegistry(workdir / "registry")
    version = registry.register(
        "credit-default-federated",
        {"params.npz": dumps_npz({"params": summary.final_params})},
        manifest,
        metrics={"holdout_auc": summary.holdout_auc, "epsilon_spent": summary.epsilon_spent},
    )
    registry.set_alias("credit-default-federated", "production", version.digest)

    rejected = [outcome for outcome in summary.rounds if not outcome.accepted]
    dropout_rounds = [outcome for outcome in summary.rounds if outcome.dropouts]
    table = markdown_table(
        ["round", "survivors", "dropouts", "accepted", "agg norm", "bound", "epsilon", "AUC"],
        [
            [
                outcome.round_id,
                len(outcome.survivors),
                len(outcome.dropouts),
                outcome.accepted,
                round(outcome.aggregate_norm, 3),
                round(outcome.norm_bound, 3),
                round(outcome.epsilon_spent, 4),
                "-" if outcome.holdout_auc is None else round(outcome.holdout_auc, 4),
            ]
            for outcome in summary.rounds
        ],
    )
    print(table)  # noqa: T201 - the demo is a CLI

    return {
        "rounds": len(summary.rounds),
        "accepted_rounds": summary.accepted_rounds,
        "rejected_rounds": summary.rejected_rounds,
        "rejected_round_ids": [outcome.round_id for outcome in rejected],
        "dropout_round_ids": [outcome.round_id for outcome in dropout_rounds],
        "dropped_silos": list(dropped),
        "survivors_in_dropout_round": (list(dropout_rounds[0].survivors) if dropout_rounds else []),
        "noise_multiplier": round(noise_multiplier, 4),
        "epsilon_spent": round(summary.epsilon_spent, 4),
        "target_epsilon": TARGET_EPSILON,
        "delta": DELTA,
        "federated_holdout_auc": round(summary.holdout_auc, 4),
        "federated_holdout_auc_no_dp": round(reference.holdout_auc, 4),
        "dp_utility_cost_auc": round(reference.holdout_auc - summary.holdout_auc, 4),
        "centralized_holdout_auc": round(centralized.auc, 4),
        "local_only_holdout_auc": round(local_only.auc, 4),
        "federated_holdout_log_loss": round(summary.holdout_log_loss, 4),
        "partition_fingerprint": experiment.partition.fingerprint[:16],
        "audit_chain_entries": audit_log.verify(summary.audit_head),
        "audit_chain_head": summary.audit_head,
        "model_version": version.digest[:16],
        "registry_roundtrip_ok": bool(
            np.allclose(
                summary.final_params,
                _restore(registry, version.digest),
            )
        ),
    }


def _restore(registry: ModelRegistry, digest: str) -> FloatArray:
    from federated_secure_aggregation.ml.serialization import loads_npz  # noqa: PLC0415

    version = registry.get("credit-default-federated", digest)
    return loads_npz(registry.load_files(version)["params.npz"])["params"]


class _RoundScopedAttacker:
    """A silo that behaves honestly except in one round, where it scales its update up.

    It reports itself as dishonest for the whole run, because the framework's "honest" flag
    decides whether the *reference client* does the clipping and a compromised silo never hands
    that decision back. In ordinary rounds it clips its own update to the bound, exactly as an
    attacker who wants to stay invisible until it strikes would.

    It lives in the demo rather than in the library because it is a scripted demonstration, not
    a threat model: the property tests use
    :class:`~federated_secure_aggregation.fl.participants.ScaledAttacker` directly.
    """

    def __init__(
        self,
        silo_id: str,
        honest: Participant,
        attack_round: int,
        scale: float,
        clip_norm: float,
    ) -> None:
        self.silo_id = silo_id
        self._honest = honest
        self._attacker = ScaledAttacker(silo_id=silo_id, inner=honest, scale=scale)  # type: ignore[arg-type]
        self._attack_round = attack_round
        self._clip_norm = clip_norm

    @property
    def participant_id(self) -> str:
        return self.silo_id

    @property
    def honest(self) -> bool:
        """Reported dishonest for the whole run: the coordinator cannot know which round."""
        return False

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        if round_id == self._attack_round:
            return self._attacker.contribute(global_params, round_id)
        clipped, _ = clip_l2(self._honest.contribute(global_params, round_id), self._clip_norm)
        return clipped


def main() -> int:  # pragma: no cover - exercised through the CLI
    print(json.dumps(run_demo(Path("var/demo")), indent=2, sort_keys=True))  # noqa: T201
    return 0
