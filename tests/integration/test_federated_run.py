"""End-to-end federated runs: aggregation-mode equivalence, dropouts, poisoning and budgets.

The first test here is the one that licenses everything else: the secure protocol and the
plaintext path must produce the *same* global model from the same seed. Without it, the longer
privacy and utility experiments could not legitimately run on the faster path.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from federated_secure_aggregation.fl.experiment import Experiment, build_experiment
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
from federated_secure_aggregation.monitoring.chain import RoundAuditLog
from federated_secure_aggregation.monitoring.metrics import create_round_metrics
from federated_secure_aggregation.privacy.accountant import calibrate_noise_multiplier
from federated_secure_aggregation.secagg.messages import RoundStage
from federated_secure_aggregation.secagg.session import TranscriptRecord

pytestmark = pytest.mark.integration

LOCAL = LocalTrainingConfig(epochs=2, batch_size=64, learning_rate=0.5, l2=1e-4)


def _experiment(n_silos: int = 5, seed: int = 4242) -> Experiment:
    return build_experiment(n_rows=3_000, n_silos=n_silos, alpha=1.0, seed=seed)


def _run(
    experiment: Experiment,
    participants: list[Participant],
    **overrides: object,
) -> FederatedRun:
    model = experiment.logistic_model()
    base: dict[str, object] = {
        "rounds": 6,
        "clip_norm": 1.0,
        "noise_multiplier": 0.0,
        "threshold": 3,
        "aggregation": AggregationMode.SECURE,
        "seed": 4242,
        "enforce_budget": False,
    }
    base.update(overrides)
    return FederatedRun(
        model,
        participants,
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(**base),  # type: ignore[arg-type]
    )


def test_the_secure_and_plaintext_paths_produce_the_same_model() -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    secure = _run(experiment, list(experiment.participants(model, LOCAL))).run()
    plaintext = _run(
        experiment,
        list(experiment.participants(model, LOCAL)),
        aggregation=AggregationMode.PLAINTEXT,
    ).run()
    assert np.array_equal(secure.final_params, plaintext.final_params)
    assert secure.holdout_auc == plaintext.holdout_auc
    assert [outcome.aggregate_norm for outcome in secure.rounds] == [
        outcome.aggregate_norm for outcome in plaintext.rounds
    ]


def test_a_round_with_dropouts_still_completes_and_excludes_them() -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    run = _run(experiment, list(experiment.participants(model, LOCAL)))
    summary = run.run(dropouts={2: ["silo_00", "silo_01"]})
    dropped_round = summary.rounds[2]
    assert dropped_round.accepted
    assert set(dropped_round.dropouts) == {"silo_00", "silo_01"}
    assert set(dropped_round.survivors) == {"silo_02", "silo_03", "silo_04"}
    assert all(outcome.accepted for outcome in summary.rounds)


def test_a_scaled_attacker_is_rejected_and_the_model_is_rolled_back() -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    honest = experiment.participants(model, LOCAL)
    participants: list[Participant] = [*honest[:-1]]
    participants.append(ScaledAttacker(silo_id=honest[-1].silo_id, inner=honest[-1], scale=50.0))
    run = _run(experiment, participants, rounds=3)
    before = run.params.copy()
    outcome = run.run_one_round(0)
    assert outcome.stage is RoundStage.REJECTED
    assert not outcome.accepted
    assert outcome.aggregate_norm > outcome.norm_bound
    assert np.array_equal(run.params, before)  # rolled back exactly


def test_a_sign_flip_attacker_passes_every_check() -> None:
    """A documented limitation: an attacker inside the clip bound is invisible to the server."""
    experiment = _experiment()
    model = experiment.logistic_model()
    honest = experiment.participants(model, LOCAL)
    participants: list[Participant] = [*honest[:-1]]
    participants.append(
        SignFlipAttacker(silo_id=honest[-1].silo_id, inner=honest[-1], clip_norm=1.0)
    )
    summary = _run(experiment, participants).run()
    assert summary.rejected_rounds == 0
    clean = _run(experiment, list(honest)).run()
    assert summary.holdout_auc <= clean.holdout_auc


def test_the_budget_stops_the_run_before_the_target_is_exceeded(tmp_path: Path) -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    multiplier = calibrate_noise_multiplier(
        target_epsilon=2.0, delta=1e-5, steps=4, sample_rate=1.0
    )
    run = _run(
        experiment,
        list(experiment.participants(model, LOCAL)),
        rounds=50,
        noise_multiplier=multiplier,
        target_epsilon=2.0,
        enforce_budget=True,
        aggregation=AggregationMode.PLAINTEXT,
    )
    summary = run.run()
    assert summary.stopped_early
    assert summary.epsilon_spent <= 2.0
    assert len(summary.rounds) < 50
    assert run.ledger is not None
    assert not run.ledger.can_run_round()


def test_a_rejected_round_spends_no_privacy_budget(tmp_path: Path) -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    honest = experiment.participants(model, LOCAL)
    participants: list[Participant] = [*honest[:-1]]
    participants.append(ScaledAttacker(silo_id=honest[-1].silo_id, inner=honest[-1], scale=50.0))
    run = _run(
        experiment,
        participants,
        rounds=2,
        noise_multiplier=4.0,
        target_epsilon=50.0,
        enforce_budget=True,
        aggregation=AggregationMode.PLAINTEXT,
    )
    assert run.ledger is not None
    outcome = run.run_one_round(0)
    assert not outcome.accepted
    assert run.ledger.steps == 0


def test_metrics_registry_and_audit_chain_are_written(tmp_path: Path) -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    audit_log = RoundAuditLog(tmp_path / "rounds.jsonl")
    metrics = create_round_metrics()
    run = FederatedRun(
        model,
        list(experiment.participants(model, LOCAL)),
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(
            rounds=4,
            noise_multiplier=0.0,
            threshold=3,
            aggregation=AggregationMode.SECURE,
            seed=4242,
            enforce_budget=False,
        ),
        metrics=metrics,
        audit_log=audit_log,
    )
    summary = run.run(dropouts={1: ["silo_00"]})
    assert audit_log.verify(summary.audit_head) == 4
    records = [entry.record for entry in audit_log.entries()]
    assert records[1].dropouts == ["silo_00"]
    assert all(record.aggregate_digest is not None for record in records)
    assert all(record.holdout_auc is not None for record in records)

    from prometheus_client import generate_latest

    text = generate_latest(metrics.registry).decode("utf-8")
    assert "fedsecagg_rounds_total 4.0" in text
    assert "fedsecagg_dropouts_total 1.0" in text


def test_the_transcript_of_a_full_run_carries_only_masked_vectors() -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    transcript = TranscriptRecord()
    run = _run(experiment, list(experiment.participants(model, LOCAL)), rounds=3)
    run.run(transcript=transcript)
    assert transcript.message_json
    assert "borrower_id" not in "\n".join(transcript.message_json)
    assert "defaulted" not in "\n".join(transcript.message_json)


def test_federated_run_validates_its_participants() -> None:
    experiment = _experiment()
    model = experiment.logistic_model()
    participants = list(experiment.participants(model, LOCAL))
    with pytest.raises(ValueError, match="below the threshold"):
        FederatedRun(
            model,
            participants[:2],
            experiment.holdout_features,
            experiment.holdout_labels,
            FederatedConfig(threshold=3, enforce_budget=False),
        )
    with pytest.raises(ValueError, match="unique"):
        FederatedRun(
            model,
            [*participants, participants[0]],
            experiment.holdout_features,
            experiment.holdout_labels,
            FederatedConfig(threshold=3, enforce_budget=False),
        )


def test_federated_config_validation() -> None:
    with pytest.raises(ValueError, match="rounds"):
        FederatedConfig(rounds=0, enforce_budget=False)
    with pytest.raises(ValueError, match="clip_norm"):
        FederatedConfig(clip_norm=0.0, enforce_budget=False)
    with pytest.raises(ValueError, match="noise_multiplier must be >= 0"):
        FederatedConfig(noise_multiplier=-1.0, enforce_budget=False)
    with pytest.raises(ValueError, match="server_learning_rate"):
        FederatedConfig(server_learning_rate=0.0, enforce_budget=False)
    with pytest.raises(ValueError, match="threshold"):
        FederatedConfig(threshold=1, enforce_budget=False)
    with pytest.raises(ValueError, match="norm_bound_multiplier"):
        FederatedConfig(norm_bound_multiplier=0.5, enforce_budget=False)
    with pytest.raises(ValueError, match="budget can only be enforced"):
        FederatedConfig(noise_multiplier=0.0, enforce_budget=True)
