"""Running the one-shot privacy audit: canary clients inside a single training run.

The audit is a *measurement of the differential-privacy mechanism*, not of the transport, so it
runs on the plaintext aggregation path. That is legitimate for two reasons made checkable
elsewhere: the secure path is proven to reconstruct the plaintext sum bit for bit
(``tests/unit/test_secagg_exactness.py``), and the two paths are proven to produce the same
global model from the same seed (``tests/integration/test_aggregation_modes.py``). Running the
audit through the protocol as well would multiply its cost without changing what it measures.

Two conditions are deliberately equalized across all variants so the comparison is fair:

- the aggregate **norm bound is relaxed** for every audit run. It is an integrity control; if it
  were tight, the ``NO_CLIPPING`` variant would simply have its rounds rejected and the audit
  would measure the norm bound instead of the privacy bug.
- every variant sees the same canaries, the same silos, the same seeds and the same number of
  rounds.

The claimed epsilon a variant is judged against is always the one the *accountant* would report
for the configured noise multiplier - that is, the number the system would publish. A broken
implementation publishes the same number while leaking more, which is exactly what the audit is
built to catch.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.fl.experiment import Experiment
from federated_secure_aggregation.fl.local import LocalTrainingConfig
from federated_secure_aggregation.fl.participants import CanaryParticipant, Participant
from federated_secure_aggregation.fl.simulation import (
    AggregationMode,
    FederatedConfig,
    FederatedRun,
)
from federated_secure_aggregation.privacy.accountant import RDPAccountant
from federated_secure_aggregation.privacy.audit import AuditResult, CanaryBank, audit_epsilon
from federated_secure_aggregation.privacy.noise import MechanismVariant

__all__ = ["AuditRunConfig", "AuditRunResult", "run_privacy_audit"]

FloatArray = npt.NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class AuditRunConfig:
    """How to run one audit. The defaults are what ``bench/bench_audit.py`` uses."""

    rounds: int = 30
    noise_multiplier: float = 4.0
    delta: float = 1e-5
    clip_norm: float = 1.0
    hidden: int = 96
    n_canaries: int = 48
    n_reference: int = 1_024
    canary_scale: float = 20.0
    server_learning_rate: float = 1.0
    local_epochs: int = 1
    confidence: float = 0.95
    seed: int = 20260920

    def __post_init__(self) -> None:
        if self.canary_scale < 1.0:
            raise ValueError("canary_scale must be >= 1")
        if self.n_canaries < 8:
            raise ValueError("at least 8 canaries are needed for a usable interval")


@dataclass(frozen=True, slots=True)
class AuditRunResult:
    """The audit outcome plus the context needed to read it."""

    variant: MechanismVariant
    result: AuditResult
    accountant_epsilon: float
    theoretical_mu: float
    dimension: int
    rounds_applied: int
    holdout_auc: float

    @property
    def flagged(self) -> bool:
        """The audit says this implementation leaks more than it claims."""
        return self.result.exceeds_claim

    def row(self) -> dict[str, float | str | bool]:
        return {
            "variant": str(self.variant),
            "mu_hat": round(self.result.mu, 4),
            "mu_low": round(self.result.mu_low, 4),
            "theoretical_mu": round(self.theoretical_mu, 4),
            "epsilon_hat": round(self.result.epsilon, 4),
            "epsilon_low": round(self.result.epsilon_low, 4),
            "claimed_epsilon": round(self.accountant_epsilon, 4),
            "flagged": self.flagged,
            "expected_broken": self.variant.is_broken,
            "holdout_auc": round(self.holdout_auc, 4),
        }


def run_privacy_audit(
    experiment: Experiment, config: AuditRunConfig, variant: MechanismVariant
) -> AuditRunResult:
    """Train once with canary clients present and estimate the run's epsilon from the result."""
    model = experiment.mlp_model(hidden=config.hidden)
    bank = CanaryBank.create(
        model.dimension,
        n_canaries=config.n_canaries,
        n_reference=config.n_reference,
        seed=config.seed,
    )
    local = LocalTrainingConfig(
        epochs=config.local_epochs, batch_size=128, learning_rate=0.3, l2=1e-4
    )
    participants: list[Participant] = list(experiment.participants(model, local))
    participants.extend(
        CanaryParticipant(
            canary_id=f"canary_{index:03d}",
            direction=bank.direction(index),
            clip_norm=config.clip_norm,
            scale=config.canary_scale,
        )
        for index in range(config.n_canaries)
    )

    accountant = RDPAccountant(noise_multiplier=config.noise_multiplier, sample_rate=1.0)
    accountant.step(config.rounds)
    claimed = accountant.epsilon(config.delta)

    run = FederatedRun(
        model,
        participants,
        experiment.holdout_features,
        experiment.holdout_labels,
        FederatedConfig(
            rounds=config.rounds,
            clip_norm=config.clip_norm,
            noise_multiplier=config.noise_multiplier,
            target_epsilon=claimed,
            delta=config.delta,
            threshold=3,
            server_learning_rate=config.server_learning_rate,
            aggregation=AggregationMode.PLAINTEXT,
            variant=variant,
            seed=config.seed,
            enforce_budget=False,
            norm_bound_multiplier=config.canary_scale + 1.0,
        ),
        initial_params=model.initial_params(seed=config.seed),
    )
    summary = run.run()
    observed, unobserved = bank.cosines(summary.model_delta)
    result = audit_epsilon(
        observed,
        unobserved,
        delta=config.delta,
        claimed_epsilon=claimed,
        confidence=config.confidence,
    )
    return AuditRunResult(
        variant=variant,
        result=result,
        accountant_epsilon=claimed,
        theoretical_mu=float(np.sqrt(summary.accepted_rounds) / config.noise_multiplier),
        dimension=model.dimension,
        rounds_applied=summary.accepted_rounds,
        holdout_auc=summary.holdout_auc,
    )
