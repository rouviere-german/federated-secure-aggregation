"""Prometheus metrics for the aggregation rounds, on a caller-supplied registry.

Nothing here is a global: the registry is passed in, so tests and the service each get their
own and metric names cannot collide between them. Label cardinality is kept at zero on purpose -
per-client labels would turn the metrics endpoint into a participation log, which is exactly
the kind of side channel the protocol is meant to close.
"""

from __future__ import annotations

from dataclasses import dataclass

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

__all__ = ["RoundMetrics", "create_round_metrics"]


@dataclass(frozen=True, slots=True)
class RoundMetrics:
    registry: CollectorRegistry
    round_duration_seconds: Histogram
    rounds_total: Counter
    rounds_rejected_total: Counter
    rounds_failed_total: Counter
    dropouts_total: Counter
    participants: Gauge
    epsilon_spent: Gauge
    aggregate_norm: Gauge
    global_holdout_auc: Gauge
    bytes_per_round: Histogram

    def observe_round(
        self,
        *,
        duration_seconds: float,
        participants: int,
        dropouts: int,
        accepted: bool,
        failed: bool,
        epsilon_spent: float,
        aggregate_norm: float,
        holdout_auc: float | None,
        bytes_received: int,
    ) -> None:
        """Record one round. Called once per round, whatever the outcome."""
        self.round_duration_seconds.observe(max(duration_seconds, 0.0))
        self.rounds_total.inc()
        if failed:
            self.rounds_failed_total.inc()
        elif not accepted:
            self.rounds_rejected_total.inc()
        if dropouts:
            self.dropouts_total.inc(dropouts)
        self.participants.set(participants)
        self.epsilon_spent.set(epsilon_spent)
        self.aggregate_norm.set(aggregate_norm)
        if holdout_auc is not None:
            self.global_holdout_auc.set(holdout_auc)
        self.bytes_per_round.observe(max(bytes_received, 0))


def create_round_metrics(
    registry: CollectorRegistry | None = None, namespace: str = "fedsecagg"
) -> RoundMetrics:
    reg = registry if registry is not None else CollectorRegistry()
    return RoundMetrics(
        registry=reg,
        round_duration_seconds=Histogram(
            "round_duration_seconds",
            "Wall-clock duration of one aggregation round.",
            namespace=namespace,
            registry=reg,
            buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0),
        ),
        rounds_total=Counter(
            "rounds_total",
            "Rounds that reached a terminal stage.",
            namespace=namespace,
            registry=reg,
        ),
        rounds_rejected_total=Counter(
            "rounds_rejected_total",
            "Rounds rolled back because the aggregate exceeded the norm bound.",
            namespace=namespace,
            registry=reg,
        ),
        rounds_failed_total=Counter(
            "rounds_failed_total",
            "Rounds abandoned because a stage closed below the quorum threshold.",
            namespace=namespace,
            registry=reg,
        ),
        dropouts_total=Counter(
            "dropouts_total",
            "Clients that joined a round but did not submit a masked input.",
            namespace=namespace,
            registry=reg,
        ),
        participants=Gauge(
            "participants",
            "Clients whose update entered the last aggregate.",
            namespace=namespace,
            registry=reg,
        ),
        epsilon_spent=Gauge(
            "epsilon_spent",
            "Privacy budget spent so far, at the configured delta.",
            namespace=namespace,
            registry=reg,
        ),
        aggregate_norm=Gauge(
            "aggregate_norm",
            "L2 norm of the last unmasked aggregate, before noise.",
            namespace=namespace,
            registry=reg,
        ),
        global_holdout_auc=Gauge(
            "global_holdout_auc",
            "ROC AUC of the global model on the held-out population.",
            namespace=namespace,
            registry=reg,
        ),
        bytes_per_round=Histogram(
            "bytes_per_round",
            "Bytes of protocol messages the coordinator received in one round.",
            namespace=namespace,
            registry=reg,
            buckets=(1e4, 5e4, 1e5, 5e5, 1e6, 5e6, 1e7, 5e7),
        ),
    )
