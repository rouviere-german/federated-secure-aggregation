"""A persisted privacy-budget ledger that refuses the round that would break the budget.

An accountant that is only consulted after the fact is decoration. This ledger is asked *before*
each round whether the round may happen, and it answers by computing the epsilon the run would
reach **including** that round. If that exceeds the target, the round is refused and training
stops with the budget intact.

Persistence matters for the same reason: a ledger that lives in memory resets when the
coordinator restarts, which turns a fixed budget into an unbounded one. The state is written
atomically under a cross-process file lock, so two coordinator processes pointed at the same
ledger cannot both spend the last round.

What this is not: tamper-proof. Anyone who can write the ledger file can reset the spend. It is
an honest-operator control, listed as such in the threat model.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from federated_secure_aggregation.core.fileio import atomic_write_text, file_lock
from federated_secure_aggregation.privacy.accountant import RDPAccountant

__all__ = ["BudgetExhaustedError", "BudgetLedger", "BudgetState"]


class BudgetExhaustedError(RuntimeError):
    """The requested round would push the run past its (epsilon, delta) target."""

    def __init__(self, spent: float, would_be: float, target: float) -> None:
        super().__init__(
            f"refusing the round: epsilon would reach {would_be:.4f}, above the target "
            f"{target:.4f} (spent so far {spent:.4f})"
        )
        self.spent = spent
        self.would_be = would_be
        self.target = target


@dataclass(frozen=True, slots=True)
class BudgetState:
    """The serialized ledger: enough to rebuild the accountant exactly."""

    steps: int
    noise_multiplier: float
    sample_rate: float
    target_epsilon: float
    delta: float

    def to_json(self) -> str:
        return json.dumps(
            {
                "schema_version": 1,
                "steps": self.steps,
                "noise_multiplier": self.noise_multiplier,
                "sample_rate": self.sample_rate,
                "target_epsilon": self.target_epsilon,
                "delta": self.delta,
            },
            indent=2,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> BudgetState:
        data = json.loads(text)
        if data.get("schema_version") != 1:
            raise ValueError("unsupported ledger schema version")
        return cls(
            steps=int(data["steps"]),
            noise_multiplier=float(data["noise_multiplier"]),
            sample_rate=float(data["sample_rate"]),
            target_epsilon=float(data["target_epsilon"]),
            delta=float(data["delta"]),
        )


class BudgetLedger:
    """Guards a run's ``(epsilon, delta)`` target, optionally persisted to ``path``."""

    def __init__(
        self,
        *,
        noise_multiplier: float,
        target_epsilon: float,
        delta: float,
        sample_rate: float = 1.0,
        path: Path | None = None,
    ) -> None:
        if target_epsilon <= 0:
            raise ValueError("target_epsilon must be > 0")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.target_epsilon = target_epsilon
        self.delta = delta
        self.path = path
        self._accountant = RDPAccountant(noise_multiplier=noise_multiplier, sample_rate=sample_rate)
        if path is not None and path.exists():
            self._load()

    @property
    def steps(self) -> int:
        return self._accountant.steps

    @property
    def noise_multiplier(self) -> float:
        return self._accountant.noise_multiplier

    def epsilon_spent(self) -> float:
        return self._accountant.epsilon(self.delta)

    def epsilon_after_next(self) -> float:
        return self._accountant.epsilon(self.delta, extra_steps=1)

    def can_run_round(self) -> bool:
        """Whether one more round fits inside the target."""
        return self.epsilon_after_next() <= self.target_epsilon

    def charge_round(self) -> float:
        """Spend one round's budget and return the new epsilon, or refuse the round."""
        if not self.can_run_round():
            raise BudgetExhaustedError(
                self.epsilon_spent(), self.epsilon_after_next(), self.target_epsilon
            )
        if self.path is None:
            self._accountant.step()
            return self.epsilon_spent()
        with file_lock(self.path):
            if self.path.exists():
                self._load_unlocked()
                if not self.can_run_round():
                    raise BudgetExhaustedError(
                        self.epsilon_spent(), self.epsilon_after_next(), self.target_epsilon
                    )
            self._accountant.step()
            self._save_unlocked()
        return self.epsilon_spent()

    def remaining_rounds(self, limit: int = 10_000) -> int:
        """How many further rounds fit, capped at ``limit`` so the search terminates."""
        count = 0
        while (
            count < limit
            and self._accountant.epsilon(self.delta, extra_steps=count + 1) <= self.target_epsilon
        ):
            count += 1
        return count

    def state(self) -> BudgetState:
        return BudgetState(
            steps=self._accountant.steps,
            noise_multiplier=self._accountant.noise_multiplier,
            sample_rate=self._accountant.sample_rate,
            target_epsilon=self.target_epsilon,
            delta=self.delta,
        )

    def _load(self) -> None:
        if self.path is None:  # pragma: no cover - guarded by the caller
            return
        with file_lock(self.path):
            self._load_unlocked()

    def _load_unlocked(self) -> None:
        if self.path is None:
            raise RuntimeError("_load_unlocked called on an unpersisted ledger")
        state = BudgetState.from_json(self.path.read_text(encoding="utf-8"))
        if (
            abs(state.noise_multiplier - self._accountant.noise_multiplier) > 1e-12
            or abs(state.sample_rate - self._accountant.sample_rate) > 1e-12
        ):
            raise ValueError(
                "the stored ledger was written with different mechanism parameters; "
                "epsilon cannot be composed across them"
            )
        self._accountant = RDPAccountant(
            noise_multiplier=state.noise_multiplier,
            sample_rate=state.sample_rate,
            steps=state.steps,
        )

    def _save_unlocked(self) -> None:
        if self.path is None:
            raise RuntimeError("_save_unlocked called on an unpersisted ledger")
        atomic_write_text(self.path, self.state().to_json() + "\n")
