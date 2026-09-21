"""The four kinds of participant a round can contain.

All of them answer the same question - "given the current global parameters, what update do you
contribute?" - and the coordinator cannot tell them apart, because it never sees an individual
update. That is the whole point of the design and also its main limitation: a malicious silo is
only visible in the aggregate.

- :class:`SiloParticipant` trains on its own rows.
- :class:`CanaryParticipant` holds no data and submits a fixed random direction; it exists for
  the privacy audit.
- :class:`ScaledAttacker` submits an honest-looking direction multiplied by a large factor, the
  attack the aggregate norm bound is designed to catch.
- :class:`SignFlipAttacker` submits the negation of an honest update, scaled to sit exactly on
  the clipping bound. It respects every check the protocol makes, which is why it is reported as
  a limitation rather than a defended-against threat.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.fl.data import derive_seed
from federated_secure_aggregation.fl.local import LocalTrainingConfig, local_update
from federated_secure_aggregation.fl.models import FederatedModel

__all__ = [
    "CanaryParticipant",
    "Participant",
    "ScaledAttacker",
    "SignFlipAttacker",
    "SiloParticipant",
]

FloatArray = npt.NDArray[np.float64]


class Participant(Protocol):
    """Anything that can contribute an update to a round."""

    @property
    def participant_id(self) -> str: ...

    @property
    def honest(self) -> bool:
        """Whether this participant runs the reference client, including its L2 clipping.

        Nothing in the protocol can force a client to clip: clipping happens on the client,
        before masking, where only the client can see it. A participant that answers ``False``
        here models a client that has been modified, and the round has to survive it.
        """
        ...

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        """The raw (pre-clipping) update this participant wants added to the sum."""
        ...


@dataclass(frozen=True, slots=True)
class SiloParticipant:
    """An honest lender training on its own borrowers."""

    silo_id: str
    model: FederatedModel
    features: FloatArray
    labels: FloatArray
    config: LocalTrainingConfig
    seed: int

    @property
    def participant_id(self) -> str:
        return self.silo_id

    @property
    def honest(self) -> bool:
        return True

    @property
    def n_rows(self) -> int:
        return int(self.features.shape[0])

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        return local_update(
            self.model,
            global_params,
            self.features,
            self.labels,
            self.config,
            seed=derive_seed(self.seed, "local", self.silo_id, round_id),
        )


@dataclass(frozen=True, slots=True)
class CanaryParticipant:
    """A privacy-audit canary: no data, one fixed random direction, every round.

    ``scale`` is expressed in multiples of the clipping norm. A correct implementation clips the
    contribution back to the clipping norm; an implementation that forgot to clip lets the whole
    ``scale`` through, which is what the audit measures.
    """

    canary_id: str
    direction: FloatArray
    clip_norm: float
    scale: float = 1.0

    def __post_init__(self) -> None:
        if self.clip_norm <= 0 or self.scale <= 0:
            raise ValueError("clip_norm and scale must be > 0")
        norm = float(np.linalg.norm(self.direction))
        if not np.isclose(norm, 1.0, atol=1e-9):
            raise ValueError("canary direction must be a unit vector")

    @property
    def participant_id(self) -> str:
        return self.canary_id

    @property
    def honest(self) -> bool:
        """A canary is an honest client: a correct pipeline clips it back to ``clip_norm``."""
        return True

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        del global_params, round_id  # a canary ignores the model entirely
        return np.asarray(self.direction * (self.clip_norm * self.scale), dtype=np.float64)


@dataclass(frozen=True, slots=True)
class ScaledAttacker:
    """A malicious silo that multiplies its update by ``scale`` to dominate the aggregate."""

    silo_id: str
    inner: SiloParticipant
    scale: float

    @property
    def participant_id(self) -> str:
        return self.silo_id

    @property
    def honest(self) -> bool:
        return False

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        return self.inner.contribute(global_params, round_id) * self.scale


@dataclass(frozen=True, slots=True)
class SignFlipAttacker:
    """A malicious silo that pushes the model backwards while respecting the clip bound."""

    silo_id: str
    inner: SiloParticipant
    clip_norm: float

    @property
    def participant_id(self) -> str:
        return self.silo_id

    @property
    def honest(self) -> bool:
        return False

    def contribute(self, global_params: FloatArray, round_id: int) -> FloatArray:
        honest = self.inner.contribute(global_params, round_id)
        norm = float(np.linalg.norm(honest))
        if norm == 0.0:
            return honest
        return np.asarray(-honest * (self.clip_norm / norm), dtype=np.float64)
