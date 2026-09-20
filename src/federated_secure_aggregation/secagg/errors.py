"""Protocol errors. Every one is a ``ValueError``, so callers can catch the family at once."""

from __future__ import annotations

__all__ = [
    "AggregateRejectedError",
    "ProtocolStateError",
    "QuorumError",
    "SecAggError",
]


class SecAggError(ValueError):
    """Base class for secure-aggregation failures."""


class ProtocolStateError(SecAggError):
    """A stage was attempted out of order, twice, or without its prerequisites."""


class QuorumError(SecAggError):
    """Too few clients reached a stage for the round to continue safely."""


class AggregateRejectedError(SecAggError):
    """The unmasked aggregate failed the norm bound, so the round is rolled back."""

    def __init__(self, norm: float, bound: float) -> None:
        super().__init__(
            f"aggregate L2 norm {norm:.4f} exceeds the bound {bound:.4f}; round rolled back"
        )
        self.norm = norm
        self.bound = bound
