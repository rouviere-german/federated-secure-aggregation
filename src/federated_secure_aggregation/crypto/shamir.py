"""Shamir ``t``-of-``n`` secret sharing of byte strings over GF(2^8) (Shamir 1979).

Each byte of the secret gets its own degree ``t - 1`` polynomial whose constant term is that
byte; share ``i`` is the vector of polynomial evaluations at ``x = i``. Reconstruction is
Lagrange interpolation at ``x = 0``. Evaluation points start at 1, and ``t >= 2`` is enforced
in :func:`split`, so no share equals the secret; the points are the *share index*, not the
participant's identity, so the same code works whatever the participant ids look like.

Properties this guarantees, all exercised by tests:

- any ``t`` distinct shares reconstruct the secret exactly;
- any ``t - 1`` shares leave every secret equally likely (each byte's residual distribution is
  uniform), so fewer than ``t`` shares reveal nothing;
- shares are the same length as the secret.

Randomness comes from ``secrets.token_bytes`` unless an explicit generator is injected, which
only tests do.
"""

from __future__ import annotations

import secrets
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from federated_secure_aggregation.crypto.field import add, div, eval_poly, mul

__all__ = ["MAX_SHARES", "Share", "combine", "interpolate", "reconstruct_from", "split"]

MAX_SHARES = 255
RandomBytes = Callable[[int], bytes]


@dataclass(frozen=True, slots=True)
class Share:
    """One evaluation of the sharing polynomials. ``index`` is the GF(2^8) point ``x``."""

    index: int
    data: bytes

    def __post_init__(self) -> None:
        if not 1 <= self.index <= MAX_SHARES:
            raise ValueError(f"share index must be in 1..{MAX_SHARES}, got {self.index}")
        if not self.data:
            raise ValueError("share data must not be empty")


def split(
    secret: bytes,
    *,
    threshold: int,
    shares: int,
    random_bytes: RandomBytes = secrets.token_bytes,
) -> list[Share]:
    """Split ``secret`` into ``shares`` shares, any ``threshold`` of which reconstruct it."""
    if not secret:
        raise ValueError("secret must not be empty")
    if threshold == 1:
        # A degree-0 polynomial makes every share equal to the secret, which contradicts this
        # module's stated contract. RoundConfig refuses t < 2 for the same reason.
        raise ValueError("threshold must be >= 2; a 1-of-n scheme protects nothing")
    if not 2 <= threshold <= shares <= MAX_SHARES:
        raise ValueError(
            f"need 2 <= threshold <= shares <= {MAX_SHARES}, got threshold={threshold} "
            f"shares={shares}"
        )
    n_bytes = len(secret)
    coefficients = np.zeros((threshold, n_bytes), dtype=np.uint8)
    coefficients[0] = np.frombuffer(secret, dtype=np.uint8)
    if threshold > 1:
        noise = random_bytes((threshold - 1) * n_bytes)
        if len(noise) != (threshold - 1) * n_bytes:
            raise RuntimeError("random_bytes returned the wrong number of bytes")
        coefficients[1:] = np.frombuffer(noise, dtype=np.uint8).reshape(threshold - 1, n_bytes)
    points = np.arange(1, shares + 1, dtype=np.uint8)
    evaluations = eval_poly(coefficients, points)
    return [Share(int(x), bytes(row)) for x, row in zip(points, evaluations, strict=True)]


def combine(shares: Iterable[Share]) -> bytes:
    """Reconstruct the secret from ``t`` or more shares by interpolating at ``x = 0``.

    Supplying more than ``t`` shares is fine and still exact. Supplying fewer than ``t`` shares
    does not raise: it returns a value that is indistinguishable from a random secret. Callers
    must therefore enforce the threshold themselves and verify the result out of band (the
    protocol does this by checking reconstructed keys against their advertised public keys).
    """
    return interpolate(list(shares), 0)


def interpolate(shares: Iterable[Share], point: int) -> bytes:
    """Evaluate the interpolating polynomial through ``shares`` at ``x = point``.

    ``point = 0`` is the secret (that is what :func:`combine` asks for). Any other point is the
    share that a holder at that index *should* have produced, which is how the coordinator
    attributes a corrupted share once ``t`` good shares have pinned the polynomial down: with
    ``t`` correct evaluations the degree ``t - 1`` polynomial is unique, so every other
    evaluation is predicted exactly and a holder whose share differs lied.
    """
    collected: list[Share] = list(shares)
    if not collected:
        raise ValueError("at least one share is required")
    if not 0 <= point <= MAX_SHARES:
        raise ValueError(f"evaluation point must be in 0..{MAX_SHARES}, got {point}")
    indices = [share.index for share in collected]
    if len(set(indices)) != len(indices):
        raise ValueError("shares must have distinct indices")
    length = len(collected[0].data)
    if any(len(share.data) != length for share in collected):
        raise ValueError("all shares must have the same length")

    x = np.array(indices, dtype=np.uint8)
    y = np.stack([np.frombuffer(share.data, dtype=np.uint8) for share in collected])
    value = np.zeros(length, dtype=np.uint8)
    for position in range(len(collected)):
        others = np.delete(x, position)
        if others.size:
            # Lagrange basis at ``point``: prod_j (point - x_j) / (x_i - x_j); subtraction is
            # XOR in GF(2^8), so both differences are plain XORs.
            numerator = _product(add(others, np.uint8(point)))
            denominator = _product(add(others, x[position]))
            basis = np.uint8(div(numerator, denominator))
        else:
            basis = np.uint8(1)
        value = add(value, mul(y[position], basis))
    return bytes(value)


def _product(values: npt.NDArray[np.uint8]) -> np.uint8:
    result = np.uint8(1)
    for value in values:
        result = np.uint8(mul(result, value))
    return result


def reconstruct_from(shares: Sequence[Share], threshold: int) -> bytes:
    """Reconstruct only when at least ``threshold`` distinct shares are present."""
    if len(shares) < threshold:
        raise ValueError(f"need at least {threshold} shares, got {len(shares)}")
    return combine(shares[:threshold])
