"""Arithmetic in GF(2^8) with the AES reduction polynomial x^8 + x^4 + x^3 + x + 1 (0x11b).

Shamir sharing of byte strings is done byte-wise in this field, which keeps a share the same
length as the secret and makes every operation table-driven and constant-shaped. The log/exp
tables are built once at import from the generator 0x03, which is primitive for this field.

This module is deliberately small and self-contained: it is used only to split short secrets
(32-byte seeds and X25519 private keys), never to encrypt data. Table lookups are not
constant-time; see docs/THREAT_MODEL.md for why that is acceptable here and what it excludes.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

__all__ = [
    "EXP",
    "LOG",
    "ORDER",
    "add",
    "div",
    "eval_poly",
    "inv",
    "mul",
]

ORDER = 256
_MODULUS = 0x11B
_GENERATOR = 0x03


def _build_tables() -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8]]:
    cycle = np.zeros(ORDER - 1, dtype=np.uint8)
    log = np.zeros(ORDER, dtype=np.uint8)
    value = 1
    for power in range(ORDER - 1):
        cycle[power] = value
        log[value] = power
        # value *= GENERATOR in GF(2^8): 0x03 = x + 1, so multiply by x then add the original.
        doubled = value << 1
        if doubled & ORDER:
            doubled ^= _MODULUS
        value = doubled ^ value
    # Repeat the multiplicative cycle so a sum of two logarithms (at most 508) indexes directly.
    exp = np.concatenate([cycle, cycle])
    return exp, log


EXP, LOG = _build_tables()


def add(left: npt.ArrayLike, right: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Field addition, which is XOR. Also the field's subtraction."""
    return np.bitwise_xor(
        np.asarray(left, dtype=np.uint8), np.asarray(right, dtype=np.uint8)
    ).astype(np.uint8)


def mul(left: npt.ArrayLike, right: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Field multiplication via logarithms, with the zero cases masked out."""
    a = np.asarray(left, dtype=np.uint8)
    b = np.asarray(right, dtype=np.uint8)
    nonzero = (a != 0) & (b != 0)
    powers = LOG[a].astype(np.uint16) + LOG[b].astype(np.uint16)
    return np.where(nonzero, EXP[powers], 0).astype(np.uint8)


def inv(value: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Multiplicative inverse. ``0`` has no inverse and raises."""
    array = np.asarray(value, dtype=np.uint8)
    if np.any(array == 0):
        raise ZeroDivisionError("0 has no multiplicative inverse in GF(2^8)")
    return EXP[(ORDER - 1 - LOG[array].astype(np.uint16)) % (ORDER - 1)].astype(np.uint8)


def div(numerator: npt.ArrayLike, denominator: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Field division ``numerator / denominator``."""
    return mul(numerator, inv(denominator))


def eval_poly(coefficients: npt.NDArray[np.uint8], x: npt.ArrayLike) -> npt.NDArray[np.uint8]:
    """Horner evaluation of polynomials with coefficients in ascending degree order.

    ``coefficients`` has shape ``(degree + 1, n_bytes)``; ``x`` is a scalar or 1-D array of
    evaluation points. The result has shape ``(len(x), n_bytes)`` (or ``(n_bytes,)`` for a
    scalar ``x``).
    """
    if coefficients.ndim != 2:
        raise ValueError("coefficients must be a 2-D (degree + 1, n_bytes) array")
    points = np.atleast_1d(np.asarray(x, dtype=np.uint8))
    accumulator = np.zeros((points.size, coefficients.shape[1]), dtype=np.uint8)
    for coefficient in reversed(coefficients):
        accumulator = add(mul(accumulator, points[:, None]), coefficient[None, :])
    return accumulator if np.ndim(x) else accumulator[0]
