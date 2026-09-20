"""Fixed-point encoding of model updates into the ring Z_2^32.

Masking only cancels inside a finite ring, so real-valued updates have to become integers first.
The encoding is:

1. **Clip** the update to L2 norm ``C``. This bounds one client's contribution, which is what
   both the aggregate norm check and the differential-privacy sensitivity rely on.
2. **Scale and round** each coordinate by ``scale``, chosen so that ``n * C * scale <= 2^31 - 1``
   for the configured maximum number of clients ``n``. After clipping, every coordinate obeys
   ``|v_k| <= C``, so the exact integer sum of ``n`` clients stays inside the signed half of the
   ring and never wraps. ``AggregationScheme.check_no_wrap`` states this invariant, and a
   hypothesis test checks it for random updates.
3. **Represent** the signed integer in ``Z_2^32`` two's-complement style, so ring addition
   matches integer addition of the signed values.

Dequantization divides by ``scale`` again. Round-trip error per coordinate is at most
``0.5 / scale``, which the tests pin down.

Quantization is *not* a privacy mechanism. It is the arithmetic that lets masks cancel; privacy
comes from the masks and the Gaussian noise added to the aggregate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = [
    "MODULUS",
    "SIGNED_MAX",
    "AggregationScheme",
    "QuantizationError",
    "add_mod",
    "clip_l2",
    "dequantize",
    "quantize",
    "sub_mod",
]

MODULUS = 1 << 32
SIGNED_MAX = (1 << 31) - 1

FloatArray = npt.NDArray[np.float64]
RingArray = npt.NDArray[np.uint32]


class QuantizationError(ValueError):
    """A vector cannot be encoded: wrong shape, non-finite values or an unsafe scale."""


def clip_l2(vector: npt.ArrayLike, clip_norm: float) -> tuple[FloatArray, float]:
    """Scale ``vector`` down to L2 norm ``clip_norm`` if it is longer. Returns the original norm.

    Vectors already within the bound are returned unchanged (not merely rescaled by 1.0), so
    clipping never perturbs well-behaved updates.
    """
    if clip_norm <= 0:
        raise QuantizationError("clip_norm must be > 0")
    array = np.asarray(vector, dtype=np.float64).ravel()
    if not np.all(np.isfinite(array)):
        raise QuantizationError("vector contains NaN or infinite values")
    norm = float(np.linalg.norm(array))
    if norm <= clip_norm or norm == 0.0:
        return array.copy(), norm
    return array * (clip_norm / norm), norm


@dataclass(frozen=True, slots=True)
class AggregationScheme:
    """Ring encoding parameters shared by every participant in a round.

    ``max_clients`` is the largest number of updates that may be summed before decoding. The
    scheme refuses to exist if the resulting scale would be smaller than 1, because then the
    quantization step would be coarser than the updates themselves.
    """

    dimension: int
    clip_norm: float
    max_clients: int
    scale: float

    def __post_init__(self) -> None:
        if self.dimension < 1:
            raise QuantizationError("dimension must be >= 1")
        if self.clip_norm <= 0:
            raise QuantizationError("clip_norm must be > 0")
        if self.max_clients < 1:
            raise QuantizationError("max_clients must be >= 1")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise QuantizationError("scale must be a positive finite number")
        self.check_no_wrap()

    @classmethod
    def create(
        cls, *, dimension: int, clip_norm: float, max_clients: int, min_scale: float = 1.0
    ) -> AggregationScheme:
        """Pick the largest scale for which ``max_clients`` updates provably cannot wrap."""
        if clip_norm <= 0 or max_clients < 1:
            raise QuantizationError("clip_norm must be > 0 and max_clients >= 1")
        scale = math.floor(SIGNED_MAX / (max_clients * clip_norm))
        if scale < min_scale:
            raise QuantizationError(
                f"clip_norm={clip_norm} with max_clients={max_clients} leaves scale={scale}, "
                f"below the minimum resolution {min_scale}"
            )
        return cls(
            dimension=dimension, clip_norm=clip_norm, max_clients=max_clients, scale=float(scale)
        )

    def check_no_wrap(self) -> None:
        """Assert the no-wrap invariant ``max_clients * clip_norm * scale <= 2^31 - 1``."""
        budget = self.max_clients * self.clip_norm * self.scale
        if budget > SIGNED_MAX:
            raise QuantizationError(
                f"scale {self.scale} allows an honest sum of {budget:.0f} > {SIGNED_MAX}, "
                "which would wrap around the ring"
            )

    @property
    def resolution(self) -> float:
        """Largest absolute error introduced per coordinate by rounding."""
        return 0.5 / self.scale

    def encode(self, vector: npt.ArrayLike) -> RingArray:
        """Clip, scale, round and map into ``Z_2^32``."""
        clipped, _ = clip_l2(vector, self.clip_norm)
        if clipped.size != self.dimension:
            raise QuantizationError(
                f"expected a vector of dimension {self.dimension}, got {clipped.size}"
            )
        return quantize(clipped, self.scale)

    def encode_raw(self, vector: npt.ArrayLike) -> RingArray:
        """Encode without the L2 clip, saturating each coordinate at the ring's signed range.

        This models a client that ignores the reference implementation and submits whatever it
        likes. It cannot submit *anything*: the ring itself bounds one client's contribution at
        ``max_clients * clip_norm`` per coordinate, because beyond that the value would wrap and
        the sum would decode to nonsense. So the worst a single malicious client can do is a
        vector of L2 norm ``max_clients * clip_norm * sqrt(dimension)`` - large enough to break
        the aggregate norm bound, which is exactly what the bound is there to catch.
        """
        array = np.asarray(vector, dtype=np.float64).ravel()
        if array.size != self.dimension:
            raise QuantizationError(
                f"expected a vector of dimension {self.dimension}, got {array.size}"
            )
        if not np.all(np.isfinite(array)):
            raise QuantizationError("vector contains NaN or infinite values")
        scaled = np.clip(np.rint(array * self.scale), -float(SIGNED_MAX), float(SIGNED_MAX))
        return np.mod(scaled.astype(np.int64), MODULUS).astype(np.uint32)

    def decode(self, ring: npt.ArrayLike, *, count: int = 1) -> FloatArray:
        """Map a ring vector back to reals. ``count`` is only used to validate the range."""
        if count < 1:
            raise QuantizationError("count must be >= 1")
        if count > self.max_clients:
            raise QuantizationError(
                f"count={count} exceeds max_clients={self.max_clients}; the sum may have wrapped"
            )
        return dequantize(ring, self.scale)


def quantize(vector: npt.ArrayLike, scale: float) -> RingArray:
    """Round ``vector * scale`` to integers and represent them in ``Z_2^32``."""
    array = np.asarray(vector, dtype=np.float64).ravel()
    if not np.all(np.isfinite(array)):
        raise QuantizationError("vector contains NaN or infinite values")
    if not math.isfinite(scale) or scale <= 0:
        raise QuantizationError("scale must be a positive finite number")
    scaled = np.rint(array * scale)
    if np.any(np.abs(scaled) > SIGNED_MAX):
        raise QuantizationError("scaled values do not fit in the signed half of Z_2^32")
    return np.mod(scaled.astype(np.int64), MODULUS).astype(np.uint32)


def dequantize(ring: npt.ArrayLike, scale: float) -> FloatArray:
    """Inverse of :func:`quantize`: interpret the ring element as a signed integer."""
    if not math.isfinite(scale) or scale <= 0:
        raise QuantizationError("scale must be a positive finite number")
    values = np.asarray(ring, dtype=np.uint32).astype(np.int64)
    signed = np.where(values > SIGNED_MAX, values - MODULUS, values)
    return signed.astype(np.float64) / scale


def add_mod(left: npt.ArrayLike, right: npt.ArrayLike) -> RingArray:
    """Addition in ``Z_2^32``. Done in ``uint64`` so no overflow warning can fire."""
    a = np.asarray(left, dtype=np.uint32).astype(np.uint64)
    b = np.asarray(right, dtype=np.uint32).astype(np.uint64)
    return ((a + b) % MODULUS).astype(np.uint32)


def sub_mod(left: npt.ArrayLike, right: npt.ArrayLike) -> RingArray:
    """Subtraction in ``Z_2^32``."""
    a = np.asarray(left, dtype=np.uint32).astype(np.uint64)
    b = np.asarray(right, dtype=np.uint32).astype(np.uint64)
    return ((a + MODULUS - b) % MODULUS).astype(np.uint32)
