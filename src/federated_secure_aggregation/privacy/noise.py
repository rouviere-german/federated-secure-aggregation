"""The central differential-privacy mechanism, plus the deliberately broken variants.

The coordinator holds the only plaintext the protocol produces: the sum of the clipped updates.
It adds spherical Gaussian noise with standard deviation ``noise_multiplier * clip_norm`` to
that sum before dividing by the number of survivors. Because every client's update is clipped to
``clip_norm`` *before* it is quantized and masked, the L2 sensitivity of the sum to one client
is exactly ``clip_norm``, which is what makes the Gaussian mechanism applicable with that
standard deviation.

This is **trusted-coordinator central DP**, and calling it anything else would be wrong: the
coordinator sees the noiseless sum first. Secure aggregation stops it from seeing *individual*
updates, not from seeing the sum. Distributed or local DP, where the noise is added before the
coordinator sees anything, is a documented non-goal.

:class:`MechanismVariant` also defines three broken implementations. They exist so the auditor
can be shown to have power: a privacy audit that flags nothing on a broken implementation is not
evidence of anything. Each one is a bug that has appeared in real code:

- ``NO_CLIPPING`` - the per-client L2 clip is skipped, so the sensitivity is unbounded while the
  noise stays calibrated as if it were ``clip_norm``.
- ``WEAK_NOISE`` - the noise multiplier is out by a factor of ten (a mis-set constant).
- ``AVERAGE_SENSITIVITY`` - the noise is calibrated to ``clip_norm / survivors``, confusing the
  sensitivity of the *average* with that of the *sum*.

A fourth variant, ``CLIP_AFTER_NOISE``, applies the aggregate norm bound to the *noised* sum.
It is included precisely because it is **not** a privacy bug: clipping a differentially private
output is post-processing, so the guarantee survives. The audit is expected not to flag it, and
``bench/bench_audit.py`` checks that it does not, which is the specificity half of the audit's
evidence.
"""

from __future__ import annotations

from enum import StrEnum

import numpy as np
import numpy.typing as npt

__all__ = ["MechanismVariant", "add_central_noise", "clip_update", "noise_std"]

FloatArray = npt.NDArray[np.float64]


class MechanismVariant(StrEnum):
    """Which implementation of the DP pipeline to run (see the module docstring)."""

    CORRECT = "correct"
    NO_CLIPPING = "no_clipping"
    WEAK_NOISE = "weak_noise"
    AVERAGE_SENSITIVITY = "average_sensitivity"
    CLIP_AFTER_NOISE = "clip_after_noise"

    @property
    def is_broken(self) -> bool:
        """True for the variants that genuinely void the claimed guarantee."""
        return self in {
            MechanismVariant.NO_CLIPPING,
            MechanismVariant.WEAK_NOISE,
            MechanismVariant.AVERAGE_SENSITIVITY,
        }


def clip_update(update: npt.ArrayLike, clip_norm: float, variant: MechanismVariant) -> FloatArray:
    """Apply the client-side L2 clip, or skip it under ``NO_CLIPPING``."""
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    vector = np.asarray(update, dtype=np.float64).ravel()
    if not np.all(np.isfinite(vector)):
        raise ValueError("update contains NaN or infinite values")
    if variant is MechanismVariant.NO_CLIPPING:
        return vector.copy()
    norm = float(np.linalg.norm(vector))
    if norm <= clip_norm or norm == 0.0:
        return vector.copy()
    return vector * (clip_norm / norm)


def noise_std(
    *, noise_multiplier: float, clip_norm: float, survivors: int, variant: MechanismVariant
) -> float:
    """Standard deviation of the noise added to the *sum*, per coordinate."""
    if noise_multiplier < 0:
        raise ValueError("noise_multiplier must be >= 0")
    if clip_norm <= 0:
        raise ValueError("clip_norm must be > 0")
    if survivors < 1:
        raise ValueError("survivors must be >= 1")
    if variant is MechanismVariant.WEAK_NOISE:
        return noise_multiplier * clip_norm / 10.0
    if variant is MechanismVariant.AVERAGE_SENSITIVITY:
        return noise_multiplier * clip_norm / survivors
    return noise_multiplier * clip_norm


def add_central_noise(
    aggregate: npt.ArrayLike,
    rng: np.random.Generator,
    *,
    noise_multiplier: float,
    clip_norm: float,
    survivors: int,
    variant: MechanismVariant = MechanismVariant.CORRECT,
) -> FloatArray:
    """Add the calibrated Gaussian noise to the summed update.

    ``noise_multiplier = 0`` means no noise at all, which is the ``epsilon = infinity`` row of
    the utility table: useful as an upper bound on accuracy and never a privacy claim.
    """
    vector = np.asarray(aggregate, dtype=np.float64).ravel()
    sigma = noise_std(
        noise_multiplier=noise_multiplier,
        clip_norm=clip_norm,
        survivors=survivors,
        variant=variant,
    )
    noised = vector if sigma == 0.0 else vector + rng.normal(0.0, sigma, size=vector.size)
    if variant is MechanismVariant.CLIP_AFTER_NOISE:
        bound = survivors * clip_norm
        norm = float(np.linalg.norm(noised))
        if norm > bound and norm > 0.0:
            noised = noised * (bound / norm)
    return np.asarray(noised, dtype=np.float64)
