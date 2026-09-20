"""One-shot empirical privacy auditing with random-direction canary clients.

The method follows Andrew, Kairouz, Oh, Oprea, McMahan and Suriyakumar, *One-shot Empirical
Privacy Estimation for Federated Learning* (2023). A set of ``k`` **canary clients** joins a
single ordinary training run. A canary holds no data: every round it submits the fixed vector
``clip_norm * u_j`` for its own random unit direction ``u_j``, drawn independently and therefore
nearly orthogonal to every other direction when the model dimension is much larger than ``k``.

After training, the total model movement ``theta_T - theta_0`` is compared, by cosine, against

- the ``k`` canary directions that *were* inserted (**observed**), and
- a large set of fresh random directions that were *not* (**unobserved**).

If the mechanism leaked nothing, the two cosine samples would be identically distributed. The
separation between them, measured in standard deviations of the unobserved sample,

    mu_hat = (mean(observed) - mean(unobserved)) / sd(unobserved)

is an estimate of the Gaussian-mechanism parameter ``mu`` of the whole training run, because the
composition of ``T`` Gaussian mechanisms with noise multiplier ``z`` is itself a Gaussian
mechanism with ``mu = sqrt(T) / z``. Converting ``mu_hat`` with the analytic Gaussian mechanism
of Balle and Wang (*Improving the Gaussian Mechanism for Differential Privacy*, ICML 2018) gives
an empirical epsilon.

What the number means, stated precisely because it is easy to overclaim:

- it is a **lower-bound estimate** obtained from one specific family of adversaries (random
  canary directions inserted in one run), not a proof of an upper bound;
- it is compared against the accountant's epsilon in the direction that catches bugs: if the
  *lower* end of the estimate's confidence interval exceeds the claimed epsilon, the claim is
  provably wrong (up to the confidence level) and CI fails;
- the reverse, an estimate comfortably below the claim, is consistent with the claim but is not
  evidence that the claim is tight. The RDP conversion is itself loose, so a correct
  implementation is expected to audit well below its claimed epsilon.

The confidence interval on ``mu_hat`` uses the normal approximation to the two sample means,
``se = sqrt(var(obs)/k + var(unobs)/m) / sd(unobs)``, which is adequate because ``k`` is in the
tens and ``m`` in the hundreds; the interval is reported alongside the point estimate so a
narrow-sample run cannot be mistaken for a precise one.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.stats import norm

__all__ = [
    "AuditResult",
    "CanaryBank",
    "audit_epsilon",
    "epsilon_from_mu",
    "gaussian_mechanism_delta",
]

FloatArray = npt.NDArray[np.float64]


def gaussian_mechanism_delta(mu: float, epsilon: float) -> float:
    """``delta`` of the ``mu``-Gaussian mechanism at ``epsilon`` (Balle and Wang 2018, Thm 8).

    delta(eps) = Phi(mu/2 - eps/mu) - e^eps * Phi(-mu/2 - eps/mu)
    """
    if mu <= 0.0:
        return 0.0
    first = float(norm.cdf(mu / 2.0 - epsilon / mu))
    second = float(norm.cdf(-mu / 2.0 - epsilon / mu))
    return max(0.0, first - math.exp(min(epsilon, 700.0)) * second)


def epsilon_from_mu(mu: float, delta: float, *, tolerance: float = 1e-9) -> float:
    """Invert :func:`gaussian_mechanism_delta`: the epsilon achieving ``delta`` for this ``mu``.

    ``delta`` is strictly decreasing in ``epsilon`` for a fixed ``mu``, so bisection converges.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if mu <= 0.0:
        return 0.0
    low, high = 0.0, 1.0
    while gaussian_mechanism_delta(mu, high) > delta:
        high *= 2.0
        if high > 1e6:  # pragma: no cover - unreachable for finite mu
            return high
    while high - low > tolerance:
        middle = 0.5 * (low + high)
        if gaussian_mechanism_delta(mu, middle) > delta:
            low = middle
        else:
            high = middle
    return high


@dataclass(frozen=True, slots=True)
class CanaryBank:
    """The canary directions inserted in a run, plus the reference directions that were not."""

    inserted: FloatArray
    reference: FloatArray

    @classmethod
    def create(cls, dimension: int, *, n_canaries: int, n_reference: int, seed: int) -> CanaryBank:
        """Draw unit directions uniformly on the sphere (normalized isotropic Gaussians)."""
        if dimension < 8:
            raise ValueError("canary auditing needs a model with at least 8 parameters")
        if n_canaries < 2 or n_reference < 2:
            raise ValueError("need at least two canary and two reference directions")
        if n_canaries >= dimension:
            raise ValueError(
                f"{n_canaries} canaries in {dimension} dimensions cannot be near-orthogonal; "
                "use a larger model or fewer canaries"
            )
        rng = np.random.default_rng(seed)
        inserted = _unit_rows(rng.normal(size=(n_canaries, dimension)))
        reference = _unit_rows(rng.normal(size=(n_reference, dimension)))
        return cls(inserted=inserted, reference=reference)

    @property
    def dimension(self) -> int:
        return int(self.inserted.shape[1])

    def direction(self, index: int) -> FloatArray:
        return np.asarray(self.inserted[index], dtype=np.float64)

    def cosines(self, model_delta: npt.ArrayLike) -> tuple[FloatArray, FloatArray]:
        """Cosines of ``model_delta`` with the inserted and the reference directions."""
        delta = np.asarray(model_delta, dtype=np.float64).ravel()
        if delta.size != self.dimension:
            raise ValueError(f"model delta has dimension {delta.size}, expected {self.dimension}")
        norm_delta = float(np.linalg.norm(delta))
        if norm_delta == 0.0:
            raise ValueError("the model did not move; there is nothing to audit")
        unit = delta / norm_delta
        return self.inserted @ unit, self.reference @ unit


def _unit_rows(matrix: FloatArray) -> FloatArray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return np.asarray(matrix / norms, dtype=np.float64)


@dataclass(frozen=True, slots=True)
class AuditResult:
    """The empirical estimate, its interval, and the verdict against a claimed epsilon."""

    mu: float
    mu_low: float
    mu_high: float
    epsilon: float
    epsilon_low: float
    epsilon_high: float
    delta: float
    n_canaries: int
    n_reference: int
    confidence: float
    claimed_epsilon: float

    @property
    def exceeds_claim(self) -> bool:
        """True when even the lower end of the interval is above the claimed epsilon."""
        return self.epsilon_low > self.claimed_epsilon

    @property
    def consistent_with_claim(self) -> bool:
        return not self.exceeds_claim

    def summary(self) -> dict[str, float | int | bool]:
        return {
            "mu": self.mu,
            "mu_low": self.mu_low,
            "mu_high": self.mu_high,
            "epsilon": self.epsilon,
            "epsilon_low": self.epsilon_low,
            "epsilon_high": self.epsilon_high,
            "claimed_epsilon": self.claimed_epsilon,
            "exceeds_claim": self.exceeds_claim,
            "n_canaries": self.n_canaries,
            "n_reference": self.n_reference,
        }


def audit_epsilon(
    observed: npt.ArrayLike,
    unobserved: npt.ArrayLike,
    *,
    delta: float,
    claimed_epsilon: float,
    confidence: float = 0.95,
) -> AuditResult:
    """Turn two cosine samples into an epsilon estimate with a confidence interval."""
    obs = np.asarray(observed, dtype=np.float64).ravel()
    unobs = np.asarray(unobserved, dtype=np.float64).ravel()
    if obs.size < 2 or unobs.size < 2:
        raise ValueError("both cosine samples need at least two values")
    if not 0.5 < confidence < 1.0:
        raise ValueError("confidence must be in (0.5, 1)")
    if claimed_epsilon <= 0:
        raise ValueError("claimed_epsilon must be > 0")

    scale = float(np.std(unobs, ddof=1))
    if scale == 0.0:
        raise ValueError("the reference cosines have zero spread; the audit is degenerate")
    mu = float(np.mean(obs) - np.mean(unobs)) / scale
    standard_error = (
        math.sqrt(float(np.var(obs, ddof=1)) / obs.size + float(np.var(unobs, ddof=1)) / unobs.size)
        / scale
    )
    half_width = float(norm.ppf(0.5 + confidence / 2.0)) * standard_error
    mu_low = max(0.0, mu - half_width)
    mu_high = mu + half_width
    return AuditResult(
        mu=mu,
        mu_low=mu_low,
        mu_high=mu_high,
        epsilon=epsilon_from_mu(max(mu, 0.0), delta),
        epsilon_low=epsilon_from_mu(mu_low, delta),
        epsilon_high=epsilon_from_mu(mu_high, delta),
        delta=delta,
        n_canaries=int(obs.size),
        n_reference=int(unobs.size),
        confidence=confidence,
        claimed_epsilon=claimed_epsilon,
    )
