"""Rényi differential privacy accounting for the subsampled Gaussian mechanism.

One round applies the Gaussian mechanism to a sum whose per-client sensitivity is the clipping
norm ``C``, with noise standard deviation ``z * C``. ``z`` is the *noise multiplier*, so the
accounting depends only on ``z``, not on ``C``. When only a fraction ``q`` of the silos take
part in a round, the mechanism is the Poisson-subsampled Gaussian.

For integer orders ``alpha >= 2``, Mironov, Talwar and Zhang (*Rényi Differential Privacy of the
Sampled Gaussian Mechanism*, 2019, Section 3.3) give the closed form

    RDP(alpha) = 1/(alpha-1) * log( sum_{i=0}^{alpha} C(alpha,i) (1-q)^(alpha-i) q^i
                                    exp(i(i-1) / (2 z^2)) )

which this module evaluates in log space so the binomial coefficients and the exponentials
cannot overflow. ``q = 1`` collapses to the plain Gaussian mechanism's ``alpha / (2 z^2)``.

RDP composes by addition, so ``T`` rounds cost ``T`` times one round's curve. The conversion to
``(epsilon, delta)`` is the improved bound of Balle et al. (2020), also in Canonne, Kamath and
Steinke (2020):

    epsilon(alpha) = RDP(alpha) + log((alpha-1)/alpha) - (log(delta) + log(alpha)) / (alpha-1)

minimized over the order grid. Restricting the grid to integer orders is a deliberate
simplification: it is what makes the closed form above exact, and the resulting epsilon is an
upper bound, so a coarser grid is conservative rather than optimistic. The cost is measured in
``tests/unit/test_accountant_reference.py``, which compares every value against Google's
``dp-accounting`` over the same grid and requires agreement to 1e-9.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

__all__ = [
    "DEFAULT_ORDERS",
    "PrivacyAccountingError",
    "RDPAccountant",
    "calibrate_noise_multiplier",
    "compute_epsilon",
    "rdp_subsampled_gaussian",
]

FloatArray = npt.NDArray[np.float64]

# Integer orders from 2 to 256: dense where practical epsilons land, sparse in the tail.
DEFAULT_ORDERS: tuple[int, ...] = (
    *range(2, 33),
    *range(34, 65, 2),
    *range(68, 129, 4),
    *range(136, 257, 8),
)


class PrivacyAccountingError(ValueError):
    """The accountant was asked for something outside its validity domain."""


def _log_binomial(alpha: int, index: int) -> float:
    return math.lgamma(alpha + 1) - math.lgamma(index + 1) - math.lgamma(alpha - index + 1)


def rdp_subsampled_gaussian(
    *, sample_rate: float, noise_multiplier: float, orders: Sequence[int]
) -> FloatArray:
    """RDP of one subsampled Gaussian step, one value per integer order."""
    if not 0.0 < sample_rate <= 1.0:
        raise PrivacyAccountingError("sample_rate must be in (0, 1]")
    if noise_multiplier <= 0.0:
        raise PrivacyAccountingError("noise_multiplier must be > 0")
    if not orders:
        raise PrivacyAccountingError("at least one order is required")

    variance_term = 1.0 / (2.0 * noise_multiplier**2)
    values = np.empty(len(orders), dtype=np.float64)
    for position, alpha in enumerate(orders):
        if alpha < 2 or int(alpha) != alpha:
            raise PrivacyAccountingError(f"orders must be integers >= 2, got {alpha}")
        if sample_rate == 1.0:
            values[position] = alpha * variance_term
            continue
        log_terms = [
            _log_binomial(alpha, index)
            + (alpha - index) * math.log1p(-sample_rate)
            + index * math.log(sample_rate)
            + index * (index - 1) * variance_term
            for index in range(alpha + 1)
        ]
        values[position] = _logsumexp(log_terms) / (alpha - 1)
    return values


def _logsumexp(values: Sequence[float]) -> float:
    largest = max(values)
    if largest == -math.inf:  # pragma: no cover - guarded by the sample_rate domain
        return -math.inf
    return largest + math.log(sum(math.exp(value - largest) for value in values))


def compute_epsilon(rdp: FloatArray, orders: Sequence[int], delta: float) -> tuple[float, int]:
    """Convert a composed RDP curve to ``(epsilon, best order)`` for the given ``delta``."""
    if not 0.0 < delta < 1.0:
        raise PrivacyAccountingError("delta must be in (0, 1)")
    curve = np.asarray(rdp, dtype=np.float64)
    if curve.shape != (len(orders),):
        raise PrivacyAccountingError("rdp and orders must have the same length")
    if np.any(curve < 0) or not np.all(np.isfinite(curve)):
        raise PrivacyAccountingError("rdp values must be finite and non-negative")
    alphas = np.asarray(orders, dtype=np.float64)
    epsilons = curve + np.log1p(-1.0 / alphas) - (math.log(delta) + np.log(alphas)) / (alphas - 1.0)
    epsilons = np.maximum(epsilons, 0.0)
    best = int(np.argmin(epsilons))
    return float(epsilons[best]), int(orders[best])


@dataclass(slots=True)
class RDPAccountant:
    """Composes identical subsampled-Gaussian steps and reports ``(epsilon, delta)``.

    The accountant is pure bookkeeping: it has no opinion about whether a step may happen. That
    decision belongs to :class:`~federated_secure_aggregation.privacy.ledger.BudgetLedger`,
    which asks this class what a step *would* cost before allowing it.
    """

    noise_multiplier: float
    sample_rate: float = 1.0
    orders: tuple[int, ...] = DEFAULT_ORDERS
    steps: int = 0
    _per_step: FloatArray = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.steps < 0:
            raise PrivacyAccountingError("steps must be >= 0")
        self._per_step = rdp_subsampled_gaussian(
            sample_rate=self.sample_rate,
            noise_multiplier=self.noise_multiplier,
            orders=self.orders,
        )

    @property
    def rdp(self) -> FloatArray:
        """The composed RDP curve after ``steps`` steps."""
        return self._per_step * self.steps

    def step(self, count: int = 1) -> None:
        if count < 1:
            raise PrivacyAccountingError("count must be >= 1")
        self.steps += count

    def epsilon(self, delta: float, *, extra_steps: int = 0) -> float:
        """Epsilon after the steps taken so far plus ``extra_steps`` hypothetical ones."""
        if extra_steps < 0:
            raise PrivacyAccountingError("extra_steps must be >= 0")
        total = self.steps + extra_steps
        if total == 0:
            return 0.0
        epsilon, _ = compute_epsilon(self._per_step * total, self.orders, delta)
        return epsilon

    def best_order(self, delta: float) -> int:
        """The order that achieves the reported epsilon (useful for manifests and logs)."""
        if self.steps == 0:
            return self.orders[0]
        _, order = compute_epsilon(self.rdp, self.orders, delta)
        return order

    def parameters(self) -> dict[str, float | int]:
        """The accountant configuration, recorded per round for reproducibility."""
        return {
            "noise_multiplier": self.noise_multiplier,
            "sample_rate": self.sample_rate,
            "steps": self.steps,
            "n_orders": len(self.orders),
            "min_order": self.orders[0],
            "max_order": self.orders[-1],
        }


def calibrate_noise_multiplier(
    *,
    target_epsilon: float,
    delta: float,
    steps: int,
    sample_rate: float = 1.0,
    orders: Sequence[int] = DEFAULT_ORDERS,
    tolerance: float = 1e-4,
    max_multiplier: float = 512.0,
) -> float:
    """Smallest ``z`` (to ``tolerance``) whose ``steps``-fold composition stays under epsilon.

    Epsilon is strictly decreasing in ``z`` for a fixed number of steps, so a bisection on
    ``[lower, upper]`` converges. The search starts by doubling until an upper bound satisfies
    the target, which fails loudly instead of silently returning a multiplier that does not.
    """
    if target_epsilon <= 0:
        raise PrivacyAccountingError("target_epsilon must be > 0")
    if steps < 1:
        raise PrivacyAccountingError("steps must be >= 1")

    def epsilon_of(multiplier: float) -> float:
        rdp = rdp_subsampled_gaussian(
            sample_rate=sample_rate, noise_multiplier=multiplier, orders=orders
        )
        value, _ = compute_epsilon(rdp * steps, orders, delta)
        return value

    upper = 0.5
    while epsilon_of(upper) > target_epsilon:
        upper *= 2.0
        if upper > max_multiplier:
            raise PrivacyAccountingError(
                f"no noise multiplier below {max_multiplier} reaches epsilon {target_epsilon} "
                f"in {steps} steps at delta {delta}"
            )
    lower = upper / 2.0
    while upper - lower > tolerance:
        middle = 0.5 * (lower + upper)
        if epsilon_of(middle) > target_epsilon:
            lower = middle
        else:
            upper = middle
    return upper
