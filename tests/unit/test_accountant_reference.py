"""Cross-check the in-repo accountant against Google's ``dp-accounting``.

The accountant is the one number this repository asks to be trusted without proof, so it is
checked against an independent implementation rather than against itself. ``dp-accounting`` is a
development dependency only: nothing in ``src/`` imports it, so the shipped package has no
privacy-accounting dependency and this test is what keeps the two in agreement.

Agreement is required to 1e-9 on the RDP curve and 1e-9 on the resulting epsilon, over a grid
that spans the noise multipliers, sampling rates and round counts the benches and the demo use.
"""

from __future__ import annotations

import numpy as np
import pytest

from federated_secure_aggregation.privacy.accountant import (
    DEFAULT_ORDERS,
    RDPAccountant,
    rdp_subsampled_gaussian,
)

pytestmark = pytest.mark.unit

CASES = [
    (1.0, 1.0, 20),
    (2.0, 1.0, 20),
    (2.8536, 1.0, 20),
    (4.0, 1.0, 30),
    (18.0916, 1.0, 20),
    (2.0, 0.5, 50),
    (0.7, 0.2, 100),
    (1.5, 0.05, 200),
]


@pytest.mark.parametrize(("multiplier", "rate", "steps"), CASES)
def test_rdp_curve_matches_dp_accounting(multiplier: float, rate: float, steps: int) -> None:
    dp_accounting = pytest.importorskip("dp_accounting")
    from dp_accounting import rdp as dp_rdp

    accountant = dp_rdp.RdpAccountant(orders=list(DEFAULT_ORDERS))
    accountant.compose(
        dp_accounting.PoissonSampledDpEvent(rate, dp_accounting.GaussianDpEvent(multiplier)),
        steps,
    )
    reference = np.asarray(accountant._rdp, dtype=np.float64)
    ours = (
        rdp_subsampled_gaussian(
            sample_rate=rate, noise_multiplier=multiplier, orders=DEFAULT_ORDERS
        )
        * steps
    )
    assert np.allclose(ours, reference, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(("multiplier", "rate", "steps"), CASES)
@pytest.mark.parametrize("delta", [1e-5, 1e-6])
def test_epsilon_matches_dp_accounting(
    multiplier: float, rate: float, steps: int, delta: float
) -> None:
    dp_accounting = pytest.importorskip("dp_accounting")
    from dp_accounting import rdp as dp_rdp

    reference = dp_rdp.RdpAccountant(orders=list(DEFAULT_ORDERS))
    reference.compose(
        dp_accounting.PoissonSampledDpEvent(rate, dp_accounting.GaussianDpEvent(multiplier)),
        steps,
    )
    ours = RDPAccountant(noise_multiplier=multiplier, sample_rate=rate, steps=steps)
    assert ours.epsilon(delta) == pytest.approx(reference.get_epsilon(delta), abs=1e-9)
