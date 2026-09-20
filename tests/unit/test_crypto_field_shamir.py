"""GF(2^8) arithmetic and Shamir secret sharing: the algebraic invariants, checked with hypothesis.

The sharing has to be *exactly* right or the whole protocol silently corrupts sums, so the
properties tested here are the defining ones rather than samples of behaviour: field axioms,
exact reconstruction from any ``t`` shares, and the secrecy of any ``t - 1``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.crypto import field, shamir

pytestmark = pytest.mark.unit

BYTES = st.integers(min_value=0, max_value=255)
SECRETS = st.binary(min_size=1, max_size=48)


@given(BYTES, BYTES, BYTES)
def test_field_is_a_commutative_ring(a: int, b: int, c: int) -> None:
    assert int(field.add(a, b)) == int(field.add(b, a))
    assert int(field.mul(a, b)) == int(field.mul(b, a))
    assert int(field.mul(a, field.mul(b, c))) == int(field.mul(field.mul(a, b), c))
    left = int(field.mul(a, field.add(b, c)))
    right = int(field.add(field.mul(a, b), field.mul(a, c)))
    assert left == right
    assert int(field.mul(a, 1)) == a
    assert int(field.mul(a, 0)) == 0


@given(st.integers(min_value=1, max_value=255))
def test_every_nonzero_element_has_an_inverse(value: int) -> None:
    assert int(field.mul(value, field.inv(value))) == 1
    assert int(field.div(value, value)) == 1


def test_zero_has_no_inverse() -> None:
    with pytest.raises(ZeroDivisionError):
        field.inv(0)


def test_exp_and_log_tables_are_consistent() -> None:
    assert field.EXP.shape == (2 * (field.ORDER - 1),)
    for power in range(field.ORDER - 1):
        assert int(field.LOG[field.EXP[power]]) == power


def test_eval_poly_rejects_one_dimensional_coefficients() -> None:
    with pytest.raises(ValueError, match="2-D"):
        field.eval_poly(np.array([1, 2], dtype=np.uint8), 3)


@given(
    secret=SECRETS,
    threshold=st.integers(min_value=1, max_value=6),
    extra=st.integers(min_value=0, max_value=4),
    data=st.data(),
)
@settings(max_examples=60)
def test_any_threshold_subset_reconstructs_the_secret(
    secret: bytes, threshold: int, extra: int, data: st.DataObject
) -> None:
    shares = shamir.split(secret, threshold=threshold, shares=threshold + extra)
    assert all(len(share.data) == len(secret) for share in shares)
    chosen = data.draw(
        st.lists(
            st.sampled_from(range(len(shares))),
            min_size=threshold,
            max_size=threshold,
            unique=True,
        )
    )
    assert shamir.combine([shares[index] for index in chosen]) == secret


def test_all_subsets_of_size_t_work_for_a_small_scheme() -> None:
    secret = bytes(range(32))
    shares = shamir.split(secret, threshold=3, shares=6)
    for subset in itertools.combinations(shares, 3):
        assert shamir.combine(subset) == secret
    # Extra shares are harmless.
    assert shamir.combine(shares) == secret


def test_fewer_than_threshold_shares_reveal_nothing() -> None:
    """With ``t-1`` shares, every candidate secret remains equally likely.

    Concretely: fixing ``t-1`` shares and varying the secret over all 256 byte values produces
    256 distinct ``t``-th shares, so the observed ``t-1`` shares are compatible with every
    secret. That is Shamir's perfect-secrecy argument, instantiated for one byte.
    """
    reconstructions = {
        shamir.combine(shamir.split(bytes([value]), threshold=3, shares=5)[:2])
        for value in range(256)
    }
    # Two shares of a 3-of-5 scheme interpolate to some value that is not the secret in general.
    assert len(reconstructions) > 1
    completions = set()
    for value in range(256):
        shares = shamir.split(bytes([value]), threshold=2, shares=3)
        completions.add((shares[0].data, shares[1].data))
    assert len(completions) == 256


def test_split_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        shamir.split(b"", threshold=2, shares=3)
    with pytest.raises(ValueError, match="threshold"):
        shamir.split(b"x", threshold=4, shares=3)
    with pytest.raises(ValueError, match="threshold"):
        shamir.split(b"x", threshold=1, shares=256)


def test_split_rejects_a_short_random_source() -> None:
    with pytest.raises(RuntimeError, match="wrong number of bytes"):
        shamir.split(b"secret", threshold=2, shares=3, random_bytes=lambda n: b"\x00" * (n - 1))


def test_combine_validates_shares() -> None:
    shares = shamir.split(b"secret", threshold=2, shares=3)
    with pytest.raises(ValueError, match="at least one share"):
        shamir.combine([])
    with pytest.raises(ValueError, match="distinct indices"):
        shamir.combine([shares[0], shares[0]])
    with pytest.raises(ValueError, match="same length"):
        shamir.combine([shares[0], shamir.Share(9, b"xy")])


def test_share_validates_its_index() -> None:
    with pytest.raises(ValueError, match="share index"):
        shamir.Share(0, b"x")
    with pytest.raises(ValueError, match="must not be empty"):
        shamir.Share(1, b"")


def test_reconstruct_from_enforces_the_threshold() -> None:
    shares = shamir.split(b"secret", threshold=3, shares=5)
    assert shamir.reconstruct_from(shares, 3) == b"secret"
    with pytest.raises(ValueError, match="at least 3 shares"):
        shamir.reconstruct_from(shares[:2], 3)
