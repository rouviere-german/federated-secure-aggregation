"""The ring encoding: clipping, the no-wrap invariant and round-trip accuracy.

The invariant that matters is stated once in ``AggregationScheme.check_no_wrap`` and tested here
against random inputs: for any set of at most ``max_clients`` clipped updates, the exact integer
sum stays inside the signed half of ``Z_2^32``, so decoding the ring sum gives the real sum and
not a wrapped one.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.secagg.quantize import (
    MODULUS,
    SIGNED_MAX,
    AggregationScheme,
    QuantizationError,
    add_mod,
    clip_l2,
    dequantize,
    quantize,
    sub_mod,
)

pytestmark = pytest.mark.unit

DIMENSIONS = st.integers(min_value=1, max_value=64)
CLIENTS = st.integers(min_value=1, max_value=32)


def _vectors(dimension: int, count: int, seed: int, scale: float = 3.0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(scale=scale, size=dimension) for _ in range(count)]


@given(dimension=DIMENSIONS, count=CLIENTS, seed=st.integers(0, 2**32 - 1))
@settings(max_examples=50)
def test_honest_sums_never_wrap(dimension: int, count: int, seed: int) -> None:
    scheme = AggregationScheme.create(dimension=dimension, clip_norm=1.0, max_clients=count)
    vectors = _vectors(dimension, count, seed)
    encoded = [scheme.encode(vector) for vector in vectors]
    ring = np.zeros(dimension, dtype=np.uint32)
    for item in encoded:
        ring = add_mod(ring, item)
    decoded = scheme.decode(ring, count=count)
    expected = sum(clip_l2(vector, 1.0)[0] for vector in vectors)
    assert np.allclose(decoded, expected, atol=count * scheme.resolution + 1e-9)


@given(dimension=DIMENSIONS, seed=st.integers(0, 2**32 - 1))
@settings(max_examples=50)
def test_round_trip_error_is_bounded_by_the_resolution(dimension: int, seed: int) -> None:
    scheme = AggregationScheme.create(dimension=dimension, clip_norm=2.0, max_clients=8)
    vector = clip_l2(_vectors(dimension, 1, seed)[0], 2.0)[0]
    decoded = scheme.decode(scheme.encode(vector))
    assert np.max(np.abs(decoded - vector)) <= scheme.resolution + 1e-12


@given(st.integers(0, MODULUS - 1), st.integers(0, MODULUS - 1))
def test_ring_arithmetic_matches_modular_integers(left: int, right: int) -> None:
    a = np.array([left], dtype=np.uint32)
    b = np.array([right], dtype=np.uint32)
    assert int(add_mod(a, b)[0]) == (left + right) % MODULUS
    assert int(sub_mod(a, b)[0]) == (left - right) % MODULUS


@given(st.integers(-SIGNED_MAX, SIGNED_MAX))
def test_signed_representation_round_trips(value: int) -> None:
    ring = quantize(np.array([float(value)]), 1.0)
    assert int(dequantize(ring, 1.0)[0]) == value


def test_clip_l2_only_shrinks_long_vectors() -> None:
    short = np.array([0.3, 0.4])
    clipped, norm = clip_l2(short, 1.0)
    assert np.array_equal(clipped, short)
    assert norm == pytest.approx(0.5)

    long = np.array([3.0, 4.0])
    clipped, norm = clip_l2(long, 1.0)
    assert float(np.linalg.norm(clipped)) == pytest.approx(1.0)
    assert norm == pytest.approx(5.0)

    zero, norm = clip_l2(np.zeros(3), 1.0)
    assert np.array_equal(zero, np.zeros(3))
    assert norm == 0.0


def test_clip_l2_rejects_bad_input() -> None:
    with pytest.raises(QuantizationError, match="clip_norm"):
        clip_l2([1.0], 0.0)
    with pytest.raises(QuantizationError, match="NaN"):
        clip_l2([float("nan")], 1.0)


def test_scheme_validates_its_parameters() -> None:
    with pytest.raises(QuantizationError, match="dimension"):
        AggregationScheme(dimension=0, clip_norm=1.0, max_clients=2, scale=1.0)
    with pytest.raises(QuantizationError, match="clip_norm"):
        AggregationScheme(dimension=1, clip_norm=0.0, max_clients=2, scale=1.0)
    with pytest.raises(QuantizationError, match="max_clients"):
        AggregationScheme(dimension=1, clip_norm=1.0, max_clients=0, scale=1.0)
    with pytest.raises(QuantizationError, match="scale"):
        AggregationScheme(dimension=1, clip_norm=1.0, max_clients=2, scale=0.0)
    with pytest.raises(QuantizationError, match="wrap"):
        AggregationScheme(dimension=1, clip_norm=1.0, max_clients=2, scale=2e9)
    with pytest.raises(QuantizationError, match="minimum resolution"):
        AggregationScheme.create(dimension=1, clip_norm=1e9, max_clients=1000)
    with pytest.raises(QuantizationError, match="clip_norm"):
        AggregationScheme.create(dimension=1, clip_norm=0.0, max_clients=2)


def test_encode_checks_the_dimension_and_decode_checks_the_count() -> None:
    scheme = AggregationScheme.create(dimension=4, clip_norm=1.0, max_clients=3)
    with pytest.raises(QuantizationError, match="dimension 4"):
        scheme.encode(np.zeros(5))
    with pytest.raises(QuantizationError, match="dimension 4"):
        scheme.encode_raw(np.zeros(5))
    with pytest.raises(QuantizationError, match="count"):
        scheme.decode(np.zeros(4, dtype=np.uint32), count=0)
    with pytest.raises(QuantizationError, match="max_clients"):
        scheme.decode(np.zeros(4, dtype=np.uint32), count=4)


def test_encode_raw_saturates_instead_of_wrapping() -> None:
    """A malicious client can overshoot the clip bound, but not past the ring's signed range."""
    scheme = AggregationScheme.create(dimension=3, clip_norm=1.0, max_clients=5)
    huge = np.array([1e6, -1e6, 0.0])
    decoded = dequantize(scheme.encode_raw(huge), scheme.scale)
    ceiling = SIGNED_MAX / scheme.scale
    assert decoded[0] == pytest.approx(ceiling)
    assert decoded[1] == pytest.approx(-ceiling)
    assert ceiling == pytest.approx(5.0, rel=1e-6)
    with pytest.raises(QuantizationError, match="NaN"):
        scheme.encode_raw(np.array([float("inf"), 0.0, 0.0]))


def test_quantize_rejects_values_that_do_not_fit() -> None:
    with pytest.raises(QuantizationError, match="signed half"):
        quantize(np.array([1e12]), 1000.0)
    with pytest.raises(QuantizationError, match="NaN"):
        quantize(np.array([float("nan")]), 1.0)
    with pytest.raises(QuantizationError, match="scale"):
        quantize(np.array([1.0]), 0.0)
    with pytest.raises(QuantizationError, match="scale"):
        dequantize(np.array([1], dtype=np.uint32), -1.0)
