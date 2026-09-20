"""The exactness property: the unmasked aggregate equals the plaintext ring sum, bit for bit.

This is the claim the whole repository rests on, so it is tested as a property over the space it
is claimed for: 3 to 20 clients, thresholds from 2 up, and every dropout count from zero up to
``n - t``. "Approximately equal" would not do - masks either cancel exactly in the ring or the
protocol is broken - so the assertion is ``np.array_equal`` on the ring vectors.

The companion property is the negative one: with fewer than ``t`` survivors the coordinator must
refuse to reconstruct rather than return a wrong answer.
"""

from __future__ import annotations

import os

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from federated_secure_aggregation.crypto.keys import KeyAgreementKeyPair
from federated_secure_aggregation.crypto.prg import PRG_SEED_BYTES
from federated_secure_aggregation.secagg.masking import (
    MaskDeriver,
    MaskingError,
    mask_vector,
    pair_sign,
    plaintext_sum,
    reconstruct_aggregate,
)
from federated_secure_aggregation.secagg.quantize import AggregationScheme

pytestmark = pytest.mark.unit

DIMENSION = 24


def _party_ids(count: int) -> list[str]:
    return [f"silo_{index:02d}" for index in range(count)]


def _setup(
    count: int, seed: int
) -> tuple[
    MaskDeriver,
    AggregationScheme,
    dict[str, KeyAgreementKeyPair],
    dict[str, bytes],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    rng = np.random.default_rng(seed)
    ids = _party_ids(count)
    deriver = MaskDeriver(
        round_nonce=bytes(rng.integers(0, 256, 16, dtype=np.uint8)), dimension=DIMENSION
    )
    scheme = AggregationScheme.create(dimension=DIMENSION, clip_norm=1.0, max_clients=count)
    keys = {name: KeyAgreementKeyPair.generate() for name in ids}
    seeds = {name: bytes(rng.integers(0, 256, PRG_SEED_BYTES, dtype=np.uint8)) for name in ids}
    updates = {name: scheme.encode(rng.normal(size=DIMENSION) * 0.1) for name in ids}
    masked = {
        name: mask_vector(
            deriver,
            updates[name],
            own_id=name,
            own_mask_private=keys[name].private_bytes,
            self_mask_seed=seeds[name],
            peer_mask_public={other: keys[other].public_bytes for other in ids if other != name},
        )
        for name in ids
    }
    return deriver, scheme, keys, seeds, updates, masked


@given(
    count=st.integers(min_value=3, max_value=20),
    dropouts=st.integers(min_value=0, max_value=17),
    seed=st.integers(min_value=0, max_value=2**31 - 1),
)
@settings(max_examples=200, deadline=None, suppress_health_check=[HealthCheck.too_slow])
def test_unmasked_aggregate_equals_the_plaintext_sum(count: int, dropouts: int, seed: int) -> None:
    threshold = max(2, count // 2 + 1)
    dropouts = min(dropouts, count - threshold)
    deriver, _, keys, seeds, updates, masked = _setup(count, seed)
    ids = _party_ids(count)
    dropped = ids[:dropouts]
    survivors = ids[dropouts:]

    aggregate = reconstruct_aggregate(
        deriver,
        {name: masked[name] for name in survivors},
        mask_public_keys={name: keys[name].public_bytes for name in ids},
        survivor_self_seeds={name: seeds[name] for name in survivors},
        dropout_mask_privates={name: keys[name].private_bytes for name in dropped},
    )
    expected = plaintext_sum(updates[name] for name in survivors)
    assert np.array_equal(aggregate, expected)


def test_masks_of_a_full_set_cancel_without_any_reconstruction() -> None:
    deriver, _, keys, seeds, updates, masked = _setup(5, seed=1)
    ids = _party_ids(5)
    aggregate = reconstruct_aggregate(
        deriver,
        masked,
        mask_public_keys={name: keys[name].public_bytes for name in ids},
        survivor_self_seeds=seeds,
        dropout_mask_privates={},
    )
    assert np.array_equal(aggregate, plaintext_sum(updates.values()))


def test_reconstruction_refuses_to_reveal_both_secrets_of_one_client() -> None:
    """Self-mask seed *and* mask key for the same client would expose that client's update."""
    deriver, _, keys, seeds, _, masked = _setup(4, seed=2)
    ids = _party_ids(4)
    # silo_00 dropped out, so the coordinator may learn its mask key. Asking for its self-mask
    # seed as well would let it subtract every mask silo_00 applied and read its update.
    with pytest.raises(MaskingError, match="expose an individual update"):
        reconstruct_aggregate(
            deriver,
            {name: masked[name] for name in ids[1:]},
            mask_public_keys={name: keys[name].public_bytes for name in ids},
            survivor_self_seeds=seeds,
            dropout_mask_privates={ids[0]: keys[ids[0]].private_bytes},
        )


def test_reconstruction_validates_its_inputs() -> None:
    deriver, _, keys, seeds, _, masked = _setup(4, seed=3)
    ids = _party_ids(4)
    public = {name: keys[name].public_bytes for name in ids}

    with pytest.raises(MaskingError, match="no masked inputs"):
        reconstruct_aggregate(
            deriver, {}, mask_public_keys=public, survivor_self_seeds={}, dropout_mask_privates={}
        )
    with pytest.raises(MaskingError, match="both a survivor and a dropout"):
        reconstruct_aggregate(
            deriver,
            masked,
            mask_public_keys=public,
            survivor_self_seeds={},
            dropout_mask_privates={ids[0]: keys[ids[0]].private_bytes},
        )
    with pytest.raises(MaskingError, match="missing self-mask seeds"):
        reconstruct_aggregate(
            deriver,
            masked,
            mask_public_keys=public,
            survivor_self_seeds={name: seeds[name] for name in ids[:2]},
            dropout_mask_privates={},
        )
    with pytest.raises(MaskingError, match="missing mask public keys"):
        reconstruct_aggregate(
            deriver,
            masked,
            mask_public_keys={name: public[name] for name in ids[:2]},
            survivor_self_seeds=seeds,
            dropout_mask_privates={},
        )
    with pytest.raises(MaskingError, match="wrong shape"):
        reconstruct_aggregate(
            deriver,
            {**masked, ids[0]: np.zeros(3, dtype=np.uint32)},
            mask_public_keys=public,
            survivor_self_seeds=seeds,
            dropout_mask_privates={},
        )


def test_mask_vector_validates_its_inputs() -> None:
    deriver, scheme, keys, seeds, _, _ = _setup(3, seed=4)
    ids = _party_ids(3)
    update = scheme.encode(np.zeros(DIMENSION))
    with pytest.raises(MaskingError, match="shape"):
        mask_vector(
            deriver,
            np.zeros(3, dtype=np.uint32),
            own_id=ids[0],
            own_mask_private=keys[ids[0]].private_bytes,
            self_mask_seed=seeds[ids[0]],
            peer_mask_public={},
        )
    with pytest.raises(MaskingError, match="own id"):
        mask_vector(
            deriver,
            update,
            own_id=ids[0],
            own_mask_private=keys[ids[0]].private_bytes,
            self_mask_seed=seeds[ids[0]],
            peer_mask_public={ids[0]: keys[ids[0]].public_bytes},
        )
    with pytest.raises(MaskingError, match="seed must be"):
        mask_vector(
            deriver,
            update,
            own_id=ids[0],
            own_mask_private=keys[ids[0]].private_bytes,
            self_mask_seed=b"short",
            peer_mask_public={},
        )


def test_pair_sign_is_antisymmetric_and_rejects_self_pairs() -> None:
    assert pair_sign("a", "b") == 1
    assert pair_sign("b", "a") == -1
    with pytest.raises(MaskingError, match="no mask with itself"):
        pair_sign("a", "a")


def test_deriver_validates_its_parameters() -> None:
    with pytest.raises(MaskingError, match="round nonce"):
        MaskDeriver(round_nonce=b"short", dimension=4)
    with pytest.raises(MaskingError, match="dimension"):
        MaskDeriver(round_nonce=bytes(16), dimension=0)


def test_plaintext_sum_requires_at_least_one_vector() -> None:
    with pytest.raises(MaskingError, match="no updates"):
        plaintext_sum([])


def test_masks_are_identical_on_both_sides_of_a_pair() -> None:
    deriver = MaskDeriver(round_nonce=os.urandom(16), dimension=DIMENSION)
    alice = KeyAgreementKeyPair.generate()
    bob = KeyAgreementKeyPair.generate()
    assert np.array_equal(
        deriver.pairwise_mask(alice.private_bytes, bob.public_bytes),
        deriver.pairwise_mask(bob.private_bytes, alice.public_bytes),
    )
