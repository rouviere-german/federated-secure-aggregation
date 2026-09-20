"""Pairwise masking with self-masks, and the server-side reconstruction that removes them.

This is the arithmetic core of the protocol (simplified from Bonawitz et al., *Practical Secure
Aggregation for Privacy-Preserving Machine Learning*, CCS 2017).

Every ordered pair of clients ``{i, j}`` in the round's pairing set derives one shared mask
``m_ij`` from an X25519 exchange between their per-round mask keys. The client whose id sorts
first adds it, the other subtracts it, so summing over a set in which *both* members survive
cancels the pair exactly. Each client additionally adds a **self-mask** ``b_i`` from a seed it
chose itself:

    y_i = x_i + b_i + sum_{j in U, j != i} sign(i, j) * m_ij   (mod 2^32)

The self-mask is what makes the protocol dropout-tolerant without leaking. At unmask time the
coordinator learns, for each client, exactly one of two things:

- for a **survivor**: its self-mask seed ``b_i`` (its pairwise masks stay secret, and they
  cancelled anyway);
- for a **dropout**: its private mask key, from which the pairwise masks it shared with
  survivors are rebuilt (its self-mask is never revealed, and it contributed no ``y``).

Learning both for one client would expose that client's own update, so
:func:`reconstruct_aggregate` refuses any request that asks for both, and a test pins that.

The masks are uniform over ``Z_2^32^d`` under the PRF assumption on AES-CTR, so each individual
``y_i`` is uniform and independent of ``x_i``; only the sum over a set with all pairs closed is
informative. ``tests/unit/test_transcript_privacy.py`` checks that empirically.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass

import numpy as np

from federated_secure_aggregation.crypto.keys import derive_key, shared_secret
from federated_secure_aggregation.crypto.prg import PRG_SEED_BYTES, prg_uint32
from federated_secure_aggregation.secagg.quantize import RingArray, add_mod, sub_mod

__all__ = [
    "MASK_INFO",
    "SEED_COMMITMENT_BYTES",
    "SEED_COMMITMENT_INFO",
    "SELF_MASK_INFO",
    "SHARE_WRAP_INFO",
    "MaskDeriver",
    "MaskingError",
    "mask_vector",
    "pair_sign",
    "plaintext_sum",
    "reconstruct_aggregate",
    "self_seed_commitment",
]

MASK_INFO = b"federated-secure-aggregation/pairwise-mask/v1"
SELF_MASK_INFO = b"federated-secure-aggregation/self-mask/v1"
SHARE_WRAP_INFO = b"federated-secure-aggregation/share-wrap/v1"
SEED_COMMITMENT_INFO = b"federated-secure-aggregation/self-mask-commitment/v1"
SEED_COMMITMENT_BYTES = 32


class MaskingError(ValueError):
    """Masks cannot be derived or removed as requested."""


def self_seed_commitment(seed: bytes, round_nonce: bytes) -> bytes:
    """A hiding, binding commitment to a self-mask seed, published in the key advertisement.

    Without this the coordinator has nothing to check a *reconstructed* self-mask seed against:
    the advertisement carries only public keys, so a survivor that hands over a corrupted seed
    share silently corrupts the sum, the aggregate norm bound then rejects the round, and a
    share-integrity failure is misreported as poisoning by an unidentifiable client. With it,
    the reconstruction is verifiable and the liar is attributable.

    HKDF keyed by the *seed* and salted with the round nonce, under its own ``info`` label so
    the commitment is domain-separated from the self-mask the same seed produces. Hiding
    because the seed is 32 uniformly random bytes and HKDF is a PRF under an unknown key;
    binding because finding a second seed with the same output is a collision. The nonce makes
    the commitment fresh per round, so two rounds of the same silo are unlinkable through it.
    """
    if len(seed) != PRG_SEED_BYTES:
        raise MaskingError(f"self-mask seed must be {PRG_SEED_BYTES} bytes")
    if len(round_nonce) < 16:
        raise MaskingError("round nonce must be at least 16 bytes")
    return derive_key(
        seed, salt=round_nonce, info=SEED_COMMITMENT_INFO, length=SEED_COMMITMENT_BYTES
    )


def pair_sign(own_id: str, peer_id: str) -> int:
    """``+1`` for the lexicographically smaller id of a pair, ``-1`` for the larger one."""
    if own_id == peer_id:
        raise MaskingError("a client has no mask with itself")
    return 1 if own_id < peer_id else -1


@dataclass(frozen=True, slots=True)
class MaskDeriver:
    """Derives every mask of one round. The round nonce is the HKDF salt, so no seed repeats."""

    round_nonce: bytes
    dimension: int

    def __post_init__(self) -> None:
        if len(self.round_nonce) < 16:
            raise MaskingError("round nonce must be at least 16 bytes")
        if self.dimension < 1:
            raise MaskingError("dimension must be >= 1")

    def pairwise_seed(self, own_mask_private: bytes, peer_mask_public: bytes) -> bytes:
        """HKDF of the X25519 shared secret, salted with the round nonce."""
        secret = shared_secret(own_mask_private, peer_mask_public)
        return derive_key(secret, salt=self.round_nonce, info=MASK_INFO, length=PRG_SEED_BYTES)

    def wrap_key(self, own_wrap_private: bytes, peer_wrap_public: bytes) -> bytes:
        """AEAD key protecting Shamir shares between two clients for this round."""
        secret = shared_secret(own_wrap_private, peer_wrap_public)
        return derive_key(
            secret, salt=self.round_nonce, info=SHARE_WRAP_INFO, length=PRG_SEED_BYTES
        )

    def pairwise_mask(self, own_mask_private: bytes, peer_mask_public: bytes) -> RingArray:
        """The unsigned mask shared by a pair. Both sides compute the same vector."""
        return prg_uint32(self.pairwise_seed(own_mask_private, peer_mask_public), self.dimension)

    def self_mask(self, seed: bytes) -> RingArray:
        """The mask a client adds from its own secret seed."""
        if len(seed) != PRG_SEED_BYTES:
            raise MaskingError(f"self-mask seed must be {PRG_SEED_BYTES} bytes")
        expanded = derive_key(
            seed, salt=self.round_nonce, info=SELF_MASK_INFO, length=PRG_SEED_BYTES
        )
        return prg_uint32(expanded, self.dimension)


def mask_vector(
    deriver: MaskDeriver,
    ring_update: RingArray,
    *,
    own_id: str,
    own_mask_private: bytes,
    self_mask_seed: bytes,
    peer_mask_public: Mapping[str, bytes],
) -> RingArray:
    """Apply the self-mask and every signed pairwise mask to one client's ring update."""
    if ring_update.shape != (deriver.dimension,):
        raise MaskingError(
            f"update must have shape ({deriver.dimension},), got {ring_update.shape}"
        )
    if own_id in peer_mask_public:
        raise MaskingError("peer_mask_public must not contain the client's own id")
    masked = add_mod(ring_update, deriver.self_mask(self_mask_seed))
    for peer_id in sorted(peer_mask_public):
        mask = deriver.pairwise_mask(own_mask_private, peer_mask_public[peer_id])
        masked = add_mod(masked, mask) if pair_sign(own_id, peer_id) > 0 else sub_mod(masked, mask)
    return masked


def reconstruct_aggregate(
    deriver: MaskDeriver,
    masked_inputs: Mapping[str, RingArray],
    *,
    mask_public_keys: Mapping[str, bytes],
    survivor_self_seeds: Mapping[str, bytes],
    dropout_mask_privates: Mapping[str, bytes],
) -> RingArray:
    """Sum the survivors' masked inputs and strip every mask that did not cancel.

    ``masked_inputs`` holds one vector per survivor. ``mask_public_keys`` must cover the whole
    pairing set (survivors and dropouts), because a survivor masked against all of it.
    """
    survivors = sorted(masked_inputs)
    dropouts = sorted(dropout_mask_privates)
    if not survivors:
        raise MaskingError("no masked inputs to aggregate")
    overlap = set(survivors) & set(dropouts)
    if overlap:
        raise MaskingError(
            "a client cannot be both a survivor and a dropout: " + ", ".join(sorted(overlap))
        )
    both = set(survivor_self_seeds) & set(dropout_mask_privates)
    if both:
        raise MaskingError(
            "refusing to remove both the self-mask and the pairwise masks of "
            + ", ".join(sorted(both))
            + ": that would expose an individual update"
        )
    missing_seed = set(survivors) - set(survivor_self_seeds)
    if missing_seed:
        raise MaskingError("missing self-mask seeds for " + ", ".join(sorted(missing_seed)))
    missing_key = (set(survivors) | set(dropouts)) - set(mask_public_keys)
    if missing_key:
        raise MaskingError("missing mask public keys for " + ", ".join(sorted(missing_key)))

    total = np.zeros(deriver.dimension, dtype=np.uint32)
    for client_id in survivors:
        vector = masked_inputs[client_id]
        if vector.shape != (deriver.dimension,):
            raise MaskingError(f"masked input of {client_id} has the wrong shape")
        total = add_mod(total, vector)
        total = sub_mod(total, deriver.self_mask(survivor_self_seeds[client_id]))

    for dropped_id in dropouts:
        private = dropout_mask_privates[dropped_id]
        for survivor_id in survivors:
            mask = deriver.pairwise_mask(private, mask_public_keys[survivor_id])
            # Undo the survivor's own contribution, whose sign is taken from its perspective.
            total = (
                sub_mod(total, mask)
                if pair_sign(survivor_id, dropped_id) > 0
                else add_mod(total, mask)
            )
    return total


def plaintext_sum(updates: Iterable[RingArray]) -> RingArray:
    """Reference sum used by the exactness tests: plain ring addition, no masks involved."""
    vectors = list(updates)
    if not vectors:
        raise MaskingError("no updates to sum")
    total = np.zeros_like(vectors[0], dtype=np.uint32)
    for vector in vectors:
        total = add_mod(total, vector)
    return total
