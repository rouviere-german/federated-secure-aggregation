"""A seeded pseudo-random generator that expands a 32-byte seed into a vector over Z_2^32.

The generator is AES-256 in counter mode over an all-zero plaintext, which is the standard way
to turn a block cipher into a stream of pseudo-random bytes. Four bytes become one little-endian
``uint32``, so the output is a uniform sample from ``Z_2^32^d`` under the PRF assumption on AES.

Two clients that derive the same seed therefore derive bit-identical masks, which is exactly
what pairwise cancellation needs, and any party without the seed sees a vector that is
computationally indistinguishable from uniform. The all-zero nonce is safe here only because
every seed is used for exactly one (round, pair, purpose) triple: seeds come from
:func:`federated_secure_aggregation.crypto.keys.derive_key`, whose HKDF salt is the round nonce.
Re-using a seed across rounds would reuse a keystream, so
:class:`~federated_secure_aggregation.secagg.masking.MaskDeriver` binds the round nonce in.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

__all__ = ["PRG_SEED_BYTES", "prg_bytes", "prg_uint32"]

PRG_SEED_BYTES = 32
_BLOCK_BYTES = 16
_ZERO_NONCE = bytes(_BLOCK_BYTES)
_MAX_OUTPUT_BYTES = 1 << 28  # 256 MiB; far above any model this project trains.


def prg_bytes(seed: bytes, length: int) -> bytes:
    """Expand ``seed`` into ``length`` pseudo-random bytes with AES-256-CTR."""
    if len(seed) != PRG_SEED_BYTES:
        raise ValueError(f"seed must be exactly {PRG_SEED_BYTES} bytes, got {len(seed)}")
    if length < 0:
        raise ValueError("length must be >= 0")
    if length > _MAX_OUTPUT_BYTES:
        raise ValueError(f"refusing to expand more than {_MAX_OUTPUT_BYTES} bytes")
    if length == 0:
        return b""
    encryptor = Cipher(algorithms.AES(seed), modes.CTR(_ZERO_NONCE)).encryptor()
    return encryptor.update(bytes(length)) + encryptor.finalize()


def prg_uint32(seed: bytes, size: int) -> npt.NDArray[np.uint32]:
    """Expand ``seed`` into ``size`` little-endian ``uint32`` values."""
    if size < 0:
        raise ValueError("size must be >= 0")
    raw = prg_bytes(seed, size * 4)
    return np.frombuffer(raw, dtype="<u4").astype(np.uint32, copy=True)
