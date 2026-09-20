"""What the coordinator's transcript actually contains - measured, not asserted by inspection.

The security claim is that a masked vector carries no information about the update inside it.
The tests here check the two observable consequences:

1. **Uniformity.** The masked words should be indistinguishable from uniform draws over
   ``Z_2^32``. A chi-square test over 256 buckets of the high byte, pooled across all clients
   and several rounds, must not reject uniformity.
2. **Independence.** The correlation between a client's masked words and its true quantized
   update should be at noise level. If masking leaked, an attacker could regress one on the
   other.

A third test inspects the wire payloads directly: no message the coordinator receives contains
the plaintext update bytes, and the only vector on the wire is the masked one.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import chisquare

from federated_secure_aggregation.secagg.session import TranscriptRecord, run_round
from tests.conftest import build_federation

pytestmark = pytest.mark.unit

DIMENSION = 256


def _collect(rounds: int = 6) -> tuple[TranscriptRecord, dict[str, np.ndarray], object]:
    federation = build_federation(n_clients=5, threshold=3, dimension=DIMENSION)
    transcript = TranscriptRecord()
    updates: dict[str, np.ndarray] = {}
    for round_id in range(rounds):
        updates = federation.updates(seed=round_id, scale=0.05)
        run_round(
            federation.coordinator,
            federation.clients,
            updates,
            round_id,
            clock=federation.clock,
            transcript=transcript,
        )
    return transcript, updates, federation.coordinator.scheme


def test_masked_words_look_uniform_over_the_ring() -> None:
    transcript, _, _ = _collect()
    words = np.concatenate(list(transcript.masked_vectors.values()))
    assert words.size >= 5 * DIMENSION
    counts = np.bincount((words.astype(np.uint64) >> 24).astype(np.int64), minlength=256)
    assert chisquare(counts).pvalue > 0.001


def test_masked_words_are_uncorrelated_with_the_update_they_hide() -> None:
    transcript, updates, scheme = _collect()
    correlations = []
    for name, masked in transcript.masked_vectors.items():
        encoded = scheme.encode(updates[name]).astype(np.float64)  # type: ignore[attr-defined]
        correlations.append(abs(float(np.corrcoef(masked, encoded)[0, 1])))
    assert max(correlations) < 0.2  # 256 samples: |r| < 0.2 is well inside the null


def test_no_plaintext_update_bytes_ever_reach_the_wire() -> None:
    """Every payload the coordinator receives is checked for the raw quantized update."""
    federation = build_federation(n_clients=4, threshold=3, dimension=32)
    transcript = TranscriptRecord()
    updates = federation.updates(seed=11, scale=0.05)
    result = run_round(
        federation.coordinator,
        federation.clients,
        updates,
        0,
        clock=federation.clock,
        transcript=transcript,
    )
    assert result.accepted

    import base64

    plaintext_blobs = {
        base64.b64encode(
            federation.coordinator.scheme.encode(update).astype("<u4").tobytes()
        ).decode("ascii")
        for update in updates.values()
    }
    joined = "\n".join(transcript.message_json)
    for blob in plaintext_blobs:
        assert blob not in joined
    # The masked vectors, by contrast, are present: that is what the coordinator is given.
    for masked in transcript.masked_vectors.values():
        encoded = base64.b64encode(masked.astype("<u4").tobytes()).decode("ascii")
        assert encoded in joined


def test_two_rounds_of_the_same_update_produce_different_masked_vectors() -> None:
    """Fresh per-round keys and nonces mean an unchanged update looks different every round."""
    federation = build_federation(n_clients=4, threshold=3, dimension=64)
    transcript = TranscriptRecord()
    updates = federation.updates(seed=5, scale=0.05)
    first_seen: dict[int, np.ndarray] = {}
    for round_id in range(2):
        run_round(
            federation.coordinator,
            federation.clients,
            updates,
            round_id,
            clock=federation.clock,
            transcript=transcript,
        )
        first_seen[round_id] = transcript.masked_vectors["silo_00"].copy()
    assert not np.array_equal(first_seen[0], first_seen[1])
