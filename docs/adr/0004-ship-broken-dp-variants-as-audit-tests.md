# 4. Ship deliberately broken DP variants and require the audit to catch them

- Status: accepted
- Date: 2026-09-20

## Context

A privacy accountant answers "given these parameters, what is epsilon?". It does not answer "does
this code implement those parameters?". Every common DP-in-federated-learning bug leaves the
accountant perfectly happy:

- clipping that is computed but never applied;
- noise scaled by `C` when the sensitivity is really `2C`, or divided by `n` because someone
  thought in terms of the *average* update rather than the sum;
- clipping applied *after* the noise, which rescales the noise instead of bounding sensitivity.

In each case the logs show the epsilon you asked for and the guarantee is void. Testing the
accountant against a reference (we do, against Google's `dp-accounting`) does not help, because the
accountant is not the part that is wrong.

An empirical audit does help. Following Andrew et al. (2023), inserting canary clients whose
updates are random unit directions and measuring how much the final model moves along those
directions gives a *lower bound* on the privacy loss from a single training run. If that lower
bound exceeds the claimed epsilon, the implementation is provably not delivering the claim.

The risk with any such audit is that it becomes a green tick nobody has tested. An audit that
always passes is indistinguishable from an audit that is broken.

## Decision

Ship the broken implementations in the library as a named enum (`MechanismVariant`), and make
`bench/bench_audit.py` audit **all** of them in the same run as the correct one:

| variant | breaks DP? | must be flagged? |
|---|---|---|
| `CORRECT` | no | no |
| `NO_CLIPPING` | yes | yes |
| `WEAK_NOISE` | yes | yes |
| `AVERAGE_SENSITIVITY` | yes | yes |
| `CLIP_AFTER_NOISE` | no (and not caught) | no |

CI fails if the count of flagged broken variants is not exactly 3, or if either sound variant is
flagged. The thresholds live in `bench/thresholds.json` as `broken_variants_flagged` and
`sound_variants_flagged`, so the audit's *power* is a regression-tested property, not an assertion
in prose.

`CLIP_AFTER_NOISE` is kept in the table even though it is not flagged, and is reported in the
README as a miss.

## Consequences

- The audit is now falsifiable. If someone weakens the estimator, the broken variants stop being
  flagged and CI goes red; if someone makes it hysterical, the sound variants get flagged and CI
  goes red.
- The repo contains code that is wrong on purpose. It is confined to one enum, documented as such
  at its definition, and never reachable from the demo or the service path -- only the bench and
  the tests select a non-`CORRECT` variant.
- The README can state the audit's limits with evidence rather than as a disclaimer:
  `clip_after_noise` is a real bug that the audit measurably does not catch at this noise scale,
  which is the honest summary of what a one-shot empirical audit buys.
- The audit costs one extra training run per variant (about 4 seconds total on CPU here), so it is
  cheap enough to run on every push rather than nightly.
- The audited epsilon for the correct implementation (mu = 1.17) sits *below* the theoretical value
  (mu = 1.37). That is the expected direction for a lower bound, and the ratio is itself pinned by
  a threshold (`mu_recovery_ratio`), so an audit that silently loses power is caught.
