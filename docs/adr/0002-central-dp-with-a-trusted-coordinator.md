# 2. Add the DP noise centrally, and say who has to be trusted

- Status: accepted
- Date: 2026-09-20

## Context

Secure aggregation hides individual updates from the coordinator, but the *sum* it reconstructs is
still a function of everyone's data and can leak training examples. Differential privacy is what
bounds that leak, and there are three places to put the noise:

1. **Local DP** -- each silo adds noise to its own update before masking. Nobody has to be trusted.
2. **Central DP with a trusted curator** -- the coordinator adds noise to the reconstructed sum.
3. **Distributed DP** -- each silo adds a share of the noise so the sum carries the right total,
   typically with a discrete Gaussian over the same ring as the masks.

Option 1 is the strongest threat model and the worst utility. Each of `n` silos must survive on its
own, so the noise the model sees grows like `sqrt(n)` relative to option 2. With five silos, a
1 345-parameter model and a single-digit epsilon, the model measured here already drops from 0.869
to 0.784 AUC at epsilon = 8 under *central* DP. Under local DP at the same epsilon it would be at
chance, and the repo would be demonstrating a technique nobody would deploy.

Option 3 is the right answer for production and the honest middle ground: it gets central-DP
utility without trusting the coordinator, because no single party ever holds the noiseless sum. It
also needs a discrete Gaussian sampled over `Z_2^32`, a per-silo share of the variance that
survives dropouts (the dropped silos' noise shares vanish with them, so the total variance has to
be padded), and its own accounting. That is a substantial second cryptographic subsystem.

## Decision

Add the Gaussian noise **centrally**, in the coordinator, after unmasking and after the norm bound
has accepted the round. State in the README, the model card and the threat model that this makes
the coordinator a **trusted curator for the DP guarantee**, while it remains only
honest-but-curious for the secure-aggregation guarantee.

Keep the two guarantees separate and separately falsifiable:

- secure aggregation is checked by transcript tests and bit-for-bit exactness tests;
- the DP guarantee is checked by an accountant cross-checked against `dp-accounting`, *and* by an
  empirical canary audit that must flag deliberately broken variants.

Record distributed DP as a non-goal rather than pretending the current design achieves it.

## Consequences

- Utility stays in a range where the privacy/accuracy trade-off is legible: the README can show
  what epsilon = 8, 4 and 1 actually cost, instead of showing three rows of noise.
- The threat model gains a named residual risk: a coordinator that silently skips the noise still
  produces a plausible run. Only the canary audit would notice, and only for gross cases.
- The audit becomes load-bearing rather than decorative. Because the coordinator is trusted to add
  noise correctly, an *independent* check that the noise is actually there is the only thing
  standing between "we used DP" and a claim nobody verified. This is why three broken variants ship
  in the repo and CI fails if they are not flagged.
- Moving to distributed DP later is a contained change: the mask ring, the quantization scheme and
  the accountant all stay: what changes is who samples the noise and how dropped noise shares are
  compensated.
