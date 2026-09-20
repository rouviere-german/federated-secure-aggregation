# 3. Defend poisoning at the aggregate, and publish what that cannot catch

- Status: accepted
- Date: 2026-09-20

## Context

Standard robust-aggregation defences -- coordinate-wise median, trimmed mean, Krum, norm-clipping
per client -- all need the server to **see each client's update**. Secure aggregation exists
precisely to stop that. The two are in direct tension, and the literature's answer (zero-knowledge
range proofs, as in Prio or RoFL) lets a client prove its update is short without revealing it.

That is the correct solution and it is expensive: a proof system, a per-coordinate commitment
scheme, and verification cost that grows with the model. Implementing it credibly is a project of
its own, not a section of this one.

What is cheap is a bound on the *reconstructed* aggregate. Every honest silo clips its update to
`C`, so an honest round's aggregate cannot be longer than `survivors x C` plus the quantization
rounding allowance. A silo that inflates its update pushes the aggregate past that bound.

## Decision

Enforce a norm bound on the **unmasked aggregate** and roll the round back when it is exceeded:
the global model is restored, the round is recorded as `REJECTED` in the audit chain, and **no
privacy budget is spent**.

Deliberately do *not*:

- try to attribute the rejection to a silo (impossible by construction, and guessing would be
  worse than nothing);
- silently drop the round (it is recorded, counted in `rounds_rejected_total`, and visible in the
  demo);
- claim this is a general poisoning defence.

Instead, measure the gap and publish it. `bench/bench_poisoning.py` runs a norm-respecting
**sign-flip** attacker alongside the scaled ones and reports that it is never rejected, with its
AUC impact, in the README's Results section.

## Consequences

- Scaled attacks are caught completely: at scale 2x and above the rejection rate is 1.0, with no
  false positives on clean rounds (measured, `bench/bench_poisoning.py`).
- The blind spot is explicit: a sign-flipping silo inside the bound passes every check. It cost
  0.4-0.7 AUC points in the measured runs, but nothing here would detect a worse variant.
- Rejection is **collective punishment**: one bad silo costs everyone the round. With `t` close to
  `n` that is a denial-of-service lever, which is a second reason the limitation is stated rather
  than buried.
- Choosing `C` is now a visible trade-off rather than a hyperparameter: a large `C` admits bigger
  poisoned updates, a small `C` clips honest ones and slows learning. The bound and the DP
  sensitivity are the same `C`, so tightening one tightens the other.
- Adding range proofs later does not invalidate this: the aggregate bound stays as a cheap
  backstop, and per-client proofs become an additional, stronger layer.
