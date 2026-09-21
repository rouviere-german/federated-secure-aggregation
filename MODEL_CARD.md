# Model card: credit-default-federated

Template based on Mitchell et al., "Model Cards for Model Reporting" (FAT* 2019). Every number
below is copied from a real run, with the command that produced it.

**Read this first:** the model is a vehicle for demonstrating a protocol, not a credit-risk
product. It is trained on synthetic borrowers generated from a known logistic model. Its AUC
describes that simulation and nothing else.

## Model details

- **Name and version:** `credit-default-federated`; registry digest printed by the demo as
  `model_version` (content-addressed, so it changes only when the artifact changes).
- **Type:** binary logistic regression, 13 parameters (12 standardized features plus bias), trained
  by federated averaging over 5 silos. A 1-hidden-layer MLP (`MLPClassifier`) is implemented and
  gradient-tested but is not the registered model.
- **Owner:** Rouviere German
- **Date:** 2026-09-21
- **License:** MIT (code). Data is synthetic and generated in-process; no external dataset.
- **Artifact format:** `.npz` with `allow_pickle=False`. Never pickle
  (`tests/unit/test_no_pickle.py::test_source_tree_has_no_pickle_family_serialization`).
- **Run manifest:** written by `Experiment.manifest` into the registry entry: global seed, derived
  per-silo seeds, partition fingerprint, Dirichlet alpha, accountant parameters, package versions.

## Intended use

- **Primary use:** demonstrating dropout-tolerant secure aggregation, an enforced DP budget and an
  empirical privacy audit, end to end, on a CPU, offline.
- **Intended users:** engineers evaluating federated-learning designs; reviewers of this portfolio.
- **Out-of-scope uses:** any real lending, credit or eligibility decision; any decision about a
  real person; benchmarking credit-risk modelling; transfer to real borrower data without
  re-deriving the feature contract, the clip norm and the privacy budget.

## Training data

- **Source:** synthetic, `federated_secure_aggregation.fl.data.make_population`, seed `20260920`,
  6 000 borrowers, base default rate 0.18. Labels are drawn from a known logistic model over the
  features, so the Bayes-optimal ranking exists and the ceiling is knowable.
- **Partitioning:** Dirichlet label and feature skew across 5 silos
  (`partition_dirichlet`), alpha = 1.0 for the demo; benches sweep alpha in {0.1, 1, 10}. Silos are
  disjoint and a held-out split is reserved before partitioning.
- **Fingerprint:** each run logs `partition_fingerprint`; two runs with the same seed produce the
  same fingerprint (`tests/integration/test_demos_cli.py::test_demo_learning_is_reproducible_bit_for_bit`).
- **Contract:** `federated_secure_aggregation.ml.contracts.BorrowerSchema` (pandera, strict, lazy).
  A silo whose rows fail the contract is **rejected**, not quarantined or coerced: it never joins
  the round. Failure summaries carry column, check and count only, never a borrower value.
- **Known gaps:** the generator has no missing values, no temporal drift, no correlated
  demographic attributes and no protected characteristics. It therefore cannot be used to study
  fairness, and the absence of a fairness section below is a limitation, not a clean bill.

## Evaluation

- **Data:** a held-out split carved off before partitioning and never given to any silo.
- **Metrics:** ROC AUC (ranking quality, the decision-relevant metric for risk scoring) and log
  loss (calibration). Both are cross-checked against scikit-learn
  (`tests/unit/test_metrics_and_data.py::test_roc_auc_matches_scikit_learn`).
- **Results:** from `uv run python bench/bench_utility.py --check` on 2026-09-21 (Windows 10,
  6 logical CPUs, Python 3.12.14, CPU only):

  | Setting (alpha = 1) | AUC | log loss |
  |---|---:|---:|
  | centralized (pooled, upper reference) | 0.8681 | 0.3317 |
  | local-only (no federation, lower reference) | 0.8413 | 0.6379 |
  | federated, plaintext, no DP | 0.8690 | 0.3305 |
  | **federated, full secure-aggregation protocol, no DP** | **0.8690** | 0.3305 |
  | federated + DP, epsilon = 8 (z = 2.85) | 0.7837 | 0.4215 |
  | federated + DP, epsilon = 4 (z = 5.18) | 0.7199 | 0.5978 |
  | federated + DP, epsilon = 1 (z = 18.09) | 0.5646 | 5.8861 |

- **Slices:** by partition skew, `uv run python bench/bench_utility.py --check`:

  | Dirichlet alpha | local-only AUC | federated AUC (no DP) | federated AUC (eps = 8) |
  |---:|---:|---:|---:|
  | 0.1 (severe skew) | 0.7562 | 0.8584 | 0.7173 |
  | 1 (moderate) | 0.8413 | 0.8690 | 0.7837 |
  | 10 (near IID) | 0.8736 | 0.8862 | 0.7891 |

- **Cost of the protocol:** zero. Secure aggregation matches plaintext federated averaging to four
  decimals at every alpha (threshold: 0.005 AUC; measured gap: 0.0000).
- **Cost of the privacy guarantee:** 8.5 AUC points at epsilon = 8, and collapse to near chance at
  epsilon = 1. On a 13-parameter model over 20 rounds with 5 silos, single-digit epsilon is
  genuinely expensive, and that is the headline result of this card.

## Privacy guarantee

- **Type:** central (trusted-curator) `(epsilon, delta)`-differential privacy over the clipped
  aggregate, with `delta = 1e-5`. The coordinator adds the noise; see
  [ADR 0002](docs/adr/0002-central-dp-with-a-trusted-coordinator.md) for who must be trusted.
- **Unit of privacy:** one silo's contribution to one round (update-level, not record-level).
  A silo's *whole* dataset influence is what is bounded, via the L2 clip `C = 1.0`.
- **Accounting:** in-repo RDP accountant for the subsampled Gaussian, cross-checked against
  Google's `dp-accounting` (`tests/unit/test_accountant_reference.py`). A persisted ledger refuses
  the round that would exceed the target and survives restarts.
- **What the reported epsilon does not cover.** The aggregate norm bound is tested against the
  *noiseless* sum, and the accept/reject outcome is published (demo table, audit chain,
  `rounds_rejected_total`). That is one data-dependent bit per round released outside the
  accountant, so the real loss is the reported epsilon **plus** that bit per round. A rejected
  round charges the ledger nothing -- honest bookkeeping, not a privacy control. Applying the
  bound to the noised aggregate would make the decision post-processing and close the gap; see
  README Non-goals for why that trade is recorded rather than taken.
- **Audited epsilon:** from `uv run python bench/bench_audit.py --check`, 48 canary clients against
  1 024 reference directions, 30 rounds, noise multiplier 4.0, 1 345-parameter MLP, **canary scale
  20.0**, claimed epsilon 6.8379:

  | variant | breaks DP? | mu hat | eps low (95%) | flagged |
  |---|---|---:|---:|---|
  | correct | no | 1.1736 | 3.7665 | no |
  | no_clipping | yes | 5.0902 | 31.57 | **yes** |
  | weak_noise | yes | 4.6750 | 28.08 | **yes** |
  | average_sensitivity | yes | 4.9635 | 30.52 | **yes** |
  | clip_after_noise | no | 1.1778 | 3.7847 | no |

  The correct implementation's empirical mu (1.1736) sits below the theoretical mu (1.3693), which
  is the right direction for a lower bound. All three broken variants are flagged; CI fails
  otherwise. See [ADR 0004](docs/adr/0004-ship-broken-dp-variants-as-audit-tests.md).
- **What `no_clipping` detection depends on.** Each canary submits 20x the clip norm, so that row
  measures a 20x-oversized client rather than a missing clip in general. At `canary_scale = 1.0`
  the same variant gives mu_hat 1.1733 against the correct implementation's 1.1736 and is not
  flagged. The audit's power against this bug is a statement about outliers; see README.

## Robustness, drift and monitoring

- **Adversarial tests:** scaled-update attackers at 1.5x-40x
  (`bench/bench_poisoning.py`, `tests/integration/test_federated_run.py::test_a_scaled_attacker_is_rejected_and_the_model_is_rolled_back`)
  and a norm-respecting sign-flip attacker
  (`tests/integration/test_federated_run.py::test_a_sign_flip_attacker_passes_every_check`).
- **Measured robustness:** scaled attacks at >= 2x are rejected 100% of the time with no false
  positives on clean rounds. **A sign-flip attacker inside the norm bound is never rejected** and
  cost 0.4-0.7 AUC points in the measured runs. See
  [ADR 0003](docs/adr/0003-aggregate-norm-bound-instead-of-per-client-defence.md).
- **Drift monitors:** PSI and KS (`src/federated_secure_aggregation/ml/drift.py`), PSI threshold
  0.2, KS alpha 0.01 with a minimum statistic of 0.05, applied to the per-round aggregate norm and
  the holdout score distribution.
- **Performance monitors:** `fedsecagg_aggregate_norm`, `fedsecagg_rounds_rejected_total`,
  `fedsecagg_epsilon_spent`, `fedsecagg_dropouts_total`, `fedsecagg_round_duration_seconds`,
  exposed on the authenticated `/metrics` endpoint. No metric carries a per-client label, so the
  endpoint cannot be used to profile individual silos.
- **`fedsecagg_global_holdout_auc` is set by the in-process runner only.** The HTTP coordinator
  holds no holdout split and no labels -- the data stays with the silos -- so it cannot evaluate
  the model, and the served gauge stays unset. AUC comes from `uv run python tasks.py demo` and
  the utility bench. Epsilon is served: the coordinator adds the calibrated noise, charges the
  budget ledger and refuses to open a round the budget cannot pay for
  (`tests/integration/test_service_privacy.py`).
- **Retraining trigger:** the budget ledger is the hard stop -- once the target epsilon is spent,
  no further round runs. Restarting training means a new budget, and that is a policy decision, not
  an automatic one.

## Ethical considerations

- **Who could be harmed:** in a real deployment, borrowers whose data trains a model that then
  scores them. The protocol reduces exposure of individual updates but does not make the model
  fair, accurate or contestable.
- **Fairness is not evaluated.** The synthetic generator has no protected attributes, so no
  subgroup analysis is possible here. A real deployment would need one before any use, and nothing
  in this repo substitutes for it.
- **`region_risk` is a feature.** In the simulation it is a synthetic scalar. In reality a
  region-level risk score is a well-known proxy for protected characteristics, and using one would
  need explicit justification and disparate-impact testing.
- **Privacy of training data:** bounded by the DP guarantee *if* the coordinator is honest about
  adding noise. The canary audit is the check on that, and it catches gross failures only.
- **Secure aggregation hides misbehaviour as well as data.** A silo poisoning the model inside the
  norm bound cannot be identified, which means accountability between institutions is weaker than
  in a plaintext setup. That is a governance cost, not just a technical one.

## Caveats and recommendations

- Results hold for this synthetic population, this 13-parameter model, 5 silos and 20 rounds. They
  do not transfer.
- The privacy guarantee assumes the coordinator follows the protocol. A malicious coordinator can
  isolate a silo by lying about the survivor set; only the "no early stage close" rule makes that
  costly, not impossible.
- `t - 1` colluding silos plus the coordinator can unmask a victim. Choose `t` accordingly.
- The audit is a lower bound and demonstrably misses `clip_after_noise`. Passing it is not proof of
  a correct implementation.
- Before any real use: re-derive the feature contract from real data, re-tune `C` against real
  update norms, redo the privacy accounting for the real record-to-silo mapping, and run a fairness
  evaluation that this card cannot provide.
