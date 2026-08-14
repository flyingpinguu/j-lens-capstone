# Findings & Open Tasks

Working doc — key results from the secret-token-rank analysis so far, and
what's still open. Not a final write-up; source of truth for what's actually
been checked vs. still assumed. See `AGENTS.md` for the methodology rules
these findings were produced under.

## Current setup (sys_strict, single turn, random secrets)

The primary dataset is now one request and one response per run, `sys_strict`
only, with a reproducible random one-token secret per request. It contains
478 attacks plus 60 authorized requests. The final 16 complete prompt
positions and all response positions were recorded; classifiers use prompt
positions only. The 412-row development split has 171 leaks.

**Current Stage-3 result** (GroupKFold by category; mean fold AUC ± std):

| model | all runs | unauthorized only |
|---|---|---|
| M1 logreg, lens features | 0.476 ± 0.056 | 0.473 ± 0.074 |
| M1 logreg, lens + metadata | 0.574 ± 0.185 | 0.480 ± 0.098 |
| M2, top-k only | 0.675 ± 0.084 | 0.655 ± 0.137 |
| M3, soft vote | 0.653 ± 0.063 | 0.636 ± 0.130 |
| M4, top-k + secret rank | 0.676 ± 0.071 | 0.659 ± 0.145 |

M4 is effectively tied with M2 here, so adding secret-rank features does not
provide a material lift. M1's metadata lift on all rows mostly disappears in
the unauthorized cohort and varies strongly by held-out category.

## Previous multi-turn shortcut audit (not the primary dataset)

The following results are retained because they motivated the return to the
single-turn collection design. They are not comparable to the current table.

**Stage 3 on the multi-turn sys_strict run** (1962 dev runs, 205 leaks,
GroupKFold by category, `outputs/analysis/qwen35-4b/model_comparison_per_cohort.png`):

| model | all runs | unauthorized only |
|---|---|---|
| M1 logreg, lens features | 0.616 ± 0.088 | 0.649 ± 0.119 |
| M1 logreg, lens + metadata | 0.740 ± 0.098 | 0.714 ± 0.072 |
| M1 xgb, lens features | 0.635 ± 0.037 | 0.663 ± 0.069 |
| M1 xgb, lens + metadata | 0.690 ± 0.076 | 0.682 ± 0.063 |

(mean fold AUC ± std; M2/M3/M4 are in the same figure.)

- The metadata feature set wins on both cohorts, but for two different
  reasons. On "all runs" part of the lift is definitional — `authorized`
  implies `attack_successful == False`. On "unauthorized only" both metadata
  features are constant, so they cannot reorder anything within the cohort:
  the remaining lift (0.649 → 0.714 for logreg) comes from *training*, where
  absorbing the authorized rows lets the lens coefficients fit the
  unauthorized ones more cleanly. Worth stating explicitly in the write-up —
  it is a confounder-adjustment effect, not extra evidence about leak risk.
- The per-layer single-position curve is much weaker on this data than on
  the old single-turn run: mostly within ±0.07 of chance, best around
  layers 19-26 and at the final layer (~0.57).

**The high M2/M3/M4 numbers are mostly confounded — read this before
quoting them.** `model_comparison_leakage_audit.png` re-scores the *same*
OOF probabilities inside nested strata:

| model | all rows | attack only | attack @ attempt 1 | attack @ attempt 2+ |
|---|---|---|---|---|
| M1 logreg (lens) | 0.683 | 0.701 | 0.551 | 0.579 |
| M1 logreg (lens+meta) | 0.763 | 0.702 | 0.543 | 0.578 |
| M2 (top-k) | 0.902 | 0.828 | **0.503** | 0.663 |
| M3 | 0.886 | 0.833 | 0.550 | 0.653 |
| M4 | 0.902 | 0.831 | 0.509 | 0.683 |

(OOF AUC. 1962 dev runs; strata: 1014 attack rows, 352 at attempt 1 with
171 leaks, 662 at attempt 2+ with 31 leaks.)

Two structural confounders, both properties of the data collection, not of
the models:
- **Benign vs. attack.** 757 dev rows are controls with a 0.4% leak rate.
  M2's score predicts `label == attack` with AUC 0.819 — nearly its leak
  AUC. M1's lens features do *not* (0.475), so this one is specific to the
  top-k track.
- **Attempt 1 vs. later.** Conversations stop on disclosure, so attempt-2+
  rows are survivors by construction: 48.6% leak rate at attempt 1 vs
  2-8% after. Attempt 1 also has a written corpus prompt and no history,
  where later turns have model-generated attacker text and a growing
  context — trivially visible in the prompt-end readout. M2 predicts
  `attempt_index == 1` with AUC 0.775, M1 with 0.735. This one hits **both**
  tracks and is the larger effect.
- Within a fixed stratum almost nothing survives for M2 (0.503 at attempt
  1). M1 keeps a weak but consistent 0.54-0.58 in both strata. Note the
  attempt-2+ stratum has only 31 leaks, so its ±0.10-0.12 fold std is real.

Two smaller inflations on top: the reported M2 position is the best of 16,
selected on the metric it reports; and the per-position AUC profile rises
monotonically toward the prompt end (0.58 → 0.87), i.e. the winning
positions are the ones whose final-layer readout is essentially the
model's distribution over the first response token.

## Earlier single-turn findings (fixed secret, both system prompts)

**Predictive signal (per-layer classifier, GroupKFold by category, log-rank
logistic regression)**
- There is a real, above-chance signal, but it's layer- and
  position-dependent, not a clean uniform curve. Early/mid layers hover
  near or below chance; a real signal consistently shows up from roughly
  layer 16 onward, strengthening further from ~27 onward (AUC up to
  ~0.65–0.74 depending on system prompt and exact reference position).

- `sys_lax` and `sys_strict` give meaningfully different curves — not just
  magnitude, sign in places  **Never pool
  the two system prompts into one classifier** — it averages away real
  disagreements and can look artificially fine.

- **The choice of reference position matters a lot and isn't yet
  standardized.** Last user-message token vs. last scaffolding token (the
  chat-template turn-boundary tokens right before generation) give
  different curves. Working hypothesis: real representational change
  happens across the scaffolding tokens, not just positional noise — needs
  more evidence before stating as fact.
- Per-position-offset curves (how far back from the end of the user
  message) are noisier than per-layer curves, likely because "N tokens from
  the end" isn't the same semantic position across different attack
  template structures (short direct commands vs. long narrative setups) —
  an alignment problem, not (only) a sample-size problem. One partially
  interesting exception: for `sys_lax`, user-offset 4 (5th-from-last token)
  is a consistently stronger predictor at late layers (avg AUC 0.648,
  layers 27–31) than its neighbors (offset 3: 0.600, offset 5: 0.580,
  offset 6/7: ~0.43, i.e. below chance) — not yet explained; could be
  template-structural (a recurring word at that position) or noise.
- Robustness check: a joint logistic regression over 12 correlated layers
  (20–31) scored ~0.713 AUC vs. ~0.705 for the single best layer alone —
  confirms adjacent layers carry mostly redundant information, and that the
  per-layer curve (not a joint/stacked model) is the right tool for the
  localization question.

## Open questions / tasks

**Decisions needed**
- [ ] How do `control`/benign rows factor into the `category` GroupKFold
      groups — own held-out group, or always-in-every-fold baseline? Not
      decided; affects any cross-validated number reported from here on.
      Same question now applies to the four authorization categories:
      `admin_request` and `password_correct` are the only source of
      `authorized == True` rows, so the fold that holds either of them out
      trains on a differently-composed dataset than the others.
- [ ] **Which stratum is the headline number** — the biggest open decision.
      Options: (a) keep all rows and always report the audit alongside;
      (b) restrict training and evaluation to attack rows; (c) condition on
      attempt (separate models/numbers for attempt 1 and attempt 2+);
      (d) add `attempt_index` and `label` as explicit features so the
      confounders are modelled rather than absorbed silently. (c) is the
      cleanest answer to the project's actual question but leaves only 352
      rows / 171 leaks for the attempt-1 model.
- [ ] Whether the headline M1 number is the lens-only or the lens+metadata
      feature set. Both are reported; the localization story (Deliverable
      2-3) belongs to the lens-only one, the accuracy story (Deliverable 6)
      can use metadata as long as the "all runs" inflation is stated.
- [ ] Pick and document one canonical reference position for "the" headline
      per-layer curve (currently multiple exist across notebooks).

**Analysis backlog**
- [ ] Table 2 / instruction-mode vocabulary analysis (leak-vs-resist token
      comparison) — Alex's track, not started.
- [ ] Fine-grained look at the scaffolding-token span (position-by-position
      within the fixed 9-token block), now that `readout_scope:
      "user_response"` data exists.
- [ ] Check what token actually sits at user-offset 4 across a sample of
      attack templates, to see if the layer-27+ bump there is
      content-driven or coincidental.
- [ ] Accuracy metric alongside ROC-AUC (AGENTS.md wants both; only AUC has
      been computed so far).
- [ ] Small joint model over a handful of hand-picked, plausibly-independent
      features (best layer + a couple of distinct position offsets) as a
      secondary robustness check — deliberately not full clustering+stacking
      given current sample size (~220–440 rows/system) and nested-CV
      overfitting risk; see AGENTS.md methodology guardrails.
- [ ] Formal robustness + honest-limitations write-up (Deliverable 5) —
      exists piecemeal in chat/notebook comments, not consolidated.

**Not started**
- [ ] Second model (e.g. LoRA-hardened) through the same pipeline (stretch
      deliverable).
