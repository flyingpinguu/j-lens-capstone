# Findings & Open Tasks

Working doc — key results from the secret-token-rank analysis so far, and
what's still open. Not a final write-up; source of truth for what's actually
been checked vs. still assumed. See `AGENTS.md` for the methodology rules
these findings were produced under.

## Key findings

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
