<!-- This is the single source of truth for AI coding agents on this project.
     CLAUDE.md only contains an @-import pointing here. Do not duplicate
     content into CLAUDE.md — edit this file only. -->

# AGENTS.md — J-Lens Capstone

Read by: Codex (native), Claude Code (via `@AGENTS.md` import in `CLAUDE.md`).
If you are another agent/tool reading this file, the same rules apply to you.

## What this project is

Interpretability study on `Qwen/Qwen3.5-4B`. Core question: does an internal
J-space signal predict prompt-injection leakage of a protected value, and at
what layer does that signal become predictive. DS bootcamp capstone, 3
people, 4 weeks.

**Team-agreed scope (final, as of team discussion):**
- **First deliverable:** single-model pipeline — one base model, one fitted
  lens. Leak-rate baseline, layer-band descriptive profile, per-layer
  predictive classifier.
- **Second analysis track (accepted, not a side-quest):** exploratory
  top-k token-readout analysis per layer (see "Two dataframes" below) —
  instruction-mode vocabulary, leak-vs-resist token comparison.
- **Second delivarable:** a second model (e.g. a LoRA-hardened variant)
  run through the *same* pipeline for a base-vs-hardened comparison.
- Pipeline work is in progress (first version). Treat existing
  pipeline code as the starting point to align with these guardrails, not to
  rebuild from scratch — but flag any conflicts with the rules below rather
  than silently overriding either the code or this file.

## jlens API — GROUND TRUTH, do not invent

`jlens` (github.com/anthropics/jacobian-lens) is new and post-dates your
training data. NEVER write jlens calls from memory or guess signatures.
Before writing or reviewing any code that touches the jlens API, read the
real source:
- `external/anthropic/jacobian-lens/jlens/lens.py`
- `external/anthropic/jacobian-lens/jlens/fitting.py`
- `external/anthropic/jacobian-lens/jlens/hf.py`
- `external/anthropic/jacobian-lens/README.md`

The submodule under `external/` is upstream/read-only — do NOT edit anything
in it. Our own pipeline code lives in `notebooks/j-lens-run.ipynb` (run
harness) and `notebooks/analysis_alex/j-lens-analysis.ipynb` /
`notebooks/analysis_friedrich/` (analysis) — `notebooks/jlens_walkthrough.ipynb`,
referenced here previously, does not exist in this repo.

If asked to summarize or confirm the API, quote the actual function
signatures found in source, not a paraphrase from general knowledge of
similar libraries.

Note: `j-lens-run.ipynb`'s `get_readouts()` does not call `lens.apply()`
directly — it reimplements the same forward+transport+unembed steps by hand
so it can slice down to a configurable subset of positions
(`READOUT_POSITIONS`, see "Run harness notes" below) before unembedding.
Keep this reimplementation consistent with `apply()`'s actual behavior in
`lens.py` when touching it.

## Methodology guardrails (do not silently violate these)

- **Validation split: `GroupKFold` by `category`, never a random
  `train_test_split`.** Injection prompts come in template families; letting variants of the same template straddle
  train/test is leakage and inflates accuracy. Group by `category`, not
  `template_id`: in `data/evaluation/injection_corpus.jsonl`, `template_id`
  is unique per row (no repeated variants), so grouping on it doesn't group
  anything. `category` (11 attack categories x 14 templates each, plus
  `control`) is the real template family. This applies to the per-layer classifiers and to any Ridge/Lasso cross-check model.
- **Open decision: how `control`/`benign` rows factor into the `category`
  fold groups.** Not yet decided whether `control` is its own held-out group
  like the attack categories, or always included in every fold as a negative
  baseline. Resolve explicitly before reporting cross-validated numbers —
  don't let this default silently per person/run.
- **In first iteration, the secret is a single token.** Verify with
  `tokenizer.encode(word, add_special_tokens=False)` → length must be 1, in
  the exact form it appears in-context (leading space matters for BPE
  tokenizers). Currently checked inline in `notebooks/j-lens-run.ipynb`
  (model/lens loading cell) — `scripts/check_secret_token.py`, referenced
  here previously, does not exist yet; extract it there if a standalone
  script is wanted.
- **Core claim rests on one model, one fitted lens.** A second model
  requires its own separately fitted lens (fine-tuning shifts internal
  representations), so a layer-level difference between two models cannot be
  cleanly attributed to the model alone — it may reflect lens calibration
  differences. If a cross-model comparison is built, this confound must be
  stated explicitly wherever results are reported, not buried in a footnote.
- **No causal claims.** This project does readout (correlational
  observation via the lens), not the ablation/swap interventions from the
  source paper. Don't phrase results as "the model decides to leak because
  X" — phrase as "X is predictive of / correlates with leakage."
- **Per-layer classifiers must be trained one layer at a time**, each on a
  single layer's feature (e.g. secret-token rank at that layer), not all
  layers jointly in one model. Layers are highly correlated (residual stream
  carries information forward additively), so a joint Ridge/Lasso model's
  coefficients are not a reliable localization signal — they redistribute
  weight across correlated neighbors or arbitrarily zero one out. Use the
  per-layer curve as the primary localization result; a joint Ridge/Lasso
  model is a secondary robustness cross-check only, not a replacement.
- **Log-transform skewed features before fitting.** `secret_rank` is heavily
  right-skewed (~1 to ~150k); fit per-layer models on `log(secret_rank)`,
  not the raw rank, or a linear model is dominated by the huge high-rank
  magnitudes and converges/calibrates poorly. This only matters for
  accuracy/calibration — ROC-AUC is invariant under monotonic transforms of
  the score, so raw-rank and log-rank give identical AUC.

## Two dataframes — the core interface

Two separate tables, joined by `run_id`. Do not merge them into one wide
table — they have different shapes and purposes.

In the current run harness (`notebooks/j-lens-run.ipynb`), "two dataframes"
(one per system prompt) means **one** combined per-run JSONL/dataframe with
`strictness` / `system_id` / `label` columns, not two physical output files
— split downstream by filtering on those columns. The row set is a full
cross-join: every `injection_corpus.jsonl` row (attack **and** `control`) is
run once per system prompt.

**1. Per-run record (wide, one row per prompt run) — primary quantitative
signal.**
```
run_id | model_id | lens_id | template_id | category | leaked | secret_rank_L12 | secret_rank_L13 | ...
```
Drives: leak-rate-by-category, layer-band descriptive profile, per-layer
predictive classifiers (GroupKFold by category).

**2. Token-readout table (long, one row per run × layer × rank position) —
exploratory/qualitative companion.**
```
run_id | layer | rank_position | token
```
Extracted from the same lens readout as table 1 (see the `lens.apply()` note
under jlens API above) — no extra model compute needed, just extract more of
what the readout already returns. Drives: instruction-mode vocabulary
detection, leak-vs-resist token comparison. Illustrative and exploratory —
does not carry the headline result; keep it clearly labeled as secondary to
the per-run predictive analysis.

Not yet populated at full-corpus scale: `READOUT_POSITIONS` (see "Run
harness notes" below) controls how many sequence positions get a readout at
all, and the full-corpus run used `"last"` (a single position per run, for
run-time reasons) — enough for table 1, not for this table's per-position
vocabulary analysis. A `"user"`, `"prompt"`, or `"all"` run is needed to actually build
table 2.

## Data files

Only `data/evaluation/injection_corpus.jsonl` and
`data/evaluation/system_prompts.jsonl` are current inputs to the jlens
pipeline. `data/evaluation/eval.jsonl`, `pilot_eval.jsonl`,
`jlens_pilot_test_20.jsonl`, and `scripts/export_synthetic_responses.py`
(with its `outputs/qwen35-4b-lora-*` artifacts) are a pre-single-token-secret
legacy track — multi-token secrets like `TRAIN-AMBER-PINE-1137`, no `jlens`
involvement, disconnected from the run harness. Do not wire them into the
jlens pipeline; doing so would violate the single-token-secret guardrail
above.

## Run harness notes (`notebooks/j-lens-run.ipynb`)

- `READOUT_POSITIONS` (`"last"` / `"last_n"` / `"user"` / `"prompt"` / `"all"`) controls how many
  sequence positions get a lens readout per run — the main lever on run
  time, since readout cost scales ~linearly with position count (each
  position needs a full-vocab unembed per layer). Default is `"user"` (only
  user-message tokens, excluding the system prompt, chat-template markers,
  and response). See Environment below for the
  timing this bought locally, and "Two dataframes" above for what it costs
  table 2.
- `"user"` reads only the user-message content. `"prompt"` includes the system
  prompt and chat-template markers but still excludes response positions.
- The run cell resumes: it reads `OUTPUT_FILE`'s existing `id`s on start,
  skips them, and appends (not overwrites). A trailing line that fails to
  parse is treated as a partial write from an interrupted run and retried,
  not counted as done.

## Environment

- Windows + Git Bash (Friedrich); confirm with each teammate what applies to
  their machine. Activate venv as `source .venv/Scripts/activate` (forward
  slashes — the `.ps1` script fails silently in Git Bash).
- Local machines may be CPU-torch only (no NVIDIA GPU) — this no longer
  rules out real runs by default. Empirically: a full injection_corpus run
  (220 prompts x 2 system prompts = 440 generate+readout calls) completed on
  a CPU-only machine (6-core/12-thread, no GPU) in ~2.5h with
  `READOUT_POSITIONS = "last"` in `j-lens-run.ipynb` (~20s/run). Use
  `torch.bfloat16`, not `float16`, for CPU runs — `float16` has poor/
  unsupported CPU kernel coverage in PyTorch. Colab/GPU is still the better
  default for `READOUT_POSITIONS = "user"`, `"prompt"`, or `"all"` runs (many positions
  need a full-vocab unembed at every layer), or for the larger
  `qwen36-27b` config. Don't write code that silently
  assumes a local GPU is available — check/handle the CPU-only case
  explicitly or flag it.

## Deliverable order (graceful degradation — each stage independently
presentable)

1. Behavioral: leak rate by injection category (base model).
2. Descriptive: layer-band profile — where the secret token surfaces in
   J-space.
3. Predictive: per-layer classifier, leak/resist, GroupKFold by category,
   per-layer ROC/accuracy curve.
4. Exploratory: token-readout vocabulary analysis (leak vs. resist).
5. Robustness + honest limitations (lens-fit sensitivity, Ridge/Lasso
   cross-check, error analysis on which template families break the
   classifier).
6. Stretch: second model (e.g. LoRA) through the same pipeline, confound
   stated explicitly.
