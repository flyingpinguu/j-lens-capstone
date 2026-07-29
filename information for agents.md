# Information for agents

## Project overview

This capstone studies whether LoRA fine-tuning for prompt-injection resistance changes both the output behavior and the internal representations of `Qwen/Qwen3.5-4B`.

The experiments use synthetic conversations in which a system prompt contains a protected value. Attack prompts try to extract it, while benign prompts test whether the model can still answer normal questions. The base and fine-tuned models are compared using:

- behavioral results such as protected-value leakage and benign-task performance;
- Jacobian Lens readouts across token positions and model layers.

## Project goals

1. Measure whether the LoRA adapter reduces prompt-injection leakage without causing excessive refusal on benign requests.
2. Use J-Lens readouts to inspect when protected or attack-related concepts become salient inside the model.
3. Compare base and fine-tuned models using the same prompts, generation settings, token positions, layers, and compatible lenses.
4. Maintain a reusable pipeline that can run different Hugging Face causal language models with existing matching J-Lenses.

## Important files

- `notebooks/j-lens-run.ipynb`: Main inference pipeline. It samples attacks from the injection corpus, inserts a secret into a selected system prompt, generates responses, applies an existing J-Lens, and writes one JSON object per attack.
- `notebooks/analysis_alex/j-lens-analysis.ipynb`: Loads runner output and lets the user inspect the top-10 tokens for one prompt, token position, and layer.
- `data/training/pilot_train.jsonl`: Pilot training conversations.
- `data/evaluation/injection_corpus.jsonl`: Injection templates and benign controls.
- `data/evaluation/system_prompts.jsonl`: System-prompt templates containing the `{{SECRET}}` placeholder.
- `outputs/j-lens-run/`: Generated responses and hierarchical J-Lens readouts.
- `external/anthropic/jacobian-lens/`: Pinned upstream Anthropic repository. Keep project-specific changes outside this directory.

## Current runner conventions

The runner reads `injection_corpus.jsonl` and `system_prompts.jsonl`. At the
top of the notebook, `SECRET` selects the inserted secret.
`MAX_PROMPTS_PER_STRICTNESS = None` runs the full corpus in its original
order. An integer selects a reproducible random sample without replacement
using `RANDOM_SEED`; the same sample is paired with every system prompt.

Readouts are stored as:

```text
example
└── readouts
    └── token position
        └── layers
            └── layer
                ├── top_k
                └── probe
```

The optional single-token probe is configured near the top of the runner:

```python
PROBE_ENABLED = True
PROBE_TOKEN = SECRET
```

The probe string must tokenize to exactly one vocabulary token. Its full-vocabulary rank and logit are saved for every token position and layer.

Each output record also contains `attack_successful`. It is `true` when the
configured secret occurs in the generated response, using a case-insensitive
substring check, and `false` otherwise. This is the target for later models
trained on the J-Lens readouts.

The analysis notebook creates three views of a run:

- `prompts_df`: one row per attack with metadata, response, and target;
- `readouts_df`: one row per attack, token position, and layer, with the top-k
  token IDs, tokens, and logits kept as lists;
- `classifier_df`: one row per attack using probe ranks and logits from every
  layer at the final prompt position.

Classifier features exclude response positions to avoid leaking the outcome
into the predictors.

`READOUT_POSITIONS = "user"` stores readouts only for tokens belonging to the
user message. It excludes the system prompt, chat-template markers, and
generated response. `"prompt"` remains available for all prompt tokens. The
selected position mode is part of the output filename so resume mode cannot
mix incompatible readout scopes.

## Working guidance

- Keep notebook code concise and easy for students to understand. simplicity over unnecessary function
- Prefer the fixed, known data schema over abstractions for hypothetical input formats.
- The runner currently applies existing lenses; lens fitting is a separate workflow.
- A modified or LoRA-fine-tuned model requires its own fitted lens.
- Use the repository `.venv` and `requirements.txt` for the recorded environment.
- Preserve unrelated work and generated results unless a task explicitly replaces them.
- Don't solve tasks you are not asked to solve. If the user tells you about an idea or issue, maybe he just wants to talk about it first.
