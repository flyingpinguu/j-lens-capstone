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

- `notebooks/j-lens-run.ipynb`: Main inference pipeline. It reads JSONL prompts, generates responses, applies an existing J-Lens, and writes one JSON object per example.
- `notebooks/j-lens-analysis.ipynb`: Loads runner output into a pandas DataFrame and visualizes common readout tokens by position and layer.
- `notebooks/qwen35_lora_pilot.ipynb`: LoRA training and behavioral pilot.
- `data/training/pilot_train.jsonl`: Pilot training conversations.
- `data/evaluation/eval.jsonl`: Held-out evaluation data.
- `data/evaluation/jlens_pilot_test_20.jsonl`: Small runner test containing 10 attack and 10 benign prompts.
- `outputs/j-lens-run/`: Generated responses and hierarchical J-Lens readouts.
- `external/anthropic/jacobian-lens/`: Pinned upstream Anthropic repository. Keep project-specific changes outside this directory.

## Current runner conventions

The runner expects each input JSONL row to contain:

```json
{
  "id": "example-id",
  "system_prompt": "...",
  "user_prompt": "..."
}
```

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
PROBE_TOKEN = " secret"
```

The probe string must tokenize to exactly one vocabulary token. Its full-vocabulary rank and logit are saved for every token position and layer.

## Working guidance

- Keep notebook code concise and easy for students to understand.
- Prefer the fixed, known data schema over abstractions for hypothetical input formats.
- The runner currently applies existing lenses; lens fitting is a separate workflow.
- A modified or LoRA-fine-tuned model requires its own fitted lens.
- Use the repository `.venv` and `requirements.txt` for the recorded environment.
- Preserve unrelated work and generated results unless a task explicitly replaces them.
