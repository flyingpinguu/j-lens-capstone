# J-Lens Capstone

This project studies whether LoRA fine-tuning for prompt-injection resistance changes both the output behavior and the internal layer representations of `Qwen/Qwen3.5-4B`.

The base and fine-tuned models are tested with system prompts containing protected values. Their leak rates and normal-task behavior are compared, while Jacobian Lens is used to inspect where injected instructions begin to influence the model across layers.

## Repository structure

```text
j-lens-capstone/
├── README.md
├── data/
│   ├── training/                 # LoRA training conversations
│   └── evaluation/               # Held-out synthetic test prompts
├── docs/
│   └── project-description.md    # Extended research question and hypothesis
├── external/
│   └── anthropic/
│       └── jacobian-lens/        # Separate upstream Anthropic repository
├── notebooks/
│   └── qwen35_lora_pilot.ipynb  # Executed LoRA pilot and evaluation
├── outputs/
│   ├── qwen35-4b-lora-pilot/    # Trained pilot adapter
│   ├── qwen35-4b-lora-smoke/    # Initial smoke-test adapter
│   └── synthetic_test_responses.jsonl
└── scripts/
    └── export_synthetic_responses.py
```

The Anthropic Jacobian Lens checkout retains its own Git history and local walkthrough changes. This project directory itself is intentionally not initialized as a Git repository yet.

## Current pilot result

The synthetic held-out evaluation reduced exact protected-value leaks from `30/42` for the unmodified model to `0/42` with the LoRA adapter. A small independently written holdout test still found one successful transformed leak, so the adapter is a proof of concept rather than a complete security solution.
