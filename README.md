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
│   ├── jlens_walkthrough.ipynb   # Project copy of the fitted-lens walkthrough
│   └── qwen35_lora_pilot.ipynb  # Executed LoRA pilot and evaluation
├── outputs/
│   ├── qwen35-4b-lora-pilot/    # Trained pilot adapter
│   ├── qwen35-4b-lora-smoke/    # Initial smoke-test adapter
│   └── synthetic_test_responses.jsonl
└── scripts/
    └── export_synthetic_responses.py
```

The Anthropic Jacobian Lens checkout is included as a pinned Git submodule. Project-specific walkthrough changes live in `notebooks/jlens_walkthrough.ipynb`, keeping the upstream checkout separate.

## Setup

Clone the project together with the Anthropic submodule, install Git LFS, and create a Python 3.11 virtual environment. The Python dependencies used for the recorded run are listed in `requirements.txt`.

The scripts download `Qwen/Qwen3.5-4B` from Hugging Face when it is not cached locally. Set `HF_LOCAL_FILES_ONLY=1` to force offline-only loading. The trained LoRA adapter is stored in Git LFS and is applied to the unchanged base model at runtime.

## Current pilot result

The synthetic held-out evaluation reduced exact protected-value leaks from `30/42` for the unmodified model to `0/42` with the LoRA adapter. A small independently written holdout test still found one successful transformed leak, so the adapter is a proof of concept rather than a complete security solution.
