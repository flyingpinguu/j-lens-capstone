# J-Lens Capstone

This project studies whether Jacobian-Lens readouts from `Qwen/Qwen3.5-4B`
predict whether a prompt injection leaks a protected one-token secret. The
current primary dataset uses one attack or authorized request, one response,
unique secrets, and the strict system prompt.

## Repository structure

```text
j-lens-capstone/
├── README.md
├── data/evaluation/              # Injection corpus and system prompts
├── external/
│   └── anthropic/
│       └── jacobian-lens/        # Separate upstream Anthropic repository
├── pipeline/
│   ├── 1_data_generation/        # Response generation and J-Lens readouts
│   ├── 2_EDA_and_FE/             # Secret-rank and Top-k/SVD features
│   ├── 3_model_predictions/      # M1-M4 classifiers
│   ├── pipeline_main.py          # Current end-to-end configuration
│   └── validation.py             # Shared category-grouped folds
├── notebooks/                    # Exploratory and presentation analyses
├── outputs/j-lens-run/           # Generated JSONL readouts
└── scripts/
    ├── finetuning/               # LoRA training/evaluation
    └── jlensfitting/             # Lens fitting for the LoRA model
```

The Anthropic Jacobian Lens checkout is included as a pinned Git submodule.

## Setup

Clone the project together with the Anthropic submodule, install Git LFS, and create a Python 3.11 virtual environment. The Python dependencies used for the recorded run are listed in `requirements.txt`.

The scripts download `Qwen/Qwen3.5-4B` from Hugging Face when it is not
cached locally. The Anthropic checkout is a pinned Git submodule, and large
run files and model artifacts use Git LFS.

## Run the current pipeline

With an existing JSONL configured in `pipeline/pipeline_main.py`:

```bash
.venv/bin/python pipeline/pipeline_main.py
```

This trains M1 (secret-rank models), M2 (Top-k/SVD), M3 (soft vote), and M4
(Top-k/SVD plus rank features). Validation uses shared `GroupKFold` splits by
injection category; the configured final holdout remains untouched unless
explicitly enabled.
