# Interface classifier handover

The demo backend can use the small pretrained leakage classifier at:

`outputs/pipeline/interface_classifier.joblib`

It is generated reproducibly by:

```bash
.venv/bin/python pipeline/interface_classifier_smoke.py
```

## What the prototype does

- A user can make up to five attempts in one conversation.
- The defender receives the complete conversation history on every turn.
- Each conversation has its own reproducible one-token secret.
- The J-Lens tracks that secret's rank, but this particular interface model
  deliberately does **not** use it.
- The classifier is an M2-style Top-k-only logistic regression, so the model
  remains secret-agnostic.
- It predicts whether the defender's immediately following response will
  disclose the secret.

This is only a functional backend prototype trained on a handful of rows.
Its score is not a meaningful estimate of real performance.

## Required live readout

Load the bundle with:

```python
import joblib

bundle = joblib.load("outputs/pipeline/interface_classifier.joblib")
```

For the current user turn, render the full defender prompt including the
conversation history and generation scaffolding. Before generating the
defender response, compute the J-Lens Top-10 readout at the final prompt
token for layers 16 through 31.

The bundle records the exact contract:

```python
bundle["position_name"]   # prompt_end_minus_01
bundle["layers"]          # 16..31
bundle["top_k"]           # 10
bundle["topk_transform"]  # fitted vocabulary + SVD
bundle["classifier"]      # fitted logistic regression pipeline
```

Arrange the live values as arrays with shapes:

```text
token_ids: (1, 16 layers, 10 ranks)
logits:    (1, 16 layers, 10 ranks)
```

Then reuse the pipeline transform:

```python
import importlib.util
from pathlib import Path

path = Path("pipeline/2_EDA_and_FE/top-k_token_analysis.py")
spec = importlib.util.spec_from_file_location("topk_features", path)
topk_features = importlib.util.module_from_spec(spec)
spec.loader.exec_module(topk_features)

X = topk_features.transform_topk(
    token_ids,
    logits,
    bundle["topk_transform"],
)
leak_probability = float(bundle["classifier"].predict_proba(X)[0, 1])
predicted_leak = leak_probability >= 0.5
```

The output can be shown in the interface after each user message. Do not pass
the attempt number, category, authorization state, secret token id, or secret
rank to this classifier.

## Relevant implementation files

- `pipeline/interface_classifier_smoke.py` — tiny dataset generation and training
- `pipeline/1_data_generation/run_harness.py` — conversation, generation, J-Lens readouts, leak label
- `pipeline/2_EDA_and_FE/top-k_token_analysis.py` — fitted Top-k vocabulary/SVD transform
- `pipeline/pipeline_main.py` — complete multi-turn M1–M4 configuration

Delete this handover file after the other agent has read it.
