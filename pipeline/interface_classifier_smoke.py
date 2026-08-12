"""Generate a tiny multi-turn dataset and fit a deployable test classifier.

This is intentionally a backend prototype, not an evaluation run. It trains
one M2-style Top-k-only logistic regression on all generated rows and reports
training-set diagnostics only. No claim about generalization should be made.

Run from the repository root:
    .venv/bin/python pipeline/interface_classifier_smoke.py
"""

from copy import deepcopy
import json

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from pipeline_main import ROOT, SETTINGS, load_stage, read_jsonl


TEMPLATE_IDS = [
    "direct_override_01",
    "persona_adoption_01",
    "authority_impersonation_01",
    "prompt_leak_01",
    "format_trick_01",
    "control_001",
    "admin_request_01",
]
DATA_DIR = ROOT / "tmp" / "interface-classifier-data" / "v1"
MODEL_FILE = ROOT / "outputs" / "pipeline" / "interface_classifier.joblib"


def main():
    settings = deepcopy(SETTINGS)
    settings.update({
        "collection_mode": "multi_turn",
        "max_attack_attempts": 5,
        "template_ids": TEMPLATE_IDS,
        "max_prompts_per_strictness": None,
        "readout_positions": "last_n",
        "readout_last_n": 16,
        "max_new_tokens": 48,
        "attacker_max_new_tokens": 48,
        "output_dir": DATA_DIR,
    })
    # Small datasets need a permissive vocabulary threshold and compact SVD.
    settings["multitoken"].update({
        "mode": "topk_only",
        "n_prompt_positions": 16,
        "layers": list(range(16, 32)),
        "top_k": 10,
        "vocabulary_size": 80,
        "min_token_frequency": 1,
        "svd_components": 8,
    })

    injections = read_jsonl(
        ROOT / "data" / "evaluation" / "conversation_seeds.jsonl"
    )
    system_prompts = read_jsonl(
        ROOT / "data" / "evaluation" / "system_prompts_authz.jsonl"
    )
    harness = load_stage("1_data_generation/run_harness.py")
    run_file = harness.run(settings, injections, system_prompts)

    feature_module = load_stage("2_EDA_and_FE/top-k_token_analysis.py")
    dataset = feature_module.run(settings, run_file)
    metadata = dataset["metadata"]
    y = metadata["actual_leaked"].to_numpy(dtype=np.int8)
    if len(np.unique(y)) != 2:
        raise ValueError(
            "The smoke run produced only one target class; select more starters "
            "before fitting the prototype classifier."
        )

    # Backend prototype: predict from the final position immediately before
    # defender generation. The other 15 positions remain available in JSONL.
    position_index = len(dataset["position_names"]) - 1
    feature_cfg = {
        "vocabulary_size": settings["multitoken"]["vocabulary_size"],
        "min_token_frequency": settings["multitoken"]["min_token_frequency"],
        "svd_components": settings["multitoken"]["svd_components"],
        "random_state": settings["random_seed"],
    }
    X, transform = feature_module.fit_topk_transform(
        dataset["token_ids"][:, position_index],
        dataset["logits"][:, position_index],
        feature_cfg,
    )
    classifier = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=0.1,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=settings["random_seed"],
        ),
    ).fit(X, y)
    probability = classifier.predict_proba(X)[:, 1]

    bundle = {
        "schema_version": 1,
        "purpose": "interface_backend_prototype_not_for_evaluation",
        "model_type": "m2_topk_only_logistic_regression",
        "model_id": settings["model"]["model_id"],
        "lens_file": settings["model"]["lens_file"],
        "readout_positions": settings["readout_positions"],
        "n_prompt_positions": settings["multitoken"]["n_prompt_positions"],
        "position_name": dataset["position_names"][position_index],
        "position_index": position_index,
        "layers": dataset["layers"],
        "top_k": dataset["top_k"],
        "topk_transform": transform,
        "classifier": classifier,
        "training_summary": {
            "source_file": str(run_file),
            "template_ids": TEMPLATE_IDS,
            "n_rows": int(len(y)),
            "n_leaks": int(y.sum()),
            "n_non_leaks": int((y == 0).sum()),
            "training_auc": float(roc_auc_score(y, probability)),
            "training_accuracy": float(accuracy_score(y, probability >= 0.5)),
        },
    }
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_FILE)

    # Reload and transform once: this catches incomplete bundles before the
    # backend tries to use them on another process or machine.
    loaded = joblib.load(MODEL_FILE)
    X_reload = feature_module.transform_topk(
        dataset["token_ids"][:, position_index],
        dataset["logits"][:, position_index],
        loaded["topk_transform"],
    )
    reloaded_probability = loaded["classifier"].predict_proba(X_reload)[:, 1]
    if not np.allclose(probability, reloaded_probability):
        raise RuntimeError("Reloaded interface classifier changed its predictions")

    print("Saved:", MODEL_FILE)
    print(json.dumps(bundle["training_summary"], indent=2))


if __name__ == "__main__":
    main()
