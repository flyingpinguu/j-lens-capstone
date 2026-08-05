"""Stage 3 -- multitoken leakage models.

Runs one classifier per prompt position using either Top-k/SVD features or
Top-k/SVD plus the compact secret-rank features from Stage 2.1.  All
feature fitting happens inside category-grouped folds.  The stage writes
only metrics, row-level predictions, and one bundle of final models.
"""

from datetime import datetime, timezone
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


FEATURE_MODES = {
    "topk_only": ("topk_only",),
    "topk_plus_rank": ("topk_plus_rank",),
    "both": ("topk_only", "topk_plus_rank"),
}


def _make_model(C, random_state):
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=C,
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=random_state,
        ),
    )


def _safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan


def _fit_features(feature_module, dataset, position_index, train, test,
                  rank_values, include_rank, feature_cfg):
    train_topk, transform = feature_module.fit_topk_transform(
        dataset["token_ids"][train, position_index],
        dataset["logits"][train, position_index],
        feature_cfg,
    )
    test_topk = feature_module.transform_topk(
        dataset["token_ids"][test, position_index],
        dataset["logits"][test, position_index],
        transform,
    )
    X_train = feature_module.append_rank_features(
        train_topk, rank_values[train], include_rank
    )
    X_test = feature_module.append_rank_features(
        test_topk, rank_values[test], include_rank
    )
    return X_train, X_test, transform


def _select_C(feature_module, validation_module, dataset, position_index,
              indices, rank_values, y, groups, include_rank, cfg):
    if not cfg["optimize_hyperparameters"]:
        return cfg["C"]

    candidates = cfg["C_grid"]
    scores = {C: [] for C in candidates}
    inner_cv = validation_module.inner_group_cv(cfg["validation"])
    local_y, local_groups = y[indices], groups[indices]
    for inner_train, inner_test in inner_cv.split(
        np.zeros(len(indices)), local_y, local_groups
    ):
        train, test = indices[inner_train], indices[inner_test]
        X_train, X_test, _ = _fit_features(
            feature_module, dataset, position_index, train, test,
            rank_values, include_rank, cfg["features"],
        )
        for C in candidates:
            model = _make_model(C, cfg["random_state"])
            model.fit(X_train, y[train])
            scores[C].append(_safe_auc(y[test], model.predict_proba(X_test)[:, 1]))
    def mean_score(C):
        valid = [score for score in scores[C] if not np.isnan(score)]
        return float(np.mean(valid)) if valid else -np.inf

    return max(candidates, key=lambda C: (mean_score(C), -C))


def _evaluate_position(feature_module, validation_module, dataset,
                       position_index, indices, rank_values, y, groups, folds,
                       include_rank, cfg):
    local_y = y[indices]
    local_folds = folds[indices]
    oof_probability = np.full(len(indices), np.nan)
    oof_prediction = np.full(len(indices), -1, dtype=np.int8)
    fold_aucs, fold_Cs = [], []

    for _, outer_train, outer_test in validation_module.outer_fold_indices(local_folds):
        train, test = indices[outer_train], indices[outer_test]
        C = _select_C(
            feature_module, validation_module, dataset, position_index,
            train, rank_values, y, groups, include_rank, cfg,
        )
        X_train, X_test, _ = _fit_features(
            feature_module, dataset, position_index, train, test,
            rank_values, include_rank, cfg["features"],
        )
        model = _make_model(C, cfg["random_state"])
        model.fit(X_train, y[train])
        probability = model.predict_proba(X_test)[:, 1]
        prediction = probability >= 0.5
        oof_probability[outer_test] = probability
        oof_prediction[outer_test] = prediction
        fold_aucs.append(_safe_auc(y[test], probability))
        fold_Cs.append(C)

    return {
        "n_auc_folds": int(np.count_nonzero(~np.isnan(fold_aucs))),
        "mean_fold_auc": float(np.nanmean(fold_aucs)),
        "fold_auc_std": float(np.nanstd(fold_aucs)),
        "oof_auc": _safe_auc(local_y, oof_probability),
        "oof_accuracy": float(accuracy_score(local_y, oof_prediction)),
        "oof_balanced_accuracy": float(
            balanced_accuracy_score(local_y, oof_prediction)
        ),
        "fold_Cs": json.dumps(fold_Cs),
        "n_dev": len(indices),
    }, oof_probability, oof_prediction


def _fit_final(feature_module, validation_module, dataset, position_index,
               dev_indices, rank_values, y, groups, include_rank, cfg):
    C = _select_C(
        feature_module, validation_module, dataset, position_index,
        dev_indices, rank_values, y, groups, include_rank, cfg,
    )
    topk_features, transform = feature_module.fit_topk_transform(
        dataset["token_ids"][dev_indices, position_index],
        dataset["logits"][dev_indices, position_index],
        cfg["features"],
    )
    X = feature_module.append_rank_features(
        topk_features, rank_values[dev_indices], include_rank
    )
    model = _make_model(C, cfg["random_state"])
    model.fit(X, y[dev_indices])
    return C, transform, model


def _aligned_ranks(metadata, rank_features, columns):
    ranks = rank_features.set_index("id")
    if not ranks.index.is_unique:
        raise ValueError("Secret-rank feature ids must be unique")
    aligned = ranks.reindex(metadata["run_id"])
    if aligned[list(columns)].isna().any().any():
        missing = metadata.loc[
            aligned[list(columns)].isna().any(axis=1).to_numpy(), "run_id"
        ].tolist()
        raise ValueError(f"Missing secret-rank features for {missing[:5]}")
    return aligned[list(columns)].to_numpy(dtype=np.float32)


def run(settings, dataset, rank_features, fold_plan, feature_module, validation_module):
    """Train configured variants and return metrics, predictions, artifacts."""
    cfg = settings["multitoken"]
    if cfg["mode"] not in FEATURE_MODES:
        raise ValueError(f"mode must be one of {tuple(FEATURE_MODES)}, got {cfg['mode']}")
    modes = FEATURE_MODES[cfg["mode"]]

    metadata = validation_module.align_fold_plan(
        dataset["metadata"], fold_plan, "run_id"
    )
    y = metadata["actual_leaked"].to_numpy(dtype=np.int8)
    groups = metadata["category"].to_numpy()
    folds = metadata["fold"].to_numpy(dtype=np.int16)
    if "topk_plus_rank" in modes:
        if rank_features is None:
            raise ValueError("topk_plus_rank requires Stage-2.1 secret-rank features")
        rank_values = _aligned_ranks(metadata, rank_features, cfg["rank_features"])
    else:
        # Top-k-only runs do not depend on probes or the Stage-2.1 table.
        rank_values = np.empty((len(metadata), 0), dtype=np.float32)
    predictions = metadata.copy()
    metrics_rows, final_models = [], {}

    shared_cfg = {
        **cfg,
        "validation": settings["validation"],
        "features": {
            "vocabulary_size": cfg["vocabulary_size"],
            "min_token_frequency": cfg["min_token_frequency"],
            "svd_components": cfg["svd_components"],
            "random_state": settings["random_seed"],
        },
        "random_state": settings["random_seed"],
    }

    if cfg["pool_system_prompts"]:
        cohorts = [("pooled", np.ones(len(metadata), dtype=bool))]
    else:
        cohorts = [
            (system_id, metadata["system_id"].eq(system_id).to_numpy())
            for system_id in sorted(metadata["system_id"].unique())
        ]

    for system_id, system in cohorts:
        dev_indices = np.flatnonzero(system & metadata["split"].eq("dev").to_numpy())
        holdout_indices = np.flatnonzero(
            system & metadata["split"].eq("holdout").to_numpy()
        )

        for mode in modes:
            include_rank = mode == "topk_plus_rank"
            system_mode_rows = []
            position_models = {}
            for position_index, position_name in enumerate(dataset["position_names"]):
                result, probability, prediction = _evaluate_position(
                    feature_module, validation_module, dataset, position_index,
                    dev_indices, rank_values, y, groups, folds, include_rank, shared_cfg,
                )
                C, transform, classifier = _fit_final(
                    feature_module, validation_module, dataset, position_index,
                    dev_indices, rank_values, y, groups, include_rank, shared_cfg,
                )
                row = {
                    "system_id": system_id,
                    "feature_mode": mode,
                    "classifier": "logistic_regression",
                    "position": position_name,
                    **result,
                    "selected": False,
                    "final_C": C,
                    "holdout_auc": np.nan,
                    "holdout_accuracy": np.nan,
                }
                metrics_rows.append(row)
                system_mode_rows.append(row)

                probability_column = f"{mode}__{position_name}__probability"
                prediction_column = f"{mode}__{position_name}__predicted_leaked"
                if probability_column not in predictions:
                    predictions[probability_column] = np.nan
                    predictions[prediction_column] = pd.Series(
                        pd.array([pd.NA] * len(predictions), dtype="boolean")
                    )
                predictions.loc[dev_indices, probability_column] = probability
                predictions.loc[dev_indices, prediction_column] = prediction.astype(bool)

                if settings["validation"]["evaluate_holdout"]:
                    holdout_topk = feature_module.transform_topk(
                        dataset["token_ids"][holdout_indices, position_index],
                        dataset["logits"][holdout_indices, position_index],
                        transform,
                    )
                    X_holdout = feature_module.append_rank_features(
                        holdout_topk, rank_values[holdout_indices], include_rank
                    )
                    holdout_probability = classifier.predict_proba(X_holdout)[:, 1]
                    holdout_prediction = holdout_probability >= 0.5
                    predictions.loc[holdout_indices, probability_column] = holdout_probability
                    predictions.loc[holdout_indices, prediction_column] = holdout_prediction
                    row["holdout_auc"] = _safe_auc(
                        y[holdout_indices], holdout_probability
                    )
                    row["holdout_accuracy"] = accuracy_score(
                        y[holdout_indices], holdout_prediction
                    )

                position_models[position_name] = {
                    "position": position_name,
                    "position_index": position_index,
                    "topk_transform": transform,
                    "classifier": classifier,
                    "C": C,
                }

            best = max(system_mode_rows, key=lambda item: item["mean_fold_auc"])
            best["selected"] = True

            model_key = f"{system_id}__{mode}"
            final_models[model_key] = {
                "system_id": system_id,
                "feature_mode": mode,
                "selected_position": best["position"],
                "layers": dataset["layers"],
                "top_k": dataset["top_k"],
                "rank_features": tuple(cfg["rank_features"]) if include_rank else (),
                "position_models": position_models,
                "model_id": settings["model"]["model_id"],
                "lens_file": settings["model"]["lens_file"],
            }

            selected_column = f"{mode}__selected_position"
            if selected_column not in predictions:
                predictions[selected_column] = pd.Series(
                    pd.array([pd.NA] * len(predictions), dtype="string")
                )
            predictions.loc[system, selected_column] = best["position"]

            # Canonical M2 output used by M3: OOF probabilities for the
            # selected position on dev, final-model probabilities on holdout.
            m2_probability_column = f"m2_{mode}_probability"
            m2_prediction_column = f"m2_{mode}_predicted_leaked"
            if m2_probability_column not in predictions:
                predictions[m2_probability_column] = np.nan
                predictions[m2_prediction_column] = pd.Series(
                    pd.array([pd.NA] * len(predictions), dtype="boolean")
                )
            selected_probability_column = (
                f"{mode}__{best['position']}__probability"
            )
            available = system & predictions[selected_probability_column].notna().to_numpy()
            selected_probability = predictions.loc[
                available, selected_probability_column
            ].to_numpy()
            predictions.loc[available, m2_probability_column] = selected_probability
            predictions.loc[available, m2_prediction_column] = selected_probability >= 0.5

            print(f"{system_id} / {mode}: best={best['position']}, "
                  f"CV AUC={best['mean_fold_auc']:.3f}, C={best['final_C']}")

    metrics = pd.DataFrame(metrics_rows)
    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_file = output_dir / "multitoken_metrics.csv"
    predictions_file = output_dir / "multitoken_predictions.csv"
    models_file = output_dir / "multitoken_models.joblib"
    metrics.to_csv(metrics_file, index=False)
    predictions.to_csv(predictions_file, index=False)
    joblib.dump({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": cfg["mode"],
        "config": {
            key: value for key, value in cfg.items() if key != "output_dir"
        },
        "position_names": dataset["position_names"],
        "skipped_run_ids": dataset["skipped_run_ids"],
        "models": final_models,
    }, models_file)
    print(f"Saved 3 multitoken outputs to {output_dir}")
    return metrics, predictions, final_models
