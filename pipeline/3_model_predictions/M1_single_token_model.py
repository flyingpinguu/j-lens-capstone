"""Stage 3 / M1 -- classifiers built on the single-token (secret-rank)
feature bank from Stage 2.1.

Every estimator variant is fitted once per *feature set*
(``settings["m1"]["feature_sets"]``), so the run compares identical models
that differ only in what they are allowed to see:

  ``lens``       -- lens readouts only, the original M1.
  ``lens_meta``  -- the same readouts plus request metadata the deployment
                    knows anyway (``user_type_is_admin``, ``authorized``).

Read the second one with care: the target is "unauthorized disclosure", and
the harness classifies a disclosure to an authorized requester as a
non-leak, so ``authorized == True`` implies ``attack_successful == False``
by construction. A large part of any lift is therefore the model learning
the *definition* of the label, not extra evidence about leak risk -- which
is why pipeline_main also reports both feature sets on the unauthorized-only
cohort, where the metadata cannot decide the label.

Models are named ``<variant>__<feature set>``. M1 is evaluated on the shared
run-to-fold plan created in validation.py. Neither category nor system_id is
a feature. The returned row-level probabilities are the input to the
parameter-free M3 soft-voting ensemble.
"""

from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import ParameterGrid
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def _build_variants(cfg, negative_positive_ratio):
    """Create the two M1 estimators and their optional search grids."""
    logreg = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=cfg["C"],
            class_weight="balanced",
            max_iter=3000,
            solver="liblinear",
            random_state=cfg["random_state"],
        ),
    )
    xgb_cfg = cfg["xgb"]
    xgb = XGBClassifier(
        n_estimators=xgb_cfg["n_estimators"],
        max_depth=xgb_cfg["max_depth"],
        learning_rate=xgb_cfg["learning_rate"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        scale_pos_weight=negative_positive_ratio,
        eval_metric="logloss",
        random_state=cfg["random_state"],
        n_jobs=xgb_cfg.get("n_jobs", 1),
    )
    return {
        "m1_logreg": (logreg, {"logisticregression__C": cfg["C_grid"]}),
        "m1_xgb": (
            xgb,
            {
                "max_depth": xgb_cfg["max_depth_grid"],
                "learning_rate": xgb_cfg["learning_rate_grid"],
            },
        ),
    }


def _fit(estimator, grid, X, y, groups, cfg, validation_module):
    """Fit one estimator, with category-grouped tuning when enabled."""
    if not cfg["optimize_hyperparameters"]:
        fitted = clone(estimator).fit(X, y)
        return fitted, {}

    scores = []
    for parameters in ParameterGrid(grid):
        fold_scores = []
        inner = validation_module.inner_group_cv(cfg["validation"])
        for train, test in inner.split(X, y, groups):
            candidate = clone(estimator).set_params(**parameters)
            candidate.fit(X[train], y[train])
            fold_scores.append(
                _safe_auc(y[test], candidate.predict_proba(X[test])[:, 1])
            )
        valid_scores = [score for score in fold_scores if not np.isnan(score)]
        mean_score = float(np.mean(valid_scores)) if valid_scores else -np.inf
        scores.append((mean_score, parameters))

    _, best_parameters = max(scores, key=lambda item: item[0])
    fitted = clone(estimator).set_params(**best_parameters).fit(X, y)
    return fitted, best_parameters


def _safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan


def run(settings, rank_features, fold_plan, validation_module):
    """Fit every (variant x feature set) M1 model and return metrics,
    aligned predictions, and the final models."""
    cfg = {
        **settings["m1"],
        "validation": settings["validation"],
        "random_state": settings["random_seed"],
    }
    aligned = validation_module.align_fold_plan(rank_features, fold_plan, "id")
    feature_sets = cfg["feature_sets"]
    for feature_set_name, feature_columns in feature_sets.items():
        missing = set(feature_columns) - set(aligned.columns)
        if missing:
            raise ValueError(
                f"M1 feature set {feature_set_name!r} is missing columns: {sorted(missing)}"
            )

    # label / attempt_index are carried for the leakage-audit strata only
    # (see validation.leakage_audit_subsets) -- never as features.
    metadata_columns = [
        column
        for column in ("run_id", "category", "system_id", "user_type", "authorized",
                       "label", "attempt_index", "conversation_id",
                       "attack_successful", "split", "fold")
        if column in aligned.columns
    ]
    predictions = aligned[metadata_columns].copy()
    dev_mask = aligned["split"].eq("dev").to_numpy()
    holdout_mask = aligned["split"].eq("holdout").to_numpy()
    dev_indices = np.flatnonzero(dev_mask)
    holdout_indices = np.flatnonzero(holdout_mask)

    # Target, groups and folds are shared by every feature set / variant --
    # only the feature matrix changes, which is what makes the comparison a
    # like-for-like one.
    y_all = aligned["attack_successful"].to_numpy(dtype=np.int8)
    groups_all = aligned["category"].to_numpy()
    y_dev = y_all[dev_indices]
    groups_dev = groups_all[dev_indices]
    folds_dev = aligned.loc[dev_mask, "fold"].to_numpy()

    available_variants = {"m1_logreg", "m1_xgb"}
    requested_variants = cfg.get("variants", sorted(available_variants))
    unknown = set(requested_variants) - available_variants
    if unknown:
        raise ValueError(f"Unknown M1 variants: {sorted(unknown)}")

    metrics_rows, final_models = [], {}
    for feature_set_name, feature_columns in feature_sets.items():
        X_all = aligned[list(feature_columns)].to_numpy(dtype=np.float32)
        X_dev = X_all[dev_indices]

        for variant_name in requested_variants:
            model_name = f"{variant_name}__{feature_set_name}"
            oof_probability = np.full(len(dev_indices), np.nan)
            fold_rows, fold_params = [], []

            for fold, train, test in validation_module.outer_fold_indices(folds_dev):
                train_ratio = float(
                    (y_dev[train] == 0).sum() / max(1, (y_dev[train] == 1).sum())
                )
                estimator, grid = _build_variants(cfg, train_ratio)[variant_name]
                fitted, best_params = _fit(
                    estimator, grid, X_dev[train], y_dev[train], groups_dev[train],
                    cfg, validation_module,
                )
                probability = fitted.predict_proba(X_dev[test])[:, 1]
                oof_probability[test] = probability
                prediction = probability >= cfg["threshold"]
                fold_balanced_accuracy = (
                    balanced_accuracy_score(y_dev[test], prediction)
                    if len(np.unique(y_dev[test])) == 2 else np.nan
                )
                fold_rows.append({
                    "fold": fold,
                    "auc": _safe_auc(y_dev[test], probability),
                    "accuracy": accuracy_score(y_dev[test], prediction),
                    "balanced_accuracy": fold_balanced_accuracy,
                })
                fold_params.append(best_params)

            probability_column = f"{model_name}_probability"
            prediction_column = f"{model_name}_predicted_leaked"
            predictions[probability_column] = np.nan
            predictions[prediction_column] = pd.Series(
                pd.array([pd.NA] * len(predictions), dtype="boolean")
            )
            predictions.loc[dev_indices, probability_column] = oof_probability
            predictions.loc[dev_indices, prediction_column] = (
                oof_probability >= cfg["threshold"]
            )

            dev_ratio = float((y_dev == 0).sum() / max(1, (y_dev == 1).sum()))
            estimator, grid = _build_variants(cfg, dev_ratio)[variant_name]
            final_model, final_params = _fit(
                estimator, grid, X_dev, y_dev, groups_dev, cfg, validation_module
            )
            holdout_auc = np.nan
            holdout_accuracy = np.nan
            if settings["validation"]["evaluate_holdout"]:
                holdout_probability = final_model.predict_proba(X_all[holdout_indices])[:, 1]
                holdout_prediction = holdout_probability >= cfg["threshold"]
                predictions.loc[holdout_indices, probability_column] = holdout_probability
                predictions.loc[holdout_indices, prediction_column] = holdout_prediction
                holdout_auc = _safe_auc(y_all[holdout_indices], holdout_probability)
                holdout_accuracy = accuracy_score(y_all[holdout_indices], holdout_prediction)

            fold_df = pd.DataFrame(fold_rows)
            oof_prediction = oof_probability >= cfg["threshold"]
            metrics_rows.append({
                "model": model_name,
                "features": feature_set_name,
                "n_features": len(feature_columns),
                "n_dev": len(dev_indices),
                "n_leaks": int(y_dev.sum()),
                "n_auc_folds": int(fold_df["auc"].notna().sum()),
                "mean_fold_auc": float(fold_df["auc"].mean()),
                "fold_auc_std": float(fold_df["auc"].std(ddof=0)),
                "oof_auc": _safe_auc(y_dev, oof_probability),
                "oof_accuracy": float(accuracy_score(y_dev, oof_prediction)),
                "oof_balanced_accuracy": float(
                    balanced_accuracy_score(y_dev, oof_prediction)
                ),
                "fold_parameters": fold_params,
                "final_parameters": final_params,
                "holdout_auc": holdout_auc,
                "holdout_accuracy": holdout_accuracy,
            })
            final_models[model_name] = {
                "feature_set": feature_set_name,
                "feature_columns": tuple(feature_columns),
                "classifier": final_model,
                "parameters": final_params,
                "model_id": settings["model"]["model_id"],
                "lens_file": settings["model"]["lens_file"],
            }
            print(f"M1 / {model_name}: mean fold AUC={fold_df['auc'].mean():.3f}")

    metrics = pd.DataFrame(metrics_rows)
    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "M1_metrics.csv", index=False)
    predictions.to_csv(output_dir / "M1_predictions.csv", index=False)
    joblib.dump({
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": {key: value for key, value in cfg.items() if key != "output_dir"},
        "models": final_models,
    }, output_dir / "M1_models.joblib")
    return metrics, predictions, final_models
