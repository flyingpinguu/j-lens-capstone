"""Stage 3 -- M1: leakage models on the single-token feature bank.

Two variants on identical folds, both reported (the linear-vs-nonlinear
comparison is itself a result):
  - m1_logreg: standardized logistic regression, C tuned per fold
  - m1_xgb:    XGBoost, small grid tuned per fold

Per-system-prompt models on the dev split (see the convention in
pipeline/validation.py: evaluation is always per system prompt). Nested CV:
the outer loop is the shared GroupKFold from validation.py; inside each
outer training fold a GridSearchCV (also grouped by category) picks the
hyperparameters -- no hyperparameter ever sees its own test fold
(Stage 3.2 in docs/pipeline_architecture.md).

Feature-bank note: the bank (12 band x position aggregates, built in Stage
2.1) is fixed; which features matter is decided by regularization /
tree splits inside each training fold -- honest selection, not manual
picking on dev scores.

Outputs to <analysis_output_dir>/<active_model>/:
  - M1_metrics.csv          -- per variant x system: ROC-AUC / accuracy /
                               balanced accuracy (mean +/- std over folds)
  - M1_oof_predictions.csv  -- out-of-fold leak probabilities per run, one
                               column per variant (m1_logreg_proba,
                               m1_xgb_proba). This is M1's hand-off to the
                               M3 soft-voting ensemble; M2 must produce the
                               same shape on the same folds. Probabilities
                               are uncalibrated (logreg is reasonably
                               calibrated by construction, XGBoost less so
                               -- with only ~73 strict positives a
                               calibration fit would be noisier than the
                               bias it removes, revisit if M3 suffers).
"""

import matplotlib
matplotlib.use("Agg")  # headless: figures are saved, never shown
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier


def build_variants(cfg, y_train_pos_ratio):
    """Estimator + parameter grid per variant. Built per system because
    scale_pos_weight depends on the class balance."""
    logreg = Pipeline([
        ("scaling", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000, class_weight=cfg["class_weight"])),
    ])
    logreg_grid = {"model__C": cfg["C_grid"]}

    # Trees need no scaling; scale_pos_weight covers the imbalance.
    xgb = XGBClassifier(
        n_estimators=cfg["xgb"]["n_estimators"],
        subsample=cfg["xgb"]["subsample"],
        colsample_bytree=cfg["xgb"]["colsample_bytree"],
        eval_metric="logloss",
        random_state=0,
    )
    xgb_grid = {
        "max_depth": cfg["xgb"]["max_depth_grid"],
        "learning_rate": cfg["xgb"]["learning_rate_grid"],
        "scale_pos_weight": [1.0, y_train_pos_ratio],
    }
    return {"m1_logreg": (logreg, logreg_grid), "m1_xgb": (xgb, xgb_grid)}


def save_metrics_table(metrics_df, out_path):
    """Readable results table as PNG next to the other figures -- the CSV
    keeps the full detail (incl. per-fold hyperparameters), this is the
    at-a-glance view. Best AUC per system is bolded."""
    display = metrics_df.copy()
    display["ROC-AUC (mean ± std)"] = display.apply(
        lambda r: f"{r['auc_mean']:.3f} ± {r['auc_std']:.3f}", axis=1
    )
    display = display[[
        "model", "system_id", "n_runs", "n_leaks",
        "ROC-AUC (mean ± std)", "accuracy_mean", "balanced_accuracy_mean",
    ]].rename(columns={
        "system_id": "system", "n_runs": "runs", "n_leaks": "leaks",
        "accuracy_mean": "accuracy", "balanced_accuracy_mean": "bal. accuracy",
    })

    best_rows = set(metrics_df.groupby("system_id")["auc_mean"].idxmax())

    fig, ax = plt.subplots(figsize=(10, 0.6 + 0.45 * len(display)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values, colLabels=display.columns,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    table.auto_set_column_width(col=list(range(len(display.columns))))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d4cf")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeede8")
        elif row - 1 in best_rows:
            cell.set_text_props(weight="bold")
    ax.set_title(
        "M1 -- single-token feature bank (dev split, shared GroupKFold by category)",
        fontsize=11, pad=16,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def run(settings, dev_features, validation):
    """Train/evaluate both M1 variants on the dev split; returns the
    metrics DataFrame."""
    cfg = settings["m1"]
    out_dir = settings["analysis_output_dir"] / settings["active_model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    feature_cols = cfg["features"]

    metric_rows = []
    oof_frames = []
    for system_id, subset in dev_features.groupby("system_id"):
        subset = subset.reset_index(drop=True)
        X = subset[feature_cols]
        y = subset["attack_successful"].to_numpy()
        groups = subset["category"].to_numpy()
        pos_ratio = float((y == 0).sum() / max(1, (y == 1).sum()))
        variants = build_variants(cfg, pos_ratio)

        oof = subset[["id", "system_id", "category", "attack_successful"]].copy()
        oof["fold"] = -1
        for variant_name, (estimator, grid) in variants.items():
            outer = validation.group_cv(settings["validation"])
            proba_oof = np.full(len(subset), np.nan)
            fold_metrics = []
            best_params = []
            for fold, (train_idx, test_idx) in enumerate(outer.split(X, y, groups)):
                search = GridSearchCV(
                    estimator, grid,
                    cv=GroupKFold(n_splits=cfg["inner_splits"]),
                    scoring="roc_auc",
                )
                search.fit(X.iloc[train_idx], y[train_idx], groups=groups[train_idx])
                best_params.append(search.best_params_)

                proba = search.predict_proba(X.iloc[test_idx])[:, 1]
                proba_oof[test_idx] = proba
                oof.loc[test_idx, "fold"] = fold

                y_test = y[test_idx]
                if len(np.unique(y_test)) < 2:
                    print(f"  {system_id} {variant_name} fold {fold}: single-class test fold, no AUC")
                    fold_auc = np.nan
                else:
                    fold_auc = roc_auc_score(y_test, proba)
                pred = proba >= 0.5
                fold_metrics.append({
                    "auc": fold_auc,
                    "accuracy": accuracy_score(y_test, pred),
                    "balanced_accuracy": balanced_accuracy_score(y_test, pred),
                })

            oof[f"{variant_name}_proba"] = proba_oof
            fold_df = pd.DataFrame(fold_metrics)
            metric_rows.append({
                "model": variant_name,
                "system_id": system_id,
                "n_runs": len(subset),
                "n_leaks": int(y.sum()),
                "auc_mean": round(fold_df["auc"].mean(), 3),
                "auc_std": round(fold_df["auc"].std(ddof=0), 3),
                "accuracy_mean": round(fold_df["accuracy"].mean(), 3),
                "balanced_accuracy_mean": round(fold_df["balanced_accuracy"].mean(), 3),
                "best_params_per_fold": "; ".join(
                    ", ".join(f"{k.removeprefix('model__')}={v}" for k, v in params.items())
                    for params in best_params
                ),
            })
        oof_frames.append(oof)

    metrics_df = pd.DataFrame(metric_rows)
    oof_df = pd.concat(oof_frames, ignore_index=True)

    metrics_csv = out_dir / "M1_metrics.csv"
    oof_csv = out_dir / "M1_oof_predictions.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    oof_df.to_csv(oof_csv, index=False)
    print("Saved:", metrics_csv)
    print("Saved:", oof_csv)
    save_metrics_table(metrics_df, out_dir / "M1_metrics.png")
    return metrics_df
