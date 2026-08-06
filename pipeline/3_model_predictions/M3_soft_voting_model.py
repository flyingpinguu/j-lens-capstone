"""Stage 3 / M3 -- parameter-free soft voting per position and across positions.

For every prompt position, M3 averages the position-independent M1
secret-rank probability and M2's Top-k/SVD probability at that position.
It then averages all per-position M3 probabilities into one stacked soft
vote. There is no trainable meta-model. IDs, categories, folds, and targets
must match before any probabilities are combined.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score


def _aligned_columns(frame, fold_plan, columns, name):
    if not frame["run_id"].is_unique:
        raise ValueError(f"{name} predictions contain duplicate run_id values")
    aligned = frame.set_index("run_id").reindex(fold_plan["run_id"])
    missing_rows = aligned[list(columns)].isna().all(axis=1)
    if missing_rows.any():
        missing = fold_plan.loc[missing_rows.to_numpy(), "run_id"].head().tolist()
        raise ValueError(f"{name} predictions are missing runs: {missing}")
    return aligned.reset_index()


def _position_names(m2_predictions, feature_mode):
    prefix = f"{feature_mode}__"
    suffix = "__probability"
    return [
        column[len(prefix):-len(suffix)]
        for column in m2_predictions.columns
        if column.startswith(prefix) and column.endswith(suffix)
    ]


def _safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan


def _metrics(y, probability, folds, threshold):
    fold_aucs = []
    for fold in sorted(np.unique(folds)):
        test = folds == fold
        fold_aucs.append(_safe_auc(y[test], probability[test]))
    prediction = probability >= threshold
    return {
        "n_auc_folds": int(np.count_nonzero(~np.isnan(fold_aucs))),
        "mean_fold_auc": float(np.nanmean(fold_aucs)),
        "fold_auc_std": float(np.nanstd(fold_aucs)),
        "oof_auc": _safe_auc(y, probability),
        "oof_accuracy": float(accuracy_score(y, prediction)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
    }


def run(settings, fold_plan, m1_predictions, m2_predictions):
    """Create one M3 prediction and metric row per prompt position."""
    cfg = settings["m3"]
    feature_mode = cfg["m2_feature_mode"]
    m1_probability_column = f"{cfg['m1_variant']}_probability"
    positions = _position_names(m2_predictions, feature_mode)
    if not positions:
        raise ValueError(f"M2 contains no per-position {feature_mode} probabilities")
    m2_probability_columns = [
        f"{feature_mode}__{position}__probability" for position in positions
    ]

    m1 = _aligned_columns(
        m1_predictions,
        fold_plan,
        [
            "category", "system_id", "attack_successful", "split", "fold",
            m1_probability_column,
        ],
        "M1",
    )
    m2 = _aligned_columns(
        m2_predictions,
        fold_plan,
        ["category", "actual_leaked", "split", "fold", *m2_probability_columns],
        "M2",
    )

    for column in ("category", "split", "fold"):
        expected = fold_plan[column].reset_index(drop=True)
        if not m1[column].reset_index(drop=True).equals(expected):
            raise ValueError(f"M1 {column} does not match the shared fold plan")
        if not m2[column].reset_index(drop=True).equals(expected):
            raise ValueError(f"M2 {column} does not match the shared fold plan")
    if not np.array_equal(
        m1["attack_successful"].to_numpy(dtype=bool),
        m2["actual_leaked"].to_numpy(dtype=bool),
    ):
        raise ValueError("M1 and M2 targets do not match")

    predictions = fold_plan.copy()
    predictions["system_id"] = m1["system_id"].to_numpy()
    predictions["actual_leaked"] = m1["attack_successful"].to_numpy(dtype=bool)
    predictions["m1_probability"] = m1[m1_probability_column].to_numpy(dtype=float)
    dev = predictions["split"].eq("dev").to_numpy()
    holdout = predictions["split"].eq("holdout").to_numpy()
    y_dev = predictions.loc[dev, "actual_leaked"].to_numpy(dtype=np.int8)
    folds_dev = predictions.loc[dev, "fold"].to_numpy(dtype=np.int16)

    metric_rows = []
    position_probability_columns = []
    for position, m2_column in zip(positions, m2_probability_columns):
        m2_probability = m2[m2_column].to_numpy(dtype=float)
        m3_probability = (predictions["m1_probability"] + m2_probability) / 2.0
        if np.isnan(m3_probability[dev]).any():
            raise ValueError(f"M1 or M2 has missing dev probabilities at {position}")

        probability_column = f"m3__{position}__probability"
        prediction_column = f"m3__{position}__predicted_leaked"
        position_probability_columns.append(probability_column)
        predictions[f"m2__{position}__probability"] = m2_probability
        predictions[probability_column] = m3_probability
        predictions[prediction_column] = pd.Series(
            pd.array([pd.NA] * len(predictions), dtype="boolean")
        )
        available = ~np.isnan(m3_probability)
        predictions.loc[available, prediction_column] = (
            m3_probability[available] >= cfg["threshold"]
        )

        row = {
            "model": "m3_soft_vote",
            "position": position,
            **_metrics(
                y_dev, m3_probability[dev], folds_dev, cfg["threshold"]
            ),
            "selected": False,
            "holdout_auc": np.nan,
            "holdout_accuracy": np.nan,
        }
        if settings["validation"]["evaluate_holdout"]:
            if np.isnan(m3_probability[holdout]).any():
                raise ValueError(f"M1 or M2 has missing holdout probabilities at {position}")
            y_holdout = predictions.loc[holdout, "actual_leaked"].to_numpy(dtype=np.int8)
            row["holdout_auc"] = _safe_auc(y_holdout, m3_probability[holdout])
            row["holdout_accuracy"] = accuracy_score(
                y_holdout, m3_probability[holdout] >= cfg["threshold"]
            )
        metric_rows.append(row)

    # Second-level soft vote: every token-position M3 classifier contributes
    # equally. This remains an OOF combination and has no fitted parameters.
    stacked_probability = predictions[position_probability_columns].mean(
        axis=1, skipna=False
    ).to_numpy(dtype=float)
    if np.isnan(stacked_probability[dev]).any():
        raise ValueError("M3 position predictions are missing for the stacked vote")
    predictions["m3_stacked_probability"] = stacked_probability
    predictions["m3_stacked_predicted_leaked"] = pd.Series(
        pd.array([pd.NA] * len(predictions), dtype="boolean")
    )
    available = ~np.isnan(stacked_probability)
    predictions.loc[available, "m3_stacked_predicted_leaked"] = (
        stacked_probability[available] >= cfg["threshold"]
    )

    stacked_row = {
        "model": "m3_stacked_soft_vote",
        "position": "all_positions",
        **_metrics(y_dev, stacked_probability[dev], folds_dev, cfg["threshold"]),
        "selected": False,
        "holdout_auc": np.nan,
        "holdout_accuracy": np.nan,
    }
    if settings["validation"]["evaluate_holdout"]:
        if np.isnan(stacked_probability[holdout]).any():
            raise ValueError("M3 stacked vote is missing holdout probabilities")
        y_holdout = predictions.loc[holdout, "actual_leaked"].to_numpy(dtype=np.int8)
        stacked_row["holdout_auc"] = _safe_auc(
            y_holdout, stacked_probability[holdout]
        )
        stacked_row["holdout_accuracy"] = accuracy_score(
            y_holdout, stacked_probability[holdout] >= cfg["threshold"]
        )
    metric_rows.append(stacked_row)

    metrics = pd.DataFrame(metric_rows)
    position_rows = metrics["model"].eq("m3_soft_vote")
    selected_index = metrics.loc[position_rows, "mean_fold_auc"].idxmax()
    metrics.loc[selected_index, "selected"] = True
    selected_position = metrics.loc[selected_index, "position"]
    predictions["m3_selected_position"] = selected_position
    predictions["m3_probability"] = predictions[
        f"m3__{selected_position}__probability"
    ]
    predictions["m3_predicted_leaked"] = predictions[
        f"m3__{selected_position}__predicted_leaked"
    ]

    output_dir = cfg["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "M3_model_comparison_metrics.csv", index=False)
    predictions.to_csv(output_dir / "M3_soft_voting_predictions.csv", index=False)
    print(
        f"M3 / soft vote: best={selected_position}, "
        f"mean fold AUC={metrics.loc[selected_index, 'mean_fold_auc']:.3f}; "
        f"stacked={stacked_row['mean_fold_auc']:.3f}"
    )
    return metrics, predictions
