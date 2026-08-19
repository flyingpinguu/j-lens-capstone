"""Static prompt-position AUC plots with position-matched model inputs.

For every named prompt position:

* M1 uses only the mean log secret rank over the configured late-layer band.
* M2 uses only that position's Top-k/SVD features.
* M3 is an equal-weight soft vote of those position-local M1 and M2 scores.
* M4 fits Top-k/SVD plus the same position-local secret-rank feature.

This is deliberately separate from the pipeline's canonical M1--M4 summary,
whose M1 rank bank aggregates several positions. Keeping all four curves local
to the x-axis position prevents that global signal from being repeated at every
point in the position plot.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


COLORS = {
    "M1": "#e66b32",
    "M2": "#2878b5",
    "M3": "#37966f",
    "M4": "#7b5cb8",
}
LABELS = {
    "M1": "M1 · secret rank",
    "M2": "M2 · Top-k/SVD",
    "M3": "M3 · position-matched soft vote",
    "M4": "M4 · Top-k/SVD + secret rank",
}


def _safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else np.nan


def _position_number(name):
    return -int(name.rsplit("_", 1)[-1])


def _metrics(y, probability, folds, threshold=0.5):
    fold_aucs = [
        _safe_auc(y[folds == fold], probability[folds == fold])
        for fold in sorted(np.unique(folds))
    ]
    prediction = probability >= threshold
    return {
        "n_auc_folds": int(np.count_nonzero(~np.isnan(fold_aucs))),
        "mean_fold_auc": float(np.nanmean(fold_aucs)),
        "fold_auc_std": float(np.nanstd(fold_aucs)),
        "oof_auc": _safe_auc(y, probability),
        "oof_accuracy": float(accuracy_score(y, prediction)),
        "oof_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
        "n_dev": len(y),
    }


def _fit_position_m1(X, y, folds, settings, validation_module):
    """Create category-grouped OOF predictions for one position."""
    oof = np.full(len(y), np.nan)
    for _, train, test in validation_module.outer_fold_indices(folds):
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=settings["m1"]["C"],
                class_weight="balanced",
                max_iter=3000,
                solver="liblinear",
                random_state=settings["random_seed"],
            ),
        )
        model.fit(X[train], y[train])
        oof[test] = model.predict_proba(X[test])[:, 1]
    return oof


def _plot_position_models(metrics, settings, out_path):
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    for model_name in ("M1", "M2", "M3", "M4"):
        frame = metrics.loc[metrics["model"].eq(model_name)].copy()
        frame["x"] = frame["position"].map(_position_number)
        frame = frame.sort_values("x")
        x = frame["x"].to_numpy()
        mean = frame["mean_fold_auc"].to_numpy()
        std = frame["fold_auc_std"].to_numpy()
        ax.plot(
            x, mean, marker="o", linewidth=2.1, markersize=4.3,
            label=LABELS[model_name], color=COLORS[model_name],
        )
        ax.fill_between(
            x, mean - std, mean + std,
            color=COLORS[model_name], alpha=0.09, linewidth=0,
        )

    ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1.2)
    ax.text(-15.8, 0.51, "chance (AUC 0.5)", color="#77756f", fontsize=9)
    ax.set_xticks(range(-16, 0))
    ax.set_xlabel("prompt token position relative to response start")
    ax.set_ylabel("mean fold ROC-AUC")
    ax.set_title(
        f"Position-local M1–M4 leakage prediction · {settings['model']['model_id']}\n"
        f"attack-only, {', '.join(settings['system_ids'])}; ribbon = ±1 fold SD"
    )
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def run(
    settings,
    dataset,
    fold_plan,
    m2_metrics,
    m2_predictions,
    feature_module,
    multitoken_model,
    validation_module,
):
    """Fit position-local M1/M3/M4, combine with M2, and save metrics + PNG."""
    del m2_metrics  # probabilities are the authoritative source for this plot
    metadata = validation_module.align_fold_plan(
        dataset["metadata"], fold_plan, "run_id"
    )
    m2_predictions = validation_module.align_fold_plan(
        m2_predictions, fold_plan, "run_id"
    )
    layers = list(dataset["layers"])
    late_band = list(settings["analysis"]["late_band"])
    missing_layers = sorted(set(late_band) - set(layers))
    if missing_layers:
        raise ValueError(
            f"Position-specific rank band is absent from Top-k layers: {missing_layers}"
        )
    layer_indices = [layers.index(layer) for layer in late_band]
    probe_ranks = dataset["probe_ranks"][:, :, layer_indices]
    if np.isnan(probe_ranks).any():
        raise ValueError("Position-specific rank models require a probe at every cell")

    y_all = metadata["actual_leaked"].to_numpy(dtype=np.int8)
    groups_all = metadata["category"].to_numpy()
    folds_all = metadata["fold"].to_numpy(dtype=np.int16)
    dev_indices = np.flatnonzero(metadata["split"].eq("dev").to_numpy())
    y = y_all[dev_indices]
    folds = folds_all[dev_indices]

    cfg = settings["multitoken"]
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
    rows = []

    for position_index, position_name in enumerate(dataset["position_names"]):
        # One aligned rank feature: log first, then average the fixed late band.
        position_rank = np.log(probe_ranks[:, position_index]).mean(axis=1).reshape(-1, 1)
        m1_probability = _fit_position_m1(
            position_rank[dev_indices], y, folds, settings, validation_module
        )

        m2_column = f"topk_only__{position_name}__probability"
        m2_probability = m2_predictions.loc[dev_indices, m2_column].to_numpy(dtype=float)
        if np.isnan(m2_probability).any():
            raise ValueError(f"Missing OOF M2 probabilities in {m2_column}")

        # The local M3 vote combines scores from the same position only.
        m3_probability = (m1_probability + m2_probability) / 2.0

        # The local M4 is refit honestly inside every outer fold: vocabulary,
        # SVD, and classifier never see their validation rows.
        m4_result, _, _ = multitoken_model._evaluate_position(
            feature_module,
            validation_module,
            dataset,
            position_index,
            dev_indices,
            position_rank,
            y_all,
            groups_all,
            folds_all,
            True,
            shared_cfg,
        )

        common = {
            "position": position_name,
            "rank_layers": f"{late_band[0]}-{late_band[-1]}",
        }
        rows.extend([
            {"model": "M1", **common, **_metrics(y, m1_probability, folds)},
            {"model": "M2", **common, **_metrics(y, m2_probability, folds)},
            {"model": "M3", **common, **_metrics(y, m3_probability, folds)},
            {"model": "M4", **common, **{
                key: value for key, value in m4_result.items() if key != "fold_Cs"
            }},
        ])

    metrics = pd.DataFrame(rows)
    output_dir = settings["m1"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(output_dir / "position_local_metrics.csv", index=False)
    # Keep the prior M1-only hand-off name for backwards compatibility.
    metrics.loc[metrics["model"].eq("M1")].to_csv(
        output_dir / "M1_position_metrics.csv", index=False
    )

    analysis_dir = settings["analysis_output_dir"] / settings["active_model"]
    analysis_dir.mkdir(parents=True, exist_ok=True)
    image_path = analysis_dir / "auc_by_prompt_position_m1_m4.png"
    _plot_position_models(metrics, settings, image_path)
    return metrics, image_path


def save_cross_model_comparison(
    small_metrics,
    large_metrics,
    out_path,
    small_label="Qwen3.5-4B",
    large_label="Qwen3-14B",
):
    """Save a two-panel, static M1/M2 position comparison across models."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.7), sharex=True, sharey=True)
    model_frames = [
        (small_label, small_metrics, "#2878b5"),
        (large_label, large_metrics, "#e66b32"),
    ]
    for ax, model_name in zip(axes, ("M1", "M2")):
        for label, all_metrics, color in model_frames:
            frame = all_metrics.loc[all_metrics["model"].eq(model_name)].copy()
            frame["x"] = frame["position"].map(_position_number)
            frame = frame.sort_values("x")
            x = frame["x"].to_numpy()
            mean = frame["mean_fold_auc"].to_numpy()
            std = frame["fold_auc_std"].to_numpy()
            ax.plot(
                x, mean, marker="o", linewidth=2.1, markersize=4,
                label=label, color=color,
            )
            ax.fill_between(
                x, mean - std, mean + std,
                color=color, alpha=0.10, linewidth=0,
            )
        ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1.2)
        ax.set_title(LABELS[model_name])
        ax.set_xticks(range(-16, 0, 2))
        ax.set_xlabel("prompt position relative to response start")
        ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("mean fold ROC-AUC")
    axes[0].legend(frameon=False, loc="upper left")
    fig.suptitle("Position-local leakage prediction · 4B vs 14B", fontsize=14)
    fig.text(
        0.5, 0.01,
        "Descriptive comparison only: model, tokenizer-selected secrets, and fitted lens differ.",
        ha="center", fontsize=9, color="#66645f",
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)
    return out_path
