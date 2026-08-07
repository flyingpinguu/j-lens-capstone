"""Stage 3 -- combined model-comparison table (M1-M4, pooled + per system).

Renders the per-system metrics collected from M1/M2/M3/M4 into one
at-a-glance PNG, next to the Stage-2.1 figures in
<analysis_output_dir>/<active_model>/ -- the CSVs under outputs/pipeline/
carry the same numbers but are unreadable as a quick overview.

Convention reminder (see pipeline/validation.py): "pooled" rows summarize a
model trained across both system prompts and are shown for reference only
-- they are inflated by system-prompt detection (base leak rates differ
~49% vs ~14%), so the per-system rows are what actually decides which
model/feature-set is best.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = [
    "model", "system_id", "n_dev", "n_leaks",
    "mean_fold_auc", "fold_auc_std", "oof_auc", "oof_balanced_accuracy",
]
COLUMN_LABELS = {
    "system_id": "cohort", "n_dev": "runs", "n_leaks": "leaks",
    "mean_fold_auc": "mean fold AUC", "fold_auc_std": "fold AUC std",
    "oof_auc": "OOF AUC", "oof_balanced_accuracy": "OOF bal. accuracy",
}
# Internal probability-column names (from validation.per_system_metrics)
# mapped to the architecture's M1-M4 labels for display only -- e.g.
# "m2_topk_plus_rank" is M4 in docs/pipeline_architecture.md, not a variant
# of M2. Unmapped names fall through unchanged.
MODEL_LABELS = {
    "m1_logreg": "M1_logreg",
    "m1_xgb": "M1_xgb",
    "m2_topk_only": "M2",
    "m2_topk_plus_rank": "M4",
    "m3": "M3",
}


def save(per_system_tables, out_path, title="Stage-3 model comparison (dev split, shared folds)"):
    """per_system_tables: list of DataFrames as returned by
    validation.per_system_metrics (one call per model/mode: M1's two
    variants, M2, M4, M3, ...) -- each already carries the correct "model"
    name. Concatenates, sorts pooled/lax/strict together per model, and
    renders one table image."""
    combined = pd.concat(per_system_tables, ignore_index=True)
    # Logical M1->M4 order (not alphabetical on internal names, which would
    # sort M4's "m2_topk_plus_rank" ahead of M3's "m3").
    model_order = {name: rank for rank, name in enumerate([
        "m1_logreg", "m1_xgb", "m2_topk_only", "m3", "m2_topk_plus_rank",
    ])}
    cohort_order = {"pooled": 0, "sys_lax": 1, "sys_strict": 2}
    combined["_model_sort"] = combined["model"].map(model_order).fillna(99)
    combined["_cohort_sort"] = combined["system_id"].map(cohort_order).fillna(99)
    combined = combined.sort_values(["_model_sort", "_cohort_sort"]).drop(
        columns=["_model_sort", "_cohort_sort"]
    )

    display = combined[COLUMNS].copy()
    display["model"] = display["model"].map(lambda name: MODEL_LABELS.get(name, name))
    display["mean_fold_auc"] = display.apply(
        lambda r: f"{r['mean_fold_auc']:.3f} ± {r['fold_auc_std']:.3f}", axis=1
    )
    display = display.drop(columns="fold_auc_std")
    display["oof_auc"] = display["oof_auc"].round(3)
    display["oof_balanced_accuracy"] = display["oof_balanced_accuracy"].round(3)
    display = display.rename(columns={**COLUMN_LABELS, "mean_fold_auc": "mean fold AUC ± std"})

    # Bold the best mean_fold_auc per cohort (pooled / sys_lax / sys_strict)
    # across all models -- the per-system rows are what should drive model
    # choice, per the pooling convention above.
    best_rows = set(
        combined.reset_index(drop=True).groupby("system_id")["mean_fold_auc"].idxmax()
    )

    fig, ax = plt.subplots(figsize=(11, 0.6 + 0.42 * len(display)))
    ax.axis("off")
    table = ax.table(
        cellText=display.values, colLabels=display.columns,
        loc="center", cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9.5)
    table.scale(1, 1.4)
    table.auto_set_column_width(col=list(range(len(display.columns))))
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#d5d4cf")
        if row == 0:
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#eeede8")
        elif row - 1 in best_rows:
            cell.set_text_props(weight="bold")
        if row > 0 and display.iloc[row - 1]["cohort"] == "pooled":
            cell.set_facecolor("#f5f4f0")
    ax.set_title(
        f"{title}\n(pooled rows shaded -- inflated by system-prompt detection, see validation.py)",
        fontsize=10.5, pad=14,
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)
    return combined
