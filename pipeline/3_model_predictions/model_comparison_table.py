"""Stage 3 -- combined model-comparison table (M1-M4, per reporting cohort).

Renders the per-cohort metrics collected from M1/M2/M3/M4 into one
at-a-glance PNG, next to the Stage-2.1 figures in
<analysis_output_dir>/<active_model>/ -- the CSVs under outputs/pipeline/
carry the same numbers but are unreadable as a quick overview.

Cohort reminder (see pipeline/validation.py): every row is a `sys_strict`
number now. "all" covers every dev run; "unauthorized" drops the admin /
correct-password runs, whose disclosures are not leaks by definition --
that is the cohort where the M1 lens+metadata feature set has to earn its
lift instead of reading the label off `authorized`.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = [
    "model", "cohort", "n_dev", "n_leaks",
    "mean_fold_auc", "fold_auc_std", "oof_auc", "oof_balanced_accuracy",
]
COLUMN_LABELS = {
    "n_dev": "runs", "n_leaks": "leaks",
    "mean_fold_auc": "mean fold AUC", "fold_auc_std": "fold AUC std",
    "oof_auc": "OOF AUC", "oof_balanced_accuracy": "OOF bal. accuracy",
}
# Internal probability-column names (from validation.per_cohort_metrics)
# mapped to the architecture's M1-M4 labels for display only -- e.g.
# "m2_topk_plus_rank" is M4 in docs/pipeline_architecture.md, not a variant
# of M2. M1 names carry their feature set after "__". Unmapped names fall
# through unchanged.
MODEL_LABELS = {
    "m1_logreg__lens": "M1_logreg (lens)",
    "m1_logreg__lens_meta": "M1_logreg (lens+meta)",
    "m1_xgb__lens": "M1_xgb (lens)",
    "m1_xgb__lens_meta": "M1_xgb (lens+meta)",
    "m2_topk_only": "M2",
    "m2_topk_plus_rank": "M4",
    "m3": "M3",
}


DEFAULT_NOTE = (
    "(shaded 'all' rows include authorized runs, whose disclosures are "
    "non-leaks by definition -- see validation.py)"
)


def save(per_cohort_tables, out_path,
         title="Stage-3 model comparison (dev split, shared folds)",
         note=DEFAULT_NOTE):
    """per_cohort_tables: list of DataFrames as returned by
    validation.per_cohort_metrics (one call per model/mode: M1's variant x
    feature-set grid, M2, M4, M3, ...) -- each already carries the correct
    "model" name. Concatenates, sorts the cohorts of one model together,
    and renders one table image."""
    combined = pd.concat(per_cohort_tables, ignore_index=True)
    # Logical M1->M4 order (not alphabetical on internal names, which would
    # sort M4's "m2_topk_plus_rank" ahead of M3's "m3").
    model_order = {name: rank for rank, name in enumerate([
        "m1_logreg__lens", "m1_logreg__lens_meta",
        "m1_xgb__lens", "m1_xgb__lens_meta",
        "m2_topk_only", "m3", "m2_topk_plus_rank",
    ])}
    cohort_order = {
        "all": 0, "unauthorized": 1, "authorized": 2,          # headline cohorts
        "attack_only": 3, "attack_attempt_1": 4,               # leakage audit
        "attack_attempt_2plus": 5,
        "sys_lax": 6, "sys_strict": 7,                         # legacy
    }
    combined["_model_sort"] = combined["model"].map(model_order).fillna(99)
    combined["_cohort_sort"] = combined["cohort"].map(cohort_order).fillna(99)
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

    # Bold the best mean_fold_auc per cohort across all models. Cohorts
    # whose fold AUCs are all NaN (too few leaks to score any fold) have no
    # winner and are simply left unbolded.
    scored = combined.reset_index(drop=True).dropna(subset=["mean_fold_auc"])
    best_rows = set(scored.groupby("cohort")["mean_fold_auc"].idxmax())

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
        if row > 0 and display.iloc[row - 1]["cohort"] == "all":
            cell.set_facecolor("#f5f4f0")
    ax.set_title(f"{title}\n{note}", fontsize=10.5, pad=14)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)
    return combined
