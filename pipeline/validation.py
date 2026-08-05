"""Shared train/test split and fold definition.

Single source of truth for how data is split -- every Stage-2 analysis and
every Stage-3 model (M1-M4) must get its split/folds from here, never build
its own: the four-model comparison is only valid on one identical split
(see docs/pipeline_architecture.md, Stage 3.1).

Two levels:
  1. Holdout split -- the categories in
     settings["validation"]["holdout_categories"] are the final out-of-fold
     test set (chosen from Karin's category analysis). They must not enter
     training, tuning, or feature selection; the final holdout evaluation
     runs once, gated behind settings["validation"]["evaluate_holdout"].
  2. GroupKFold by category on the dev set -- the shared CV used for all
     development-time numbers (per-layer curves, hyperparameter search,
     M1-M4 comparison).

Team convention on the two system prompts: evaluation and reporting are
ALWAYS per system prompt. Pooled training is allowed only with system_id
as an explicit feature -- the base rates differ hugely (dev leak rate
~49% lax vs ~14% strict), so a pooled AUC partly measures system
detection instead of leak prediction. Never report a pooled overall AUC.
"""

import numpy as np
from sklearn.model_selection import GroupKFold


def split_holdout(df, validation_cfg, category_col="category"):
    """Split a per-run table into (dev_df, holdout_df) by category.

    Raises if a configured holdout category does not appear in the data --
    a typo here would silently shrink the test set."""
    holdout = set(validation_cfg["holdout_categories"])
    missing = holdout - set(df[category_col].unique())
    if missing:
        raise ValueError(f"holdout categories not present in data: {sorted(missing)}")
    holdout_mask = df[category_col].isin(holdout)
    return df[~holdout_mask].copy(), df[holdout_mask].copy()


def group_cv(validation_cfg):
    """The shared dev-set CV: GroupKFold by category. Pass
    groups=df[category_col] alongside this splitter."""
    return GroupKFold(n_splits=validation_cfg["n_splits"])


def inner_group_cv(validation_cfg):
    """Inner category folds used only when hyperparameters are optimized."""
    return GroupKFold(n_splits=validation_cfg["inner_splits"])


def make_fold_plan(df, validation_cfg, id_col="run_id", category_col="category"):
    """Create the one run-level split/fold table consumed by every model.

    Build this from the common M1/M2 row universe.  That matters because a
    model-specific call to GroupKFold can otherwise change category-to-fold
    assignments when one feature extractor drops rows.
    """
    required = {id_col, category_col}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"fold-plan input is missing columns: {sorted(missing)}")
    if not df[id_col].is_unique:
        raise ValueError(f"{id_col} must be unique when building the fold plan")

    plan = df[[id_col, category_col]].rename(
        columns={id_col: "run_id", category_col: "category"}
    ).copy()
    holdout = set(validation_cfg["holdout_categories"])
    plan["split"] = np.where(plan["category"].isin(holdout), "holdout", "dev")
    plan["fold"] = -1

    dev = plan[plan["split"].eq("dev")].sort_values(
        ["category", "run_id"]
    )
    splitter = group_cv(validation_cfg)
    for fold, (_, test) in enumerate(
        splitter.split(np.zeros(len(dev)), groups=dev["category"])
    ):
        plan.loc[dev.index[test], "fold"] = fold

    if (plan.loc[plan["split"].eq("dev"), "fold"] < 0).any():
        raise RuntimeError("not every dev run received a fold")
    folds_per_category = plan[plan["split"].eq("dev")].groupby("category")["fold"].nunique()
    if not folds_per_category.eq(1).all():
        raise RuntimeError("a category was split across multiple folds")
    return plan.reset_index(drop=True)


def align_fold_plan(df, fold_plan, id_col):
    """Return ``df`` ordered to the shared plan, with split/fold attached."""
    if not df[id_col].is_unique:
        raise ValueError(f"{id_col} must be unique when aligning the fold plan")
    aligned = df.set_index(id_col).reindex(fold_plan["run_id"])
    if aligned.isna().all(axis=1).any():
        missing = fold_plan.loc[aligned.isna().all(axis=1).to_numpy(), "run_id"]
        raise ValueError(f"features missing for run ids: {missing.head().tolist()}")
    aligned = aligned.reset_index().rename(columns={id_col: "run_id"})
    for column in ("category", "split", "fold"):
        if column == "category" and column in aligned:
            if not aligned[column].reset_index(drop=True).equals(
                fold_plan[column].reset_index(drop=True)
            ):
                raise ValueError("category differs between features and fold plan")
        else:
            aligned[column] = fold_plan[column].to_numpy()
    return aligned


def outer_fold_indices(fold_values):
    """Yield shared outer train/test indices for an aligned dev table."""
    fold_values = np.asarray(fold_values)
    if (fold_values < 0).any():
        raise ValueError("outer_fold_indices expects dev rows only")
    for fold in sorted(np.unique(fold_values)):
        yield int(fold), np.flatnonzero(fold_values != fold), np.flatnonzero(fold_values == fold)
