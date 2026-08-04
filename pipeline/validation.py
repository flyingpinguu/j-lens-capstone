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
