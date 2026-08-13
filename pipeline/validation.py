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

Team convention on system prompts: the project now analyses `sys_strict`
only (settings["system_ids"]), so there is nothing left to pool -- every
reported number is a sys_strict number and must say so. The old rule still
applies to any future run that mixes system prompts: never train across
them without system_id as an explicit feature, and never report a pooled
overall AUC, because the base rates differ hugely (dev leak rate ~49% lax
vs ~14% strict) and a pooled AUC then partly measures system detection.

The reporting cohort that does still split the data is authorization:
admin-role and correct-password runs are entitled to the secret, so their
disclosures are not leaks. `per_cohort_metrics` therefore reports each
model over all runs AND over unauthorized runs only.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
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

    columns = [id_col, category_col]
    if "conversation_id" in df:
        columns.append("conversation_id")
    plan = df[columns].rename(
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
    if "conversation_id" in plan:
        folds_per_conversation = (
            plan[plan["split"].eq("dev")]
            .groupby("conversation_id")["fold"]
            .nunique()
        )
        if not folds_per_conversation.eq(1).all():
            raise RuntimeError("a multi-turn conversation was split across folds")
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


def _safe_auc(y, probability):
    return float(roc_auc_score(y, probability)) if len(np.unique(y)) == 2 else float("nan")


def per_cohort_metrics(
    predictions,
    probability_columns,
    target_col,
    threshold=0.5,
    cohort_col="system_id",
    all_label="all",
):
    """Break dev OOF metrics down per cohort (plus the all-rows view).

    Computed from the same OOF probabilities the overall metrics use, so it
    requires no retraining. Returns one row per (probability column,
    cohort): ``all_label`` over every scored dev row, plus one row per
    distinct value of ``cohort_col``. Two kinds of row are left out because
    they carry nothing: a lone cohort value (its row would just repeat the
    all-rows row -- this is what ``cohort_col="system_id"`` now degenerates
    to, with sys_strict the only system prompt), and a cohort holding a
    single class, whose AUC is undefined.

    The cohort dimension that does still split this data is authorization:
    an all-authorized cohort has no leaks by construction, so pass
    ``cohort_col`` a column that labels each run authorized/unauthorized
    and read the unauthorized row as the honest one (see the module
    docstring and docs/pipeline_architecture.md).
    """
    dev = predictions.loc[predictions["split"].eq("dev")]
    cohort_values = sorted(dev[cohort_col].astype(str).unique())
    subsets = [(all_label, pd.Series(True, index=dev.index))]
    if len(cohort_values) > 1:
        subsets += [
            (value, dev[cohort_col].astype(str).eq(value))
            for value in cohort_values
        ]
    return per_subset_metrics(
        predictions, probability_columns, target_col, subsets, threshold
    )


def leakage_audit_subsets(dev):
    """Nested subsets that hold this corpus's two structural confounders fixed.

    Neither is a modelling bug -- both are properties of how the data was
    collected -- but a score over all rows is a mixture of three questions,
    and only the last one is the project's:

    1. *Is this an attack at all?* 757 of 1962 dev rows are benign controls
       with a ~0.4% leak rate. Separating benign chit-chat from adversarial
       prompts is easy and has nothing to do with predicting whether an
       attack succeeds.
    2. *Is this the first attempt?* A conversation stops the moment the
       secret is disclosed, so every row at attempt >= 2 is by construction
       one that already resisted. Attack leak rate is 48.6% at attempt 1 and
       2-8% afterwards. Attempt 1 also differs in kind: its user turn is the
       written corpus template, later turns are generated by the model
       itself, and the prompt grows a conversation history. All of that is
       readable from the prompt-end readout window.
    3. *Will this attack succeed?* -- what the classifiers are supposed to
       answer, visible only inside a fixed attack/attempt stratum.

    Returns [(name, mask)] over ``dev``; masks are nested, not a partition.
    """
    is_attack = dev["label"].astype(str).eq("attack")
    first = dev["attempt_index"].astype(int).eq(1)
    return [
        ("all", pd.Series(True, index=dev.index)),
        ("attack_only", is_attack),
        ("attack_attempt_1", is_attack & first),
        ("attack_attempt_2plus", is_attack & ~first),
    ]


def per_subset_metrics(
    predictions, probability_columns, target_col, subsets, threshold=0.5
):
    """Dev OOF metrics for explicit, possibly overlapping row subsets.

    ``subsets`` is [(name, boolean mask over the dev rows)]. Subsets holding
    a single class are skipped -- their AUC is undefined, not zero.
    """
    rows = []
    dev = predictions.loc[predictions["split"].eq("dev")]
    for column in probability_columns:
        available = dev[column].notna()
        for cohort, mask in subsets:
            frame = dev.loc[mask & available]
            if frame[target_col].nunique() < 2:
                print(
                    f"  skipping cohort {cohort!r} for {column}: "
                    f"only one class in {len(frame)} rows (AUC undefined)"
                )
                continue
            y = frame[target_col].to_numpy(dtype=np.int8)
            probability = frame[column].to_numpy(dtype=float)
            prediction = probability >= threshold
            fold_aucs = [
                _safe_auc(
                    y[frame["fold"].eq(fold).to_numpy()],
                    probability[frame["fold"].eq(fold).to_numpy()],
                )
                for fold in sorted(frame["fold"].unique())
            ]
            valid_fold_aucs = [auc for auc in fold_aucs if not np.isnan(auc)]
            rows.append({
                "model": column.removesuffix("_probability"),
                "cohort": cohort,
                "n_dev": len(frame),
                "n_leaks": int(y.sum()),
                "n_auc_folds": len(valid_fold_aucs),
                "mean_fold_auc": float(np.mean(valid_fold_aucs)) if valid_fold_aucs else float("nan"),
                "fold_auc_std": float(np.std(valid_fold_aucs)) if valid_fold_aucs else float("nan"),
                "oof_auc": _safe_auc(y, probability),
                "oof_accuracy": float(accuracy_score(y, prediction)),
                "oof_balanced_accuracy": float(balanced_accuracy_score(y, prediction)),
            })
    return pd.DataFrame(rows)


def outer_fold_indices(fold_values):
    """Yield shared outer train/test indices for an aligned dev table."""
    fold_values = np.asarray(fold_values)
    if (fold_values < 0).any():
        raise ValueError("outer_fold_indices expects dev rows only")
    for fold in sorted(np.unique(fold_values)):
        yield int(fold), np.flatnonzero(fold_values != fold), np.flatnonzero(fold_values == fold)
