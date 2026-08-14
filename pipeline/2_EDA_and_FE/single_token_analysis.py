"""Stage 2.1 -- single-token (secret-rank) analysis.

Ported from notebooks/analysis_friedrich/secret_token_analysis.ipynb. One
streaming pass over the Stage-1 output JSONL (the file is ~1GB -- never
loaded into memory at once), then:

  1. Per-layer secret-token-rank classifier curve (GroupKFold by category),
     read at the mean log-rank over the last 3 scaffolding tokens.
  2. Per-layer ROC-AUC by position offset as a heatmap (user + scaffolding
     segments, diverging around chance = 0.5).
  3. The 2.1.2 feature bank as a DataFrame -- returned to pipeline_main for
     Stage 3 and written to CSV as the track's hand-off artifact.

Scope: settings["system_ids"] restricts every output here to the listed
system prompts. The project now analyses `sys_strict` only, so the two
curves/panels that used to contrast sys_lax with sys_strict contrast the
two *authorization cohorts* instead (see cohort_frames below) -- with
admin/password rows in the corpus, "all runs" and "unauthorized runs only"
are no longer the same question.

Figures and the feature CSV go to <analysis_output_dir>/<active_model>/.
Band/window parameters come from settings["analysis"] -- they are config,
not code, so a second model can re-derive them from its own curves (see
docs/pipeline_architecture.md).
"""

import json

import matplotlib
matplotlib.use("Agg")  # headless: this script saves figures, never shows them
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

COHORT_COLORS = {"all": "#eb6834", "unauthorized": "#2a78d6"}
COHORT_LABELS = {
    "all": "all runs",
    "unauthorized": "unauthorized runs only",
}
DIVERGING_CMAP = LinearSegmentedColormap.from_list(
    "auc_diverging", ["#0d366b", "#f0efec", "#8c1f1f"]
)


def cohort_frames(df):
    """The two evaluation cohorts of the strict-only setup.

    An authorized run (admin role, or the correct access password in the
    user turn) has attack_successful == False *by construction* -- the
    harness classifies an authorized disclosure as a non-leak. Those runs
    are exactly the ones where the model does surface the secret
    internally, so leaving them in makes the curve partly answer "was this
    request authorized" instead of "did the protection fail". Both views
    are reported: "all" is the headline number over the corpus as run,
    "unauthorized" is the same analysis restricted to runs where the
    protection was actually supposed to hold.
    """
    return [("all", df), ("unauthorized", df[~df["authorized"].astype(bool)])]


def positions_by_segment(readouts):
    """User and scaffolding positions, each ordered by distance from its own
    boundary: user[0] = last user token, scaffold[0] = last scaffolding
    token. Response positions are never returned -- using them as features
    would be circular (they describe the outcome instead of predicting it)."""
    user = sorted((int(p) for p, d in readouts.items() if d["segment"] == "user"), reverse=True)
    scaffold = sorted((int(p) for p, d in readouts.items() if d["segment"] == "prompt_suffix"), reverse=True)
    return user, scaffold


def band_log_rank(layers_data, band):
    """Mean log secret-rank over a layer band at one position (adjacent
    layers are highly correlated; the band mean is the same signal with
    less noise)."""
    return float(np.mean([np.log(layers_data[str(layer)]["probe"]["rank"]) for layer in band]))


def collect(input_file, analysis_cfg, system_ids=None):
    """Single streaming pass -> everything the three outputs below need.

    ``system_ids`` (None = keep every system prompt) is applied here, so the
    feature table, the figures and every Stage-3 model downstream see the
    same run universe.
    """
    user_window_n = analysis_cfg["user_window_n"]
    peak_quantile = analysis_cfg["peak_quantile"]
    late_band = analysis_cfg["late_band"]
    mid_band = analysis_cfg["mid_band"]
    scaffold_ref_n = analysis_cfg["scaffold_ref_n"]
    heatmap_layers = analysis_cfg["heatmap_layers"]

    n_layers = None
    curve_rows = []          # graph 1: per-layer scaffold-ref mean log-rank
    offset_rows = None       # graph 2: per heatmap layer, raw ranks per offset
    feature_rows = []        # F1-F4
    n_skipped_short = 0
    n_skipped_system = 0

    with input_file.open(encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping unparseable line {line_no}")
                continue

            if system_ids is not None and row["system_id"] not in system_ids:
                n_skipped_system += 1
                continue

            if n_layers is None:
                n_layers = len(row["readout_layers"])
                offset_rows = {layer: [] for layer in heatmap_layers}

            readouts = row["readouts"]
            user_positions, scaffold_positions = positions_by_segment(readouts)
            meta = {
                "id": row["id"],
                "conversation_id": row.get("conversation_id", row["id"]),
                "attempt_index": row.get("attempt_index", 1),
                "template_id": row["template_id"],
                "label": row["label"],
                "category": row["category"],
                "system_id": row["system_id"],
                "user_type": row.get("user_type", "user"),
                "password": row.get("password", "none"),
                "authorized": bool(row.get("authorized", False)),
                # numeric twin of user_type for the lens+metadata feature
                # set; `authorized` is already 0/1-usable as a bool
                "user_type_is_admin": int(row.get("user_type", "user") == "admin"),
                "attack_successful": row["attack_successful"],
            }

            # --- graph 1: mean log-rank over the last scaffold_ref_n
            #     scaffolding tokens, per layer
            scaffold_ref = [readouts[str(p)]["layers"] for p in scaffold_positions[:scaffold_ref_n]]
            curve_row = dict(meta)
            for layer in range(n_layers):
                curve_row[f"scaffold_ref_logrank_L{layer}"] = float(np.mean(
                    [np.log(layers[str(layer)]["probe"]["rank"]) for layers in scaffold_ref]
                ))
            curve_rows.append(curve_row)

            # --- graph 2: raw ranks per position offset (log happens in the
            #     CV pipeline, as in the notebook); short runs are dropped so
            #     every row has the full offset set
            if len(user_positions) < user_window_n:
                n_skipped_short += 1
            else:
                for layer in heatmap_layers:
                    offset_row = dict(meta)
                    for offset, position in enumerate(user_positions[:user_window_n]):
                        offset_row[f"user_offset{offset}"] = readouts[str(position)]["layers"][str(layer)]["probe"]["rank"]
                    for offset, position in enumerate(scaffold_positions):
                        offset_row[f"scaffold_offset{offset}"] = readouts[str(position)]["layers"][str(layer)]["probe"]["rank"]
                    offset_rows[layer].append(offset_row)

            # --- feature bank (short runs contribute a smaller window
            #     instead of being dropped: the feature table must cover
            #     every run Stage 3 will see).
            #     The bank is the systematic band x position-aggregate grid
            #     defined ONCE and then frozen -- model-side selection
            #     happens inside the CV folds, not by editing this list.
            #     The original F1-F4 keep their documented names: F1 =
            #     late_ref_level, F2 = late_user_peak, F3 = scaffold_delta,
            #     F4 = mid_user_peak.
            ref_layers = readouts[str(scaffold_positions[0])]["layers"]
            last_user_layers = readouts[str(user_positions[0])]["layers"]
            window_layers = [readouts[str(p)]["layers"] for p in user_positions[:user_window_n]]
            feature_row = {**meta, "n_user_positions": len(user_positions)}
            for band_name, band in [("late", late_band), ("mid", mid_band)]:
                scaffold_last = band_log_rank(ref_layers, band)
                user_last = band_log_rank(last_user_layers, band)
                window = [band_log_rank(layers, band) for layers in window_layers]
                scaffold_ref_vals = [band_log_rank(layers, band) for layers in scaffold_ref]
                prefix_map = {
                    f"{band_name}_scaffold_last": scaffold_last,
                    f"{band_name}_scaffold_ref_mean": float(np.mean(scaffold_ref_vals)),
                    f"{band_name}_user_last": user_last,
                    f"{band_name}_user_peak": float(np.quantile(window, peak_quantile)),
                    f"{band_name}_user_mean": float(np.mean(window)),
                    f"{band_name}_scaffold_delta": scaffold_last - user_last,
                }
                feature_row.update(prefix_map)
            # documented F1/F3 aliases (same values, original names; F2/F4
            # already carry their documented names late_user_peak /
            # mid_user_peak from the loop above)
            feature_row["late_ref_level"] = feature_row.pop("late_scaffold_last")
            feature_row["scaffold_delta"] = feature_row.pop("late_scaffold_delta")
            feature_rows.append(feature_row)
            del row, readouts

    print(f"Runs processed: {len(curve_rows)} "
          f"(heatmap skips {n_skipped_short} runs with < {user_window_n} user positions"
          + (f"; {n_skipped_system} runs skipped by system_ids={sorted(system_ids)}"
             if system_ids is not None else "")
          + ")")
    return {
        "n_layers": n_layers,
        "curve_df": pd.DataFrame(curve_rows),
        "offset_dfs": {layer: pd.DataFrame(rows) for layer, rows in offset_rows.items()},
        "features_df": pd.DataFrame(feature_rows),
    }


def cv_auc(X, y, groups, pipeline_):
    scores = cross_val_score(
        pipeline_, X, y, cv=GroupKFold(5), groups=groups, scoring="roc_auc"
    )
    return scores.mean(), scores.std()


def plot_per_layer_curve(curve_df, n_layers, scaffold_ref_n, scope_label, out_path):
    """Graph 1 -- the notebook's per-layer classifier curve, read at the
    mean over the last scaffold_ref_n scaffolding tokens, one line per
    authorization cohort. Features are already log-scale, so the pipeline
    scales but does not log again."""
    pipeline_ = Pipeline([
        ("scaling", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])
    layers = list(range(n_layers))

    fig, ax = plt.subplots(figsize=(10, 6))
    for cohort, subset in cohort_frames(curve_df):
        means, stds = [], []
        for layer in layers:
            mean, std = cv_auc(
                subset[[f"scaffold_ref_logrank_L{layer}"]],
                subset["attack_successful"], subset["category"], pipeline_,
            )
            means.append(mean)
            stds.append(std)
        means, stds = np.array(means), np.array(stds)
        color = COHORT_COLORS.get(cohort, "#52514e")
        ax.plot(layers, means, color=color, linewidth=2, marker="o", markersize=4,
                label=f"{COHORT_LABELS.get(cohort, cohort)} (n={len(subset)})")
        ax.fill_between(layers, means - stds, means + stds, color=color, alpha=0.15, linewidth=0)

    ax.axhline(0.5, color="#898781", linestyle="--", linewidth=1.2, zorder=1)
    ax.text(0.3, 0.51, "chance (AUC 0.5)", color="#898781", fontsize=9, va="bottom")
    ax.set_xlabel("layer")
    ax.set_ylabel("cross-validated ROC-AUC")
    ax.set_xticks(range(0, n_layers, 2))
    ax.set_title(
        f"Per-layer secret-token-rank classifier -- {scope_label} "
        "(GroupKFold by category, dev categories only)\n"
        f"reference: mean log-rank over last {scaffold_ref_n} scaffolding tokens"
    )
    ax.grid(axis="y", color="#e1e0d9", linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print("Saved:", out_path)


def plot_offset_heatmap(offset_dfs, analysis_cfg, scope_label, out_path):
    """Graph 2 -- the notebook's layer x position-offset AUC heatmap, rows
    in chronological order (user 7..0, then scaffold 8..0; user-0 and
    scaffold-8 are the two adjacent tokens, so they sit on either side of
    the divider). Diverging color scale around chance, shared across both
    authorization-cohort panels."""
    heatmap_layers = analysis_cfg["heatmap_layers"]
    user_offsets = list(range(analysis_cfg["user_window_n"]))
    any_df = next(iter(offset_dfs.values()))
    scaffold_offsets = sorted(
        int(c.removeprefix("scaffold_offset"))
        for c in any_df.columns if c.startswith("scaffold_offset")
    )
    # Raw ranks -> log inside the CV pipeline, as in the notebook.
    pipeline_ = Pipeline([
        ("log_transform", FunctionTransformer(np.log)),
        ("scaling", StandardScaler()),
        ("model", LogisticRegression(max_iter=1000)),
    ])

    cohorts = [name for name, _ in cohort_frames(any_df)]
    row_specs = (
        [("user", o) for o in reversed(user_offsets)]
        + [("scaffold", o) for o in reversed(scaffold_offsets)]
    )
    row_labels = [f"{segment} {offset}" for segment, offset in row_specs]

    matrices, cohort_sizes = {}, {}
    for cohort in cohorts:
        matrix = np.zeros((len(row_specs), len(heatmap_layers)))
        for row_idx, (segment, offset) in enumerate(row_specs):
            for col_idx, layer in enumerate(heatmap_layers):
                subset = dict(cohort_frames(offset_dfs[layer]))[cohort]
                cohort_sizes[cohort] = len(subset)
                mean, _ = cv_auc(
                    subset[[f"{segment}_offset{offset}"]],
                    subset["attack_successful"], subset["category"], pipeline_,
                )
                matrix[row_idx, col_idx] = mean
        matrices[cohort] = matrix

    all_values = np.concatenate([m.ravel() for m in matrices.values()])
    vmax_dev = max(abs(all_values.min() - 0.5), abs(all_values.max() - 0.5))
    norm = TwoSlopeNorm(vmin=0.5 - vmax_dev, vcenter=0.5, vmax=0.5 + vmax_dev)

    fig, axes = plt.subplots(1, len(cohorts), figsize=(13, 8), sharey=True)
    im = None
    for ax, cohort in zip(np.atleast_1d(axes), cohorts):
        im = ax.imshow(matrices[cohort], aspect="auto", cmap=DIVERGING_CMAP, norm=norm)
        ax.set_xticks(range(len(heatmap_layers)))
        ax.set_xticklabels(heatmap_layers, rotation=90)
        ax.set_yticks(range(len(row_labels)))
        ax.set_yticklabels(row_labels, fontsize=8)
        ax.axhline(len(user_offsets) - 0.5, color="white", linewidth=2)
        ax.set_xlabel("layer")
        ax.set_title(f"{COHORT_LABELS.get(cohort, cohort)} (n={cohort_sizes[cohort]})")

    np.atleast_1d(axes)[0].set_ylabel("position offset (top-to-bottom = chronological order)")
    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("cross-validated ROC-AUC")
    fig.suptitle(
        f"Per-layer ROC-AUC by position offset -- {scope_label} "
        "(diverging around chance = 0.5, dev categories only)"
    )
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", out_path)


def run(settings, input_file):
    """Run the single-token analysis; returns the feature-bank DataFrame
    (one row per run: id, category, system_id, user_type, authorized,
    user_type_is_admin, attack_successful, n_user_positions, the 12 band x
    position-aggregate lens features, split).

    Restricted to settings["system_ids"] (the project analyses sys_strict
    only). Figures and their CV run on dev categories only -- the holdout
    categories are the final test set, and letting them into the curves
    would leak test information into band/window choices. The feature
    table covers ALL runs (holdout rows are needed for the final gated
    evaluation) and marks each row via the "split" column.
    """
    analysis_cfg = settings["analysis"]
    holdout = set(settings["validation"]["holdout_categories"])
    system_ids = settings.get("system_ids")
    system_ids = set(system_ids) if system_ids is not None else None
    scope_label = (
        " + ".join(sorted(system_ids)) if system_ids is not None else "all system prompts"
    )
    out_dir = settings["analysis_output_dir"] / settings["active_model"]
    out_dir.mkdir(parents=True, exist_ok=True)
    print("Reading:", input_file)

    data = collect(input_file, analysis_cfg, system_ids)
    if data["features_df"].empty:
        raise ValueError(f"No runs left after filtering to system_ids={sorted(system_ids)}")

    curve_dev = data["curve_df"][~data["curve_df"]["category"].isin(holdout)]
    offset_dev = {
        layer: df[~df["category"].isin(holdout)]
        for layer, df in data["offset_dfs"].items()
    }
    print(f"Figures use dev categories only: {len(curve_dev)} of "
          f"{len(data['curve_df'])} runs (holdout: {', '.join(sorted(holdout))})")

    plot_per_layer_curve(
        curve_dev, data["n_layers"], analysis_cfg["scaffold_ref_n"], scope_label,
        out_dir / "per_layer_auc_scaffold_ref.png",
    )
    plot_offset_heatmap(
        offset_dev, analysis_cfg, scope_label,
        out_dir / "offset_layer_auc_heatmap.png",
    )

    features_df = data["features_df"]
    features_df["split"] = np.where(
        features_df["category"].isin(holdout), "holdout", "dev"
    )
    features_csv = out_dir / "single_token_features.csv"
    features_df.to_csv(features_csv, index=False)
    print("Saved:", features_csv)
    return features_df
