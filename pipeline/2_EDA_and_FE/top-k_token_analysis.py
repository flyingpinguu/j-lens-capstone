"""Stage 2.2 -- compact Top-k readouts and fold-local feature transforms.

The JSONL is streamed into small arrays once.  Vocabulary selection and SVD
are deliberately exposed as ``fit``/``transform`` functions: Stage 3 fits
them only on the training part of each fold, never on validation or holdout
rows.
"""

from collections import Counter
import json

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD


def _prompt_positions(row):
    """Saved user + scaffolding positions in chronological order."""
    if row.get("readout_scope") in {"last_n", "last_n_prompt_plus_response"}:
        # Compact runs align examples by exact distance from the defender
        # prompt end. Short current messages can therefore include a few
        # preceding chat-template/history tokens among these 16 positions.
        # Response positions remain excluded even when they share the JSONL.
        return sorted(
            int(position)
            for position in row["readouts"]
            if int(position) < row["response_start_position"]
        )
    return sorted(
        int(position)
        for position, readout in row["readouts"].items()
        if int(position) < row["response_start_position"]
        and readout["segment"] in {"user", "prompt_suffix"}
    )


def run(settings, input_file):
    """Stream a run JSONL into run × position × layer × Top-k arrays.

    ``settings["system_ids"]`` (None = every system prompt) and optional
    ``settings["analysis_include_labels"]`` are applied with the same rule
    as Stage 2.1, so both tracks -- and therefore the shared fold plan --
    cover exactly the same runs.
    """
    cfg = settings["multitoken"]
    system_ids = settings.get("system_ids")
    system_ids = set(system_ids) if system_ids is not None else None
    include_labels = settings.get("analysis_include_labels")
    include_labels = set(include_labels) if include_labels is not None else None
    layers = tuple(cfg["layers"])
    top_k = cfg["top_k"]
    n_positions = cfg["n_prompt_positions"]
    position_names = tuple(
        f"prompt_end_minus_{offset:02d}"
        for offset in range(n_positions, 0, -1)
    )

    metadata_rows, all_ids, all_logits, all_probe_ranks, skipped = [], [], [], [], []
    token_text = {}
    n_skipped_system = 0
    n_skipped_label = 0

    with input_file.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                print(f"  skipping unparseable line {line_number}")
                continue

            if system_ids is not None and row["system_id"] not in system_ids:
                n_skipped_system += 1
                continue
            if include_labels is not None and row["label"] not in include_labels:
                n_skipped_label += 1
                continue

            prompt_positions = _prompt_positions(row)
            if len(prompt_positions) < n_positions:
                skipped.append(row["id"])
                continue

            run_ids, run_logits, run_probe_ranks = [], [], []
            for offset in range(n_positions, 0, -1):
                readout = row["readouts"][str(prompt_positions[-offset])]
                position_ids, position_logits, position_probe_ranks = [], [], []
                for layer in layers:
                    layer_readout = readout["layers"][str(layer)]
                    top = layer_readout["top_k"]
                    ids = [int(value) for value in top["token_ids"][:top_k]]
                    logits = [float(value) for value in top["logits"][:top_k]]
                    if len(ids) != top_k or len(logits) != top_k:
                        raise ValueError(
                            f"{row['id']} position {-offset} layer {layer}: expected Top-{top_k}"
                        )
                    position_ids.append(ids)
                    position_logits.append(logits)
                    probe = layer_readout.get("probe")
                    position_probe_ranks.append(
                        float(probe["rank"]) if probe is not None else np.nan
                    )
                    token_text.update(zip(ids, top["tokens"][:top_k]))
                run_ids.append(position_ids)
                run_logits.append(position_logits)
                run_probe_ranks.append(position_probe_ranks)

            metadata_rows.append({
                "run_id": row["id"],
                "conversation_id": row.get("conversation_id", row["id"]),
                "attempt_index": row.get("attempt_index", 1),
                "template_id": row["template_id"],
                "label": row["label"],
                "category": row["category"],
                "system_id": row["system_id"],
                "authorized": bool(row.get("authorized", False)),
                "actual_leaked": bool(row["attack_successful"]),
            })
            all_ids.append(run_ids)
            all_logits.append(run_logits)
            all_probe_ranks.append(run_probe_ranks)

    token_ids = np.asarray(all_ids, dtype=np.int32)
    logits = np.asarray(all_logits, dtype=np.float32)
    probe_ranks = np.asarray(all_probe_ranks, dtype=np.float32)
    expected = (len(metadata_rows), n_positions, len(layers), top_k)
    if token_ids.shape != expected or logits.shape != expected:
        raise ValueError(f"Unexpected Top-k array shape: {token_ids.shape}, expected {expected}")
    expected_ranks = expected[:3]
    if probe_ranks.shape != expected_ranks:
        raise ValueError(
            f"Unexpected probe-rank array shape: {probe_ranks.shape}, expected {expected_ranks}"
        )

    print(f"Stage 2.2: {len(metadata_rows):,} runs, shape {token_ids.shape}; "
          f"skipped {len(skipped)} runs shorter than {n_positions} positions"
          + (f", {n_skipped_system} runs outside system_ids={sorted(system_ids)}"
             if system_ids is not None else ""))
    if include_labels is not None:
        print(
            f"Stage 2.2: skipped {n_skipped_label} runs outside "
            f"analysis_include_labels={sorted(include_labels)}"
        )
    return {
        "metadata": pd.DataFrame(metadata_rows),
        "token_ids": token_ids,
        "logits": logits,
        "probe_ranks": probe_ranks,
        "position_names": position_names,
        "layers": layers,
        "top_k": top_k,
        "token_text": token_text,
        "skipped_run_ids": tuple(skipped),
        "source_file": str(input_file),
    }


def select_vocabulary(token_ids, vocabulary_size, min_frequency):
    """Select tokens by training-run frequency only (never test rows)."""
    frequency = Counter()
    for row in token_ids:
        frequency.update(set(int(token_id) for token_id in row.ravel()))
    eligible = [
        (token_id, count)
        for token_id, count in frequency.items()
        if count >= min_frequency
    ]
    eligible.sort(key=lambda item: (-item[1], item[0]))
    return [token_id for token_id, _ in eligible[:vocabulary_size]]


def layer_token_matrix(token_ids, logits, vocabulary, n_layers, top_k):
    """Sparse layer × named-token logit matrix."""
    n_rows = len(token_ids)
    width = len(vocabulary)
    if not width:
        raise ValueError("No tokens passed the configured minimum frequency")

    max_id = max(int(token_ids.max()), max(vocabulary))
    lookup = np.full(max_id + 1, -1, dtype=np.int32)
    lookup[np.asarray(vocabulary, dtype=np.int32)] = np.arange(width)

    token_columns = lookup[token_ids.reshape(-1)]
    layers = np.tile(np.repeat(np.arange(n_layers), top_k), n_rows)
    rows = np.repeat(np.arange(n_rows), n_layers * top_k)
    present = token_columns >= 0
    columns = layers[present] * width + token_columns[present]
    return csr_matrix(
        (logits.reshape(-1)[present], (rows[present], columns)),
        shape=(n_rows, n_layers * width),
        dtype=np.float32,
    )


def fit_topk_transform(token_ids, logits, feature_cfg):
    """Fit vocabulary + SVD on training rows and return dense features."""
    vocabulary = select_vocabulary(
        token_ids,
        feature_cfg["vocabulary_size"],
        feature_cfg["min_token_frequency"],
    )
    raw = layer_token_matrix(
        token_ids, logits, vocabulary, token_ids.shape[1], token_ids.shape[2]
    )
    n_components = min(feature_cfg["svd_components"], raw.shape[1] - 1)
    reducer = TruncatedSVD(
        n_components=n_components,
        n_iter=7,
        random_state=feature_cfg["random_state"],
    )
    features = reducer.fit_transform(raw).astype(np.float32)
    return features, {
        "vocabulary": vocabulary,
        "svd": reducer,
        "n_layers": token_ids.shape[1],
        "top_k": token_ids.shape[2],
    }


def transform_topk(token_ids, logits, transform):
    """Apply a training-fitted vocabulary and SVD to unseen rows."""
    raw = layer_token_matrix(
        token_ids,
        logits,
        transform["vocabulary"],
        transform["n_layers"],
        transform["top_k"],
    )
    return transform["svd"].transform(raw).astype(np.float32)


def append_rank_features(topk_features, rank_features, include_secret_rank):
    """Ranks stay interpretable: append them after SVD, never into it."""
    if not include_secret_rank:
        return topk_features
    return np.column_stack([topk_features, rank_features]).astype(np.float32)
