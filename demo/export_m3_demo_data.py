"""Export the recorded Leak Radar examples with real M3 OOF predictions.

Run after the analysis pipeline has produced ``outputs/pipeline``::

    python demo/export_m3_demo_data.py

The generated JavaScript is intentionally committed. This keeps the static
HTML demo reproducible and usable offline without loading the ~1 GB run file.
"""

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RUNS = (
    ROOT
    / "outputs"
    / "j-lens-run"
    / "qwen35-4b-full-corpus-user-response-positions-top10-relabeled.jsonl"
)
DEFAULT_PREDICTIONS = ROOT / "outputs" / "pipeline" / "M3_soft_voting_predictions.csv"
DEFAULT_METRICS = ROOT / "outputs" / "pipeline" / "M3_model_comparison_metrics.csv"
DEFAULT_OUTPUT = ROOT / "demo" / "leak_radar_m3_data.js"

DEMO_RUN_IDS = (
    "sys_lax__control_001",
    "sys_lax__direct_override_05",
    "sys_lax__role_play_02",
    "sys_lax__format_trick_14",
    "sys_lax__hypothetical_scenario_10_v2",
    "sys_strict__control_001",
    "sys_strict__direct_override_05",
    "sys_strict__role_play_02",
    "sys_strict__format_trick_14",
    "sys_strict__hypothetical_scenario_10_v2",
)


def read_csv_by_id(path):
    with path.open(newline="", encoding="utf-8") as file:
        return {row["run_id"]: row for row in csv.DictReader(file)}


def selected_m3_metrics(path):
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    selected = [
        row
        for row in rows
        if row["model"] == "m3_soft_vote"
        and row["selected"].strip().lower() == "true"
    ]
    if len(selected) != 1:
        raise ValueError(f"Expected one selected M3 position, found {len(selected)}")
    return selected[0]


def prompt_positions(row):
    return sorted(
        int(position)
        for position, readout in row["readouts"].items()
        if int(position) < row["response_start_position"]
        and readout["segment"] in {"user", "prompt_suffix"}
    )


def load_demo_runs(path):
    wanted = set(DEMO_RUN_IDS)
    rows = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            if row["id"] in wanted:
                rows[row["id"]] = row
                if len(rows) == len(wanted):
                    break
    missing = wanted - rows.keys()
    if missing:
        raise ValueError(f"Run file is missing demo IDs: {sorted(missing)}")
    return rows


def build_payload(run_file, predictions_file, metrics_file):
    metrics = selected_m3_metrics(metrics_file)
    position_name = metrics["position"]
    position_offset = int(position_name.rsplit("_", 1)[-1])
    probability_column = f"m3__{position_name}__probability"
    m2_column = f"m2__{position_name}__probability"

    predictions = read_csv_by_id(predictions_file)
    runs = load_demo_runs(run_file)
    examples = {}

    for run_id in DEMO_RUN_IDS:
        row = runs[run_id]
        prediction = predictions.get(run_id)
        if prediction is None:
            raise ValueError(f"M3 predictions are missing {run_id}")
        if probability_column not in prediction:
            raise ValueError(f"M3 predictions are missing {probability_column}")

        positions = prompt_positions(row)
        if len(positions) < position_offset:
            raise ValueError(f"{run_id} is too short for {position_name}")
        absolute_position = positions[-position_offset]
        readout = row["readouts"][str(absolute_position)]
        layers = sorted(int(layer) for layer in readout["layers"])
        deepest = readout["layers"][str(layers[-1])]
        probability = float(prediction[probability_column])
        leaked = bool(row["attack_successful"])

        if prediction["actual_leaked"].strip().lower() != str(leaked).lower():
            raise ValueError(f"Ground-truth mismatch for {run_id}")

        examples[run_id] = {
            "user": row["user_prompt"],
            "resp": row["response"],
            "leaked": leaked,
            "split": prediction["split"],
            "fold": int(prediction["fold"]),
            "p": round(probability, 6),
            "predicted_leaked": probability >= 0.5,
            "m1_p": round(float(prediction["m1_probability"]), 6),
            "m2_p": round(float(prediction[m2_column]), 6),
            "position": position_name,
            "position_index": absolute_position,
            "position_token": readout["token"],
            "rank": [
                int(readout["layers"][str(layer)]["probe"]["rank"])
                for layer in layers
            ],
            "read": [
                [token, rank]
                for rank, token in enumerate(
                    deepest["top_k"]["tokens"][:6], start=1
                )
            ],
        }

    return {
        "schema_version": 1,
        "classifier": {
            "name": "M3 soft vote",
            "position": position_name,
            "threshold": 0.5,
            "formula": "0.5 * M1_logreg + 0.5 * M2_topk_SVD_at_position",
            "mean_fold_auc": float(metrics["mean_fold_auc"]),
            "oof_auc": float(metrics["oof_auc"]),
            "oof_accuracy": float(metrics["oof_accuracy"]),
        },
        "source": {
            "runs": str(run_file.relative_to(ROOT)),
            "predictions": str(predictions_file.relative_to(ROOT)),
            "metrics": str(metrics_file.relative_to(ROOT)),
        },
        "examples": examples,
    }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main():
    args = parse_args()
    payload = build_payload(args.runs, args.predictions, args.metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "window.LEAK_RADAR_DATA = "
        + json.dumps(payload, indent=2, ensure_ascii=True)
        + ";\n",
        encoding="utf-8",
    )
    print(f"Saved {len(payload['examples'])} M3 demo examples to {args.output}")


if __name__ == "__main__":
    main()
