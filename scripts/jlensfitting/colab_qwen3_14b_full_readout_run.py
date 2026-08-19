"""Run all strict single-turn requests through Qwen3-14B and its J-Lens.

The JSONL is appended directly to Google Drive and fsynced after every row.
Rerunning the script resumes the same file by run id. A completion report is
written only after the full JSONL passes structural and content checks.
"""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.pipeline_main import MODEL_CONFIGS, SETTINGS, load_stage, read_jsonl  # noqa: E402


DRIVE_ROOT = Path(
    os.environ.get(
        "JLENS_DRIVE_ROOT",
        "/content/drive/MyDrive/j-lens-capstone/qwen3-14b-full-readout",
    )
)


def verify_drive_is_durable(output_dir):
    """Require a working Drive mount and a successful flush+fsync round trip."""
    my_drive = Path("/content/drive/MyDrive")
    if not my_drive.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted at /content/drive. "
            "Run drive.mount('/content/drive') first."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    marker = output_dir / "drive-write-test.json"
    payload = {
        "status": "ok",
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "flush+fsync persistence check before the full run",
    }
    encoded = (json.dumps(payload, indent=2) + "\n").encode()
    with marker.open("wb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    if marker.read_bytes() != encoded:
        raise RuntimeError("Google Drive persistence round trip failed")
    print("Drive persistence check passed:", marker)


def validate_and_summarize(output_file, expected_runs):
    expected_layers = {str(layer) for layer in range(40)}
    digest = hashlib.sha256()
    seen_ids = set()
    secret_token_ids = set()
    categories = Counter()
    labels = Counter()
    stop_reasons = Counter()
    response_tokens = []
    readout_positions = []
    record_count = 0
    system_ids = set()

    with output_file.open("rb") as source:
        for line_number, raw_line in enumerate(source, 1):
            digest.update(raw_line)
            if not raw_line.strip():
                continue
            try:
                row = json.loads(raw_line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on line {line_number}") from error
            record_count += 1
            if row["id"] in seen_ids:
                raise ValueError(f"Duplicate run id: {row['id']}")
            seen_ids.add(row["id"])
            if row["secret_token_count"] != 1:
                raise ValueError(f"Secret is not one token in {row['id']}")
            secret_token_ids.add(row["probe"]["token_id"])
            system_ids.add(row["system_id"])
            categories[row["category"]] += 1
            labels[row["label"]] += 1
            stop_reasons[row["stop_reason"]] += 1
            response_tokens.append(row["benchmark"]["generation_tokens"])
            readout_positions.append(row["benchmark"]["readout_positions"])
            for position, cell in row["readouts"].items():
                found_layers = set(cell["layers"])
                if found_layers != expected_layers:
                    raise ValueError(
                        f"Layer coverage mismatch in {row['id']} position {position}: "
                        f"found {sorted(found_layers, key=int)}"
                    )
                if any("probe" not in layer for layer in cell["layers"].values()):
                    raise ValueError(
                        f"Missing secret rank/logit in {row['id']} position {position}"
                    )

    if record_count != expected_runs:
        raise ValueError(f"Expected {expected_runs} runs, found {record_count}")
    if system_ids != {"sys_strict"}:
        raise ValueError(f"Unexpected system ids: {sorted(system_ids)}")
    if len(secret_token_ids) != expected_runs:
        raise ValueError(
            f"Expected {expected_runs} distinct secret token ids, "
            f"found {len(secret_token_ids)}"
        )

    return {
        "status": "complete_and_validated",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "output_file": str(output_file),
        "output_bytes": output_file.stat().st_size,
        "sha256": digest.hexdigest(),
        "record_count": record_count,
        "unique_run_ids": len(seen_ids),
        "unique_secret_token_ids": len(secret_token_ids),
        "system_ids": sorted(system_ids),
        "layer_count_per_position": 40,
        "layers": list(range(40)),
        "categories": dict(sorted(categories.items())),
        "labels": dict(sorted(labels.items())),
        "stop_reasons": dict(sorted(stop_reasons.items())),
        "response_tokens": {
            "min": min(response_tokens),
            "mean": statistics.mean(response_tokens),
            "max": max(response_tokens),
        },
        "readout_positions": {
            "min": min(readout_positions),
            "mean": statistics.mean(readout_positions),
            "max": max(readout_positions),
        },
    }


def main():
    injections = read_jsonl(
        ROOT / "data" / "evaluation" / "conversation_seeds.jsonl"
    )
    system_prompts = read_jsonl(
        ROOT / "data" / "evaluation" / "system_prompts_authz.jsonl"
    )
    included_labels = {"attack", "authorized"}
    included_system_ids = {"sys_strict"}
    expected_runs = sum(
        row["label"] in included_labels for row in injections
    ) * sum(
        row["system_id"] in included_system_ids for row in system_prompts
    )
    if expected_runs != 538:
        raise ValueError(
            f"Corpus scope changed: expected the reviewed 538 runs, got {expected_runs}"
        )

    verify_drive_is_durable(DRIVE_ROOT)
    settings = deepcopy(SETTINGS)
    settings.update(
        {
            "active_model": "qwen3-14b",
            "model": MODEL_CONFIGS["qwen3-14b"],
            "collection_mode": "single_turn",
            "include_labels": sorted(included_labels),
            "include_system_ids": sorted(included_system_ids),
            "max_prompts_per_strictness": None,
            "template_ids": None,
            "readout_positions": "last_n_prompt_plus_response",
            "readout_last_n": 16,
            "max_new_tokens": 64,
            "probe_enabled": True,
            "benchmark_timing": True,
            "fsync_output": True,
            "output_dir": DRIVE_ROOT,
        }
    )

    started = time.perf_counter()
    harness = load_stage("1_data_generation/run_harness.py")
    output_file = harness.run(settings, injections, system_prompts)
    summary = validate_and_summarize(output_file, expected_runs)
    summary["wall_seconds_this_invocation"] = time.perf_counter() - started
    summary["model"] = settings["model"]["model_id"]
    summary["lens_file"] = settings["model"]["lens_file"]
    summary["secret_seed"] = settings["secret_seed"]
    summary["secret_min_token_id"] = settings["secret_min_token_id"]

    summary_file = output_file.with_suffix(".complete.json")
    encoded = (json.dumps(summary, indent=2) + "\n").encode()
    with summary_file.open("wb") as output:
        output.write(encoded)
        output.flush()
        os.fsync(output.fileno())
    print("FULL_RUN_COMPLETE")
    print(json.dumps(summary, indent=2))
    print("Completion report saved:", summary_file)


if __name__ == "__main__":
    main()
