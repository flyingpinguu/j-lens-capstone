"""Run the strict, single-turn attack and authorized-request collection.

Set MAX_REQUESTS to a small integer for a smoke test or to None for the full
run; the same configuration and output format are used either way.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.pipeline_main import SETTINGS, load_stage, read_jsonl  # noqa: E402


MAX_REQUESTS = None


def main():
    settings = deepcopy(SETTINGS)
    settings.update({
        "collection_mode": "single_turn",
        "include_labels": ["attack", "authorized"],
        "include_system_ids": ["sys_strict"],
        "max_prompts_per_strictness": MAX_REQUESTS,
        "readout_positions": "last_n_prompt_plus_response",
        "readout_last_n": 16,
        "max_new_tokens": 64,
        "probe_enabled": True,
        "output_dir": ROOT / "outputs" / "j-lens-run",
    })

    injections = read_jsonl(
        ROOT / "data" / "evaluation" / "conversation_seeds.jsonl"
    )
    system_prompts = read_jsonl(
        ROOT / "data" / "evaluation" / "system_prompts_authz.jsonl"
    )

    started = time.perf_counter()
    harness = load_stage("1_data_generation/run_harness.py")
    output_file = harness.run(settings, injections, system_prompts)
    elapsed = time.perf_counter() - started

    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    response_positions = [
        sum(cell["segment"] == "response" for cell in row["readouts"].values())
        for row in records
    ]
    prompt_positions = [
        len(row["readouts"]) - response_count
        for row, response_count in zip(records, response_positions)
    ]
    print(
        "Smoke summary:",
        len(records),
        "runs | elapsed:",
        round(elapsed, 1),
        "s | seconds/run:",
        round(elapsed / len(records), 1),
    )
    print("Prompt readout positions:", prompt_positions)
    print("Response readout positions:", response_positions)
    print("Output bytes:", output_file.stat().st_size)


if __name__ == "__main__":
    main()
