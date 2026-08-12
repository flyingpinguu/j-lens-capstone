"""Small multi-turn smoke run on the conversation-seed corpus.

Run from the repository root:
    .venv/bin/python pipeline/1_data_generation/multiturn_smoke.py

It exercises per-conversation secrets and the tactic/strategy/goal fields.
"""

from copy import deepcopy
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pipeline.pipeline_main import SETTINGS, load_stage, read_jsonl  # noqa: E402


TEMPLATE_IDS = ["direct_override_01", "control_001", "admin_request_01"]
MAX_ATTACK_ATTEMPTS = 2


def main():
    settings = deepcopy(SETTINGS)
    settings.update({
        "collection_mode": "multi_turn",
        "secret_mode": "random_single_token_per_conversation",
        "secret_seed": 20260812,
        "secret_min_token_id": 10_000,
        "max_attack_attempts": MAX_ATTACK_ATTEMPTS,
        "template_ids": TEMPLATE_IDS,
        "max_prompts_per_strictness": None,
        "readout_positions": "last_n",
        "readout_last_n": 16,
        "attacker_last_n": 16,
        "probe_enabled": True,
        "max_new_tokens": 48,
        "attacker_max_new_tokens": 48,
        "output_dir": ROOT / "tmp" / "multiturn-smoke" / "single-token-full-map",
    })

    injections = read_jsonl(
        ROOT / "data" / "evaluation" / "conversation_seeds.jsonl"
    )
    system_prompts = read_jsonl(
        ROOT / "data" / "evaluation" / "system_prompts_authz.jsonl"
    )
    harness = load_stage("1_data_generation/run_harness.py")
    output_file = harness.run(settings, injections, system_prompts)

    records = [
        json.loads(line)
        for line in output_file.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    print(
        "Smoke summary:",
        len(records),
        "defender turns | leaks:",
        sum(record["attack_successful"] for record in records),
    )

    # Verify both feature branches can consume the new per-turn rows. The
    # tiny smoke sample deliberately does not fit classifiers: it has too few
    # categories and usually only one target class for meaningful CV.
    single_token = load_stage("2_EDA_and_FE/single_token_analysis.py")
    rank_data = single_token.collect(output_file, settings["analysis"])
    topk = load_stage("2_EDA_and_FE/top-k_token_analysis.py")
    topk_data = topk.run(settings, output_file)
    print(
        "Feature smoke:",
        len(rank_data["features_df"]),
        "rank rows | Top-k shape:",
        topk_data["token_ids"].shape,
    )


if __name__ == "__main__":
    main()
