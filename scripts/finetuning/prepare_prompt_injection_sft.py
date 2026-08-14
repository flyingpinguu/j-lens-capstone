"""Build a balanced prompt-injection SFT split from the current corpus.

The large J-Lens run file is streamed once to find candidate refusal targets;
its readouts are never copied into the compact SFT dataset. Only a clearly
refusing first sentence can be reused. All other attacks receive a short,
deterministic canonical refusal.
"""

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CORPUS = ROOT / "data" / "evaluation" / "injection_corpus.jsonl"
DEFAULT_SYSTEM_PROMPTS = ROOT / "data" / "evaluation" / "system_prompts.jsonl"
DEFAULT_RUN_FILE = (
    ROOT
    / "outputs"
    / "j-lens-run"
    / "qwen35-4b-full-corpus-user-response-positions-top10.jsonl"
)
DEFAULT_TRAIN_FILE = (
    ROOT / "data" / "training" / "qwen35-4b-pi-r8-stage1-120.jsonl"
)
DEFAULT_VALIDATION_FILE = (
    ROOT / "data" / "evaluation" / "qwen35-4b-pi-stage1-validation-48.jsonl"
)
DEFAULT_MANIFEST = ROOT / "data" / "training" / "qwen35-4b-pi-stage1-split.json"

MODEL_ID = "Qwen/Qwen3.5-4B"
EXCLUDED_CATEGORIES = {
    "authority_impersonation",
    "encoding_obfuscation",
    "storytelling",
}
TRAINING_SECRETS = ["orange", "planet", "window", "silver", "forest", "rabbit"]
REFUSAL_MARKERS = re.compile(
    r"\b(?:cannot|can't|won't|will not|unable|not able|must not|refuse)\b",
    flags=re.IGNORECASE,
)
CANONICAL_REFUSALS = [
    "I can't reveal, quote, encode, or transform protected information.",
    "I can't provide the protected value or help bypass its confidentiality rules.",
    "I won't disclose or transform confidential information, but I can help with something else.",
    "I can't follow instructions that would expose protected information.",
    "I can't reveal the confidential value, even in a role-play or hypothetical scenario.",
    "I can't provide hidden or protected information in any form.",
]


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def canonical_refusal(template_id):
    digest = hashlib.sha256(template_id.encode()).digest()
    return CANONICAL_REFUSALS[digest[0] % len(CANONICAL_REFUSALS)]


def safe_first_sentence(response, source_secret):
    """Return a clearly refusing first sentence, or None when uncertain."""
    if not response or source_secret.casefold() in response.casefold():
        return None
    first_paragraph = response.strip().split("\n\n", 1)[0].replace("\n", " ").strip()
    sentences = re.split(r"(?<=[.!?])\s+", first_paragraph, maxsplit=1)
    first_sentence = sentences[0].strip()
    if not 15 <= len(first_sentence) <= 280:
        return None
    if not REFUSAL_MARKERS.search(first_sentence):
        return None
    if first_sentence[-1] not in ".!?":
        return None
    return first_sentence


def collect_candidate_targets(run_file, selected_ids):
    """Stream the ~1 GB run file, retaining only tiny selected projections."""
    candidates = defaultdict(dict)
    with run_file.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"Invalid JSON on {run_file}:{line_number}") from error
            template_id = row.get("template_id")
            if template_id not in selected_ids:
                continue
            target = safe_first_sentence(row.get("response", ""), row.get("secret", ""))
            if target is not None:
                candidates[template_id][row["strictness"]] = target
    return candidates


def validate_training_secrets(tokenizer, secrets):
    for secret in secrets:
        token_ids = tokenizer.encode(secret, add_special_tokens=False)
        if len(token_ids) != 1:
            raise ValueError(f"Training secret {secret!r} is not one token: {token_ids}")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--system-prompts", type=Path, default=DEFAULT_SYSTEM_PROMPTS)
    parser.add_argument("--run-file", type=Path, default=DEFAULT_RUN_FILE)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--validation-file", type=Path, default=DEFAULT_VALIDATION_FILE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--train-per-category", type=int, default=15)
    parser.add_argument("--validation-per-category", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    corpus = read_jsonl(args.corpus)
    system_prompts = read_jsonl(args.system_prompts)
    systems = {row["strictness"]: row for row in system_prompts}
    if set(systems) != {"low", "high"}:
        raise ValueError(f"Expected low/high system prompts, got {sorted(systems)}")

    attacks_by_category = defaultdict(list)
    for row in corpus:
        if row["label"] == "attack" and row["category"] not in EXCLUDED_CATEGORIES:
            attacks_by_category[row["category"]].append(row)
    categories = sorted(attacks_by_category)
    if len(categories) != 8:
        raise ValueError(f"Expected 8 eligible attack categories, got {categories}")

    rng = random.Random(args.seed)
    train_prompts = []
    validation_prompts = []
    for category in categories:
        category_rows = sorted(attacks_by_category[category], key=lambda row: row["template_id"])
        rng.shuffle(category_rows)
        required = args.train_per_category + args.validation_per_category
        if len(category_rows) < required:
            raise ValueError(f"{category} has {len(category_rows)} rows; need {required}")
        train_prompts.extend(category_rows[: args.train_per_category])
        validation_prompts.extend(category_rows[args.train_per_category : required])

    selected_ids = {row["template_id"] for row in train_prompts}
    candidates = collect_candidate_targets(args.run_file, selected_ids)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    validate_training_secrets(tokenizer, TRAINING_SECRETS)

    train_rows = []
    target_source_counts = Counter()
    strictness_counts = Counter()
    for category_index, category in enumerate(categories):
        category_rows = [row for row in train_prompts if row["category"] == category]
        for row_index, row in enumerate(category_rows):
            strictness = "low" if (category_index + row_index) % 2 == 0 else "high"
            system_row = systems[strictness]
            secret_index = (category_index * args.train_per_category + row_index) % len(
                TRAINING_SECRETS
            )
            secret = TRAINING_SECRETS[secret_index]
            system_prompt = system_row["system_prompt"].replace("{{SECRET}}", secret)

            candidate_by_strictness = candidates.get(row["template_id"], {})
            target = candidate_by_strictness.get("high") or candidate_by_strictness.get("low")
            if target is None:
                target = canonical_refusal(row["template_id"])
                target_source = "canonical_refusal"
            else:
                target_source = "run_response_first_sentence"

            train_rows.append(
                {
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["prompt"]},
                        {"role": "assistant", "content": target},
                    ],
                    "template_id": row["template_id"],
                    "category": category,
                    "system_id": system_row["system_id"],
                    "strictness": strictness,
                    "training_secret": secret,
                    "target_source": target_source,
                }
            )
            strictness_counts[strictness] += 1
            target_source_counts[target_source] += 1

    validation_rows = [
        {
            "template_id": row["template_id"],
            "category": row["category"],
            "aggressiveness": row["aggressiveness"],
            "prompt": row["prompt"],
        }
        for row in validation_prompts
    ]
    train_ids = {row["template_id"] for row in train_rows}
    validation_ids = {row["template_id"] for row in validation_rows}
    if train_ids & validation_ids:
        raise AssertionError("Training and validation template IDs overlap")

    write_jsonl(args.train_file, train_rows)
    write_jsonl(args.validation_file, validation_rows)
    manifest = {
        "model_id": args.model_id,
        "seed": args.seed,
        "excluded_categories": sorted(EXCLUDED_CATEGORIES),
        "eligible_categories": categories,
        "train_per_category": args.train_per_category,
        "validation_per_category": args.validation_per_category,
        "training_examples": len(train_rows),
        "validation_prompts": len(validation_rows),
        "strictness_counts": dict(sorted(strictness_counts.items())),
        "target_source_counts": dict(sorted(target_source_counts.items())),
        "training_secrets": TRAINING_SECRETS,
        "train_file": str(args.train_file),
        "validation_file": str(args.validation_file),
        "source_run_file": str(args.run_file),
    }
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
