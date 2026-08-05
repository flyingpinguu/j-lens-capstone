"""Build one 90-prompt evaluation set spanning seen and withheld categories."""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CORPUS = ROOT / "data" / "evaluation" / "injection_corpus.jsonl"
DEFAULT_TRAIN = ROOT / "data" / "training" / "qwen35-4b-pi-r8-stage1-120.jsonl"
DEFAULT_OUTPUT = ROOT / "data" / "evaluation" / "qwen35-4b-pi-consolidated-90.jsonl"
WITHHELD_CATEGORIES = {
    "authority_impersonation",
    "encoding_obfuscation",
    "storytelling",
}


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seen-total", type=int, default=60)
    parser.add_argument("--withheld-per-category", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def compact_row(row, category_exposure):
    return {
        "template_id": row["template_id"],
        "category": row["category"],
        "category_exposure": category_exposure,
        "aggressiveness": row["aggressiveness"],
        "prompt": row["prompt"],
    }


def main():
    args = parse_args()
    corpus = read_jsonl(args.corpus)
    train_rows = read_jsonl(args.train_file)
    train_ids = {row["template_id"] for row in train_rows}

    attacks_by_category = defaultdict(list)
    for row in corpus:
        if row["label"] == "attack":
            attacks_by_category[row["category"]].append(row)

    withheld_rows = []
    for category in sorted(WITHHELD_CATEGORIES):
        # Preserve corpus order so these are the same first ten templates
        # already used by the two earlier withheld-category smoke runs.
        candidates = attacks_by_category[category]
        selected = candidates[: args.withheld_per_category]
        if len(selected) != args.withheld_per_category:
            raise ValueError(f"Not enough withheld rows for {category}")
        withheld_rows.extend(compact_row(row, "withheld_category") for row in selected)

    seen_categories = sorted(set(attacks_by_category) - WITHHELD_CATEGORIES)
    base_count, remainder = divmod(args.seen_total, len(seen_categories))
    rng = random.Random(args.seed)
    seen_rows = []
    seen_counts = {}
    for category_index, category in enumerate(seen_categories):
        requested = base_count + (1 if category_index < remainder else 0)
        candidates = [
            row for row in attacks_by_category[category] if row["template_id"] not in train_ids
        ]
        candidates.sort(key=lambda row: row["template_id"])
        rng.shuffle(candidates)
        selected = candidates[:requested]
        if len(selected) != requested:
            raise ValueError(f"Not enough non-training rows for {category}")
        seen_counts[category] = requested
        seen_rows.extend(compact_row(row, "seen_category") for row in selected)

    output_rows = seen_rows + withheld_rows
    output_ids = {row["template_id"] for row in output_rows}
    if len(output_ids) != len(output_rows):
        raise AssertionError("Consolidated evaluation has duplicate template IDs")
    if output_ids & train_ids:
        raise AssertionError("Consolidated evaluation overlaps training templates")

    write_jsonl(args.output, output_rows)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "unique_prompts": len(output_rows),
                "seen_category_prompts": len(seen_rows),
                "withheld_category_prompts": len(withheld_rows),
                "seen_category_counts": seen_counts,
                "withheld_category_counts": {
                    category: args.withheld_per_category
                    for category in sorted(WITHHELD_CATEGORIES)
                },
                "training_template_overlap": 0,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
