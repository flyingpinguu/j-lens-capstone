"""Fit a resumable all-layer Jacobian Lens for the merged Qwen LoRA model."""

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import jlens
import torch

from common import (
    ADAPTER_DIR,
    MODEL_ID,
    OUTPUT_DIR,
    load_merged_lora_lens_model,
    load_or_create_wikitext_prompts,
    memory_snapshot,
)


def prompts_sha256(prompts):
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def configure_logging(log_path):
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.handlers[:] = [stream_handler, file_handler]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--n-prompts", type=int, required=True)
    parser.add_argument("--dim-batch", type=int, required=True)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=1)
    parser.add_argument("--no-resume", action="store_true")
    parser.add_argument("--overwrite-final", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.n_prompts < 1 or args.dim_batch < 1 or args.checkpoint_every < 1:
        raise ValueError("prompt count, dim batch, and checkpoint interval must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.output_dir / f"fit-n{args.n_prompts}-dim{args.dim_batch}.log"
    configure_logging(log_path)

    prompt_path = args.output_dir / f"wikitext-prompts-n{args.n_prompts}.jsonl"
    checkpoint_path = args.output_dir / f"fit-n{args.n_prompts}-checkpoint.pt"
    lens_path = args.output_dir / f"jacobian-lens-n{args.n_prompts}.pt"
    manifest_path = args.output_dir / f"fit-n{args.n_prompts}-manifest.json"
    if lens_path.exists() and not args.overwrite_final:
        raise FileExistsError(f"Final lens already exists: {lens_path}")

    prompts = load_or_create_wikitext_prompts(prompt_path, args.n_prompts)
    manifest = {
        "model_id": args.model_id,
        "adapter_dir": str(args.adapter_dir.resolve()),
        "n_prompts": args.n_prompts,
        "prompt_sha256": prompts_sha256(prompts),
        "source_layers": list(range(31)),
        "target_layer": 31,
        "dim_batch": args.dim_batch,
        "max_seq_len": args.max_seq_len,
        "skip_first": args.skip_first,
        "checkpoint_every": args.checkpoint_every,
    }
    if checkpoint_path.exists() and not args.no_resume:
        if not manifest_path.exists():
            raise FileNotFoundError(f"Checkpoint exists without manifest: {manifest_path}")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if existing_manifest != manifest:
            raise ValueError(
                "Existing checkpoint manifest differs from requested settings; "
                "use matching arguments or a different output directory"
            )
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    logging.info("Loading and merging LoRA adapter")
    _, _, model, device, dtype = load_merged_lora_lens_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        local_files_only=args.local_files_only,
    )
    if model.n_layers != 32 or model.d_model != 2560:
        raise ValueError(
            f"Expected Qwen3.5-4B dimensions (32, 2560), got "
            f"({model.n_layers}, {model.d_model})"
        )
    source_layers = list(range(model.n_layers - 1))
    logging.info(
        "Starting fit: device=%s dtype=%s prompts=%d dim_batch=%d memory=%s",
        device,
        dtype,
        len(prompts),
        args.dim_batch,
        memory_snapshot(device),
    )
    started = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=source_layers,
        target_layer=model.n_layers - 1,
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        skip_first=args.skip_first,
        checkpoint_path=str(checkpoint_path),
        checkpoint_every=args.checkpoint_every,
        resume=not args.no_resume,
    )
    fit_seconds = time.perf_counter() - started
    lens.save(str(lens_path), dtype=torch.float16)
    completion = {
        **manifest,
        "device": str(device),
        "dtype": str(dtype),
        "fit_seconds": fit_seconds,
        "lens_path": str(lens_path),
        "lens_size_bytes": lens_path.stat().st_size,
        "completed_n_prompts": lens.n_prompts,
    }
    (args.output_dir / f"fit-n{args.n_prompts}-complete.json").write_text(
        json.dumps(completion, indent=2) + "\n",
        encoding="utf-8",
    )
    logging.info("Completed lens: %s", completion)


if __name__ == "__main__":
    main()
