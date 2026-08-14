"""Benchmark a few real Jacobian backward passes and extrapolate overnight fit size."""

import argparse
import json
import logging
import math
import statistics
import time
from pathlib import Path

import torch
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder

from common import (
    ADAPTER_DIR,
    MODEL_ID,
    OUTPUT_DIR,
    load_merged_lora_lens_model,
    load_or_create_wikitext_prompts,
    memory_snapshot,
    synchronize,
)


def benchmark_passes(model, prompt, *, dim_batch, max_seq_len, test_passes, device):
    source_layers = list(range(model.n_layers - 1))
    target_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len)
    valid_positions_cpu = position_mask.nonzero(as_tuple=True)[0]
    n_total_passes = math.ceil(model.d_model / dim_batch)
    test_passes = min(test_passes, n_total_passes)

    pass_seconds = []
    memory_after_forward = None
    max_driver_gib = 0.0
    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        replicated_ids = input_ids.expand(dim_batch, -1)
        synchronize(device)
        forward_started = time.perf_counter()
        model.forward(replicated_ids)
        synchronize(device)
        forward_seconds = time.perf_counter() - forward_started
        memory_after_forward = memory_snapshot(device)

        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        valid_positions = valid_positions_cpu.to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_index in range(test_passes):
            dim_start = pass_index * dim_batch
            n_dims = min(dim_batch, model.d_model - dim_start)
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            synchronize(device)
            pass_started = time.perf_counter()
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_index < test_passes - 1),
            )
            # Match fit(): reduce every requested layer and copy its rows to CPU.
            cpu_rows = []
            for grad in grads:
                positions_on_device = valid_positions.to(grad.device)
                cpu_rows.append(
                    grad[:n_dims, positions_on_device, :].float().mean(dim=1).cpu()
                )
            synchronize(device)
            pass_seconds.append(time.perf_counter() - pass_started)
            snapshot = memory_snapshot(device)
            max_driver_gib = max(
                max_driver_gib,
                snapshot.get("driver_allocated_gib", snapshot.get("peak_allocated_gib", 0.0)),
            )
            del cpu_rows, grads

    stable_passes = pass_seconds[1:] if len(pass_seconds) > 1 else pass_seconds
    seconds_per_pass = statistics.median(stable_passes)
    estimated_prompt_seconds = forward_seconds + n_total_passes * seconds_per_pass
    return {
        "dim_batch": dim_batch,
        "max_seq_len": max_seq_len,
        "seq_len": seq_len,
        "valid_positions": int(position_mask.sum()),
        "source_layers": len(source_layers),
        "d_model": model.d_model,
        "total_backward_passes_per_prompt": n_total_passes,
        "measured_backward_passes": test_passes,
        "forward_seconds": forward_seconds,
        "backward_pass_seconds": pass_seconds,
        "median_stable_backward_seconds": seconds_per_pass,
        "estimated_prompt_seconds": estimated_prompt_seconds,
        "memory_after_forward": memory_after_forward,
        "max_observed_driver_or_peak_gib": max_driver_gib,
    }


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--adapter-dir", type=Path, default=ADAPTER_DIR)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--dim-batch", type=int, default=4)
    parser.add_argument("--max-seq-len", type=int, default=128)
    parser.add_argument("--test-passes", type=int, default=4)
    parser.add_argument("--overnight-hours", type=float, default=10.0)
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.dim_batch < 1 or args.test_passes < 1:
        raise ValueError("--dim-batch and --test-passes must be positive")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_cache = args.output_dir / "benchmark-wikitext-prompt.jsonl"
    prompts = load_or_create_wikitext_prompts(prompt_cache, 1)

    _, _, model, device, dtype = load_merged_lora_lens_model(
        model_id=args.model_id,
        adapter_dir=args.adapter_dir,
        local_files_only=args.local_files_only,
    )
    report = benchmark_passes(
        model,
        prompts[0],
        dim_batch=args.dim_batch,
        max_seq_len=args.max_seq_len,
        test_passes=args.test_passes,
        device=device,
    )
    estimated_seconds = report["estimated_prompt_seconds"]
    report.update(
        {
            "model_id": args.model_id,
            "adapter_dir": str(args.adapter_dir),
            "device": str(device),
            "dtype": str(dtype),
            "overnight_hours": args.overnight_hours,
            "estimated_prompts_overnight": math.floor(
                args.overnight_hours * 3600 / estimated_seconds
            ),
            "two_fp32_all_layer_jacobian_buffers_gib": (
                2 * report["source_layers"] * report["d_model"] ** 2 * 4 / 1024**3
            ),
        }
    )
    report_path = args.output_dir / f"benchmark-dim-batch-{args.dim_batch}.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"Saved benchmark: {report_path}", flush=True)


if __name__ == "__main__":
    main()
