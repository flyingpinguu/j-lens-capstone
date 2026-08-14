"""A100-only dim_batch and persistent-checkpoint smoke test for Colab.

This intentionally does not fit or save a Jacobian Lens.  It loads the merged
Qwen3.5-4B LoRA model, runs two real Jacobian backward passes for each
``dim_batch`` candidate, and records time plus peak CUDA memory.  A separate
helper writes and reloads an exact-shape jlens checkpoint payload on a mounted
Google Drive path so persistence can be tested independently.
"""

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import shutil
import statistics
import time
from pathlib import Path

import psutil
import torch
from jlens.examples import load_wikitext_prompts
from jlens.fitting import valid_position_mask
from jlens.hooks import ActivationRecorder
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens


MODEL_ID = "Qwen/Qwen3.5-4B"
CONTENT_ROOT = Path("/content")
ADAPTER_DIR = CONTENT_ROOT / "qwen35-4b-lora-pi-r8-stage1-120"
REPORT_PATH = CONTENT_ROOT / "qwen35_lora_a100_dim_batch_report.json"
MAX_SEQ_LEN = 128
SKIP_FIRST = 16


def gib(n_bytes: int) -> float:
    return n_bytes / 1024**3


def prepare_adapter() -> Path:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        destination = ADAPTER_DIR / filename
        source = CONTENT_ROOT / filename
        if not destination.exists():
            if not source.exists():
                raise FileNotFoundError(
                    f"Upload {filename} into /content before running this test"
                )
            shutil.copy2(source, destination)
    return ADAPTER_DIR


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before running the benchmark")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name:
        raise RuntimeError(f"This benchmark requires an A100, got {gpu_name}")

    os.environ.setdefault("HF_HOME", "/content/hf-cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter_dir = prepare_adapter()
    config = json.loads((adapter_dir / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("The uploaded adapter does not target Qwen/Qwen3.5-4B")

    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    lens_model = jlens.from_hf(merged_model, tokenizer)
    if (lens_model.n_layers, lens_model.d_model) != (32, 2560):
        raise ValueError(
            "Unexpected Qwen dimensions: "
            f"{lens_model.n_layers} layers, d_model={lens_model.d_model}"
        )
    return lens_model, dtype, gpu_name


def benchmark_candidate(model, prompt: str, dim_batch: int) -> dict:
    """Run one forward plus two real backward passes, matching jlens.fit."""
    source_layers = list(range(model.n_layers - 1))
    target_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=MAX_SEQ_LEN)
    position_mask = valid_position_mask(input_ids.shape[1], skip_first=SKIP_FIRST)
    valid_positions_cpu = position_mask.nonzero(as_tuple=True)[0]

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    started = time.perf_counter()
    backward_seconds: list[float] = []

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        model.forward(input_ids.expand(dim_batch, -1))
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        valid_positions = valid_positions_cpu.to(target_activation.device)
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        # Two passes matter: the first retains the graph exactly as production
        # fit() must do, while the second releases it.
        for pass_index in range(2):
            dim_start = pass_index * dim_batch
            n_dims = min(dim_batch, model.d_model - dim_start)
            if n_dims <= 0:
                break
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims, None],
            ] = 1.0
            torch.cuda.synchronize()
            pass_started = time.perf_counter()
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_index == 0),
            )
            # Match jlens.fit(): reduce all layers and copy their rows to CPU.
            cpu_rows = [
                grad[:n_dims, valid_positions.to(grad.device), :]
                .float()
                .mean(dim=1)
                .cpu()
                for grad in grads
            ]
            torch.cuda.synchronize()
            backward_seconds.append(time.perf_counter() - pass_started)
            del cpu_rows, grads

    torch.cuda.synchronize()
    total_seconds = time.perf_counter() - started
    peak_allocated = torch.cuda.max_memory_allocated()
    total_vram = torch.cuda.get_device_properties(0).total_memory
    estimated_passes = math.ceil(model.d_model / dim_batch)
    median_backward = statistics.median(backward_seconds)
    return {
        "dim_batch": dim_batch,
        "status": "ok",
        "seq_len": int(input_ids.shape[1]),
        "valid_positions": int(position_mask.sum()),
        "measured_backward_passes": len(backward_seconds),
        "backward_seconds": backward_seconds,
        "benchmark_seconds": total_seconds,
        "peak_allocated_gib": gib(peak_allocated),
        "peak_headroom_gib": gib(total_vram - peak_allocated),
        "estimated_backward_passes_per_prompt": estimated_passes,
        "estimated_prompt_seconds": estimated_passes * median_backward,
    }


def candidate_batches(total_vram_gib: float) -> list[int]:
    # Colab A100 instances are commonly either 40 or 80 GB.
    if total_vram_gib < 60:
        return [4, 8, 12, 14, 16, 20, 24, 28, 32]
    return [8, 16, 24, 32, 40, 48, 56, 64]


def run_batch_benchmark() -> dict:
    model, dtype, gpu_name = load_model()
    prompt = load_wikitext_prompts(n_prompts=1)[0]
    total_vram_gib = gib(torch.cuda.get_device_properties(0).total_memory)
    results: list[dict] = []

    for dim_batch in candidate_batches(total_vram_gib):
        print(f"\nTesting dim_batch={dim_batch} ...", flush=True)
        try:
            result = benchmark_candidate(model, prompt, dim_batch)
        except torch.cuda.OutOfMemoryError as exc:
            result = {
                "dim_batch": dim_batch,
                "status": "oom",
                "error": str(exc).splitlines()[0],
            }
            results.append(result)
            print(json.dumps(result, indent=2), flush=True)
            gc.collect()
            torch.cuda.empty_cache()
            break
        results.append(result)
        print(json.dumps(result, indent=2), flush=True)

    successful = [row for row in results if row["status"] == "ok"]
    if not successful:
        raise RuntimeError("No dim_batch candidate completed")
    # Reserve at least 5 GiB for allocator variation and Colab background use.
    recommended = max(
        (row for row in successful if row["peak_headroom_gib"] >= 5.0),
        key=lambda row: row["dim_batch"],
        default=successful[0],
    )
    report = {
        "purpose": "capacity test only; no lens fitted",
        "model_id": MODEL_ID,
        "adapter": ADAPTER_DIR.name,
        "gpu": gpu_name,
        "dtype": str(dtype),
        "total_vram_gib": total_vram_gib,
        "system_ram_total_gib": gib(psutil.virtual_memory().total),
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST,
        "results": results,
        "largest_successful_dim_batch": successful[-1]["dim_batch"],
        "recommended_dim_batch_5gib_margin": recommended["dim_batch"],
        "checkpoint_persistence": "not tested yet",
    }
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print("\nFINAL BATCH REPORT")
    print(json.dumps(report, indent=2), flush=True)
    return report


def test_checkpoint_persistence(drive_directory: str) -> dict:
    """Atomically save and reload an exact-shape jlens checkpoint on Drive."""
    destination_dir = Path(drive_directory)
    if not str(destination_dir).startswith("/content/drive/"):
        raise ValueError("Pass a directory under the mounted /content/drive path")
    destination_dir.mkdir(parents=True, exist_ok=True)
    destination = destination_dir / "a100-checkpoint-persistence-test.pt"
    temporary = destination.with_suffix(".pt.tmp")

    # Same tensor count, dtype, keys, and approximate 775 MiB size as the real
    # all-layer Qwen3.5-4B fit checkpoint, without computing a lens.
    state = {
        "jacobian_sum": {
            layer: torch.zeros(2560, 2560, dtype=torch.float32)
            for layer in range(31)
        },
        "n_done": 0,
        "next_idx": 0,
        "source_layers": list(range(31)),
        "target_layer": 31,
        "skip_first": SKIP_FIRST,
    }
    started = time.perf_counter()
    torch.save(state, temporary)
    os.replace(temporary, destination)
    write_seconds = time.perf_counter() - started
    size_bytes = destination.stat().st_size
    del state
    gc.collect()

    loaded = torch.load(destination, map_location="cpu", weights_only=True)
    valid = (
        loaded["source_layers"] == list(range(31))
        and loaded["target_layer"] == 31
        and loaded["jacobian_sum"][0].shape == (2560, 2560)
        and loaded["jacobian_sum"][30].dtype == torch.float32
    )
    del loaded
    gc.collect()
    with destination.open("rb") as handle:
        digest = hashlib.sha256()
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)

    persistence = {
        "status": "ok" if valid else "invalid",
        "path": str(destination),
        "size_bytes": size_bytes,
        "size_gib": gib(size_bytes),
        "write_seconds": write_seconds,
        "sha256": digest.hexdigest(),
        "reload_valid": valid,
        "note": "Synthetic exact-shape checkpoint; no lens was fitted.",
    }
    report = json.loads(REPORT_PATH.read_text())
    report["checkpoint_persistence"] = persistence
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    drive_report_path = destination_dir / "a100-checkpoint-persistence-report.json"
    drive_report_path.write_text(json.dumps(report, indent=2) + "\n")
    destination.unlink()
    persistence["test_checkpoint_deleted"] = not destination.exists()
    persistence["drive_report_path"] = str(drive_report_path)
    report["checkpoint_persistence"] = persistence
    REPORT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    drive_report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(persistence, indent=2), flush=True)
    return persistence


benchmark_report = run_batch_benchmark()
