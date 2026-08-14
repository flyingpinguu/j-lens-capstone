"""Resume the LoRA Qwen3.5-4B Jacobian Lens from n20 to n100 on Colab A100.

Expected transient uploads under ``/content``:

* ``adapter_config.json``
* ``adapter_model.safetensors``
* ``fit-n20-checkpoint.pt`` (optional; initializes a fresh Drive run at n20)

Without the optional n20 checkpoint, a fresh Drive run starts at prompt 1.
Google Drive must already be mounted at ``/content/drive``. The fit checkpoint
is written atomically after every prompt, so rerunning this script resumes from
the next unprocessed WikiText prompt. The finished n100 lens is saved to Drive.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import shutil
import sys
import time
from pathlib import Path

import psutil
import torch
from jlens.examples import load_wikitext_prompts
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens


MODEL_ID = "Qwen/Qwen3.5-4B"
ADAPTER_NAME = "qwen35-4b-lora-pi-r8-stage1-120"
CONTENT_ROOT = Path("/content")
ADAPTER_DIR = CONTENT_ROOT / ADAPTER_NAME
UPLOADED_N20_CHECKPOINT = CONTENT_ROOT / "fit-n20-checkpoint.pt"
DRIVE_ROOT = Path("/content/drive/MyDrive")
OUTPUT_DIR = (
    DRIVE_ROOT
    / "j-lens-capstone"
    / "jlens"
    / ADAPTER_NAME
    / "n100"
)
CHECKPOINT_PATH = OUTPUT_DIR / "fit-n100-checkpoint.pt"
LENS_PATH = OUTPUT_DIR / "jacobian-lens-n100.pt"
MANIFEST_PATH = OUTPUT_DIR / "fit-n100-manifest.json"
COMPLETE_PATH = OUTPUT_DIR / "fit-n100-complete.json"
PROMPTS_PATH = OUTPUT_DIR / "wikitext-prompts-n100.jsonl"
LOG_PATH = OUTPUT_DIR / "fit-n100-dim14.log"

N_PROMPTS = 100
STARTING_N_PROMPTS = 20
DIM_BATCH = 14
MAX_SEQ_LEN = 128
SKIP_FIRST = 16
SOURCE_LAYERS = list(range(31))
TARGET_LAYER = 31

# Verified locally against the exact cached prompts used for the n20 fit.
FIRST20_PROMPT_SHA256 = (
    "c36fa2afe5974fa2389654a210efe2c5323d3f78a45f8eb2300b441fb909d08e"
)
ALL100_PROMPT_SHA256 = (
    "bac183365fab4539801fdf11c2d33108b3e12121fae7371a01e6b16c39a4c553"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_prompts(prompts: list[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        digest.update(prompt.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    shutil.copy2(source, temporary)
    os.replace(temporary, destination)


def atomic_json(data: dict, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, destination)


def configure_logging() -> None:
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(formatter)
    file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root = logging.getLogger()
    root.handlers[:] = [stream, file_handler]
    root.setLevel(logging.INFO)


def prepare_adapter() -> Path:
    ADAPTER_DIR.mkdir(parents=True, exist_ok=True)
    for filename in ("adapter_config.json", "adapter_model.safetensors"):
        destination = ADAPTER_DIR / filename
        source = CONTENT_ROOT / filename
        if not destination.exists():
            if not source.exists():
                raise FileNotFoundError(f"Upload {filename} into /content")
            shutil.copy2(source, destination)
    config = json.loads((ADAPTER_DIR / "adapter_config.json").read_text())
    if config.get("base_model_name_or_path") != MODEL_ID:
        raise ValueError("Uploaded LoRA adapter targets the wrong base model")
    return ADAPTER_DIR


def validate_checkpoint(path: Path, *, minimum_next_idx: int) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=True)
    expected = {
        "source_layers": SOURCE_LAYERS,
        "target_layer": TARGET_LAYER,
        "skip_first": SKIP_FIRST,
    }
    for key, value in expected.items():
        if state.get(key) != value:
            raise ValueError(
                f"Checkpoint {path} has {key}={state.get(key)!r}, expected {value!r}"
            )
    if state.get("next_idx", -1) < minimum_next_idx:
        raise ValueError(
            f"Checkpoint next_idx={state.get('next_idx')} is below {minimum_next_idx}"
        )
    if state.get("n_done", -1) < minimum_next_idx:
        raise ValueError(
            f"Checkpoint n_done={state.get('n_done')} is below {minimum_next_idx}"
        )
    for layer in SOURCE_LAYERS:
        tensor = state["jacobian_sum"][layer]
        if tensor.shape != (2560, 2560) or tensor.dtype != torch.float32:
            raise ValueError(f"Invalid Jacobian accumulator at layer {layer}")
    summary = {
        "n_done": int(state["n_done"]),
        "next_idx": int(state["next_idx"]),
        "size_bytes": path.stat().st_size,
    }
    del state
    gc.collect()
    return summary


def prepare_prompts() -> list[str]:
    prompts = load_wikitext_prompts(n_prompts=N_PROMPTS)
    if len(prompts) != N_PROMPTS:
        raise ValueError(f"Expected {N_PROMPTS} WikiText prompts, got {len(prompts)}")
    first20_hash = sha256_prompts(prompts[:STARTING_N_PROMPTS])
    all100_hash = sha256_prompts(prompts)
    if first20_hash != FIRST20_PROMPT_SHA256:
        raise ValueError(
            "The first 20 WikiText prompts do not match the completed local n20 fit"
        )
    if all100_hash != ALL100_PROMPT_SHA256:
        raise ValueError("The 100-prompt WikiText sample changed unexpectedly")

    if PROMPTS_PATH.exists():
        cached = [
            json.loads(line)["text"]
            for line in PROMPTS_PATH.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if cached != prompts:
            raise ValueError(f"Prompt cache differs: {PROMPTS_PATH}")
    else:
        temporary = PROMPTS_PATH.with_name(PROMPTS_PATH.name + ".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for index, text in enumerate(prompts):
                handle.write(json.dumps({"index": index, "text": text}) + "\n")
        os.replace(temporary, PROMPTS_PATH)
    return prompts


def initialize_or_validate_drive_checkpoint() -> dict:
    if CHECKPOINT_PATH.exists():
        summary = validate_checkpoint(CHECKPOINT_PATH, minimum_next_idx=0)
        logging.info("Using persisted Drive checkpoint: %s", summary)
        return summary

    if not UPLOADED_N20_CHECKPOINT.exists():
        logging.info("No n20 checkpoint uploaded; starting the n100 fit from zero")
        return {"n_done": 0, "next_idx": 0, "size_bytes": 0}
    summary = validate_checkpoint(
        UPLOADED_N20_CHECKPOINT, minimum_next_idx=STARTING_N_PROMPTS
    )
    if summary["next_idx"] != STARTING_N_PROMPTS:
        raise ValueError(
            "The initialization checkpoint must end exactly after prompt 20"
        )
    checkpoint_sha256 = sha256_file(UPLOADED_N20_CHECKPOINT)
    logging.info(
        "Copying verified n20 checkpoint to Drive (%d bytes, sha256=%s)",
        summary["size_bytes"],
        checkpoint_sha256,
    )
    atomic_copy(UPLOADED_N20_CHECKPOINT, CHECKPOINT_PATH)
    persisted_sha256 = sha256_file(CHECKPOINT_PATH)
    if persisted_sha256 != checkpoint_sha256:
        raise IOError("Drive checkpoint SHA-256 differs after copy")
    return validate_checkpoint(
        CHECKPOINT_PATH, minimum_next_idx=STARTING_N_PROMPTS
    )


def load_model():
    if not torch.cuda.is_available():
        raise RuntimeError("Select an A100 GPU runtime")
    gpu_name = torch.cuda.get_device_name(0)
    if "A100" not in gpu_name:
        raise RuntimeError(f"Expected A100, got {gpu_name}")
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    os.environ.setdefault("HF_HOME", "/content/hf-cache")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    adapter_dir = prepare_adapter()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        dtype=dtype,
        low_cpu_mem_usage=True,
    ).to("cuda")
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    model = jlens.from_hf(merged_model, tokenizer)
    if (model.n_layers, model.d_model) != (32, 2560):
        raise ValueError(
            f"Unexpected Qwen dimensions: layers={model.n_layers}, d={model.d_model}"
        )
    return model, dtype, gpu_name


def main() -> None:
    if not DRIVE_ROOT.is_dir():
        raise RuntimeError(
            "Google Drive is not mounted. Run drive.mount('/content/drive') first."
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_logging()

    if LENS_PATH.exists() and COMPLETE_PATH.exists():
        existing = jlens.JacobianLens.load(str(LENS_PATH))
        if existing.n_prompts != N_PROMPTS:
            raise ValueError(f"Existing final lens has n_prompts={existing.n_prompts}")
        logging.info("Final n100 lens already exists: %s", LENS_PATH)
        return

    prompts = prepare_prompts()
    checkpoint_summary = initialize_or_validate_drive_checkpoint()
    manifest_core = {
        "model_id": MODEL_ID,
        "adapter": ADAPTER_NAME,
        "n_prompts": N_PROMPTS,
        "prompt_sha256": ALL100_PROMPT_SHA256,
        "first20_prompt_sha256": FIRST20_PROMPT_SHA256,
        "source_layers": SOURCE_LAYERS,
        "target_layer": TARGET_LAYER,
        "dim_batch": DIM_BATCH,
        "max_seq_len": MAX_SEQ_LEN,
        "skip_first": SKIP_FIRST,
        "checkpoint_every": 1,
        "checkpoint_path": str(CHECKPOINT_PATH),
    }
    if MANIFEST_PATH.exists():
        previous = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
        previous_core = {
            key: previous[key]
            for key in manifest_core
        }
        if previous_core != manifest_core:
            raise ValueError("Persisted manifest differs from this run configuration")
        manifest = previous
    else:
        manifest = {
            **manifest_core,
            "initial_checkpoint_next_idx": checkpoint_summary["next_idx"],
        }
        atomic_json(manifest, MANIFEST_PATH)

    logging.info("Loading and merging %s + %s", MODEL_ID, ADAPTER_NAME)
    model, dtype, gpu_name = load_model()
    free_vram, total_vram = torch.cuda.mem_get_info()
    logging.info(
        "Starting/resuming n100 fit: checkpoint=%s gpu=%s dtype=%s "
        "next_idx=%d n_done=%d free_vram=%.2f/%.2f GiB system_ram_available=%.2f GiB",
        CHECKPOINT_PATH,
        gpu_name,
        dtype,
        checkpoint_summary["next_idx"],
        checkpoint_summary["n_done"],
        free_vram / 1024**3,
        total_vram / 1024**3,
        psutil.virtual_memory().available / 1024**3,
    )

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    lens = jlens.fit(
        model,
        prompts,
        source_layers=SOURCE_LAYERS,
        target_layer=TARGET_LAYER,
        dim_batch=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        skip_first=SKIP_FIRST,
        checkpoint_path=str(CHECKPOINT_PATH),
        checkpoint_every=1,
        resume=True,
    )
    fit_seconds = time.perf_counter() - started
    if lens.n_prompts != N_PROMPTS:
        raise ValueError(f"Fit returned n_prompts={lens.n_prompts}, expected 100")

    temporary_lens = LENS_PATH.with_name(LENS_PATH.name + ".tmp")
    lens.save(str(temporary_lens), dtype=torch.float16)
    os.replace(temporary_lens, LENS_PATH)
    completion = {
        **manifest,
        "gpu": gpu_name,
        "dtype": str(dtype),
        "resumed_from_next_idx": checkpoint_summary["next_idx"],
        "fit_seconds_this_run": fit_seconds,
        "completed_n_prompts": lens.n_prompts,
        "peak_allocated_vram_gib": torch.cuda.max_memory_allocated() / 1024**3,
        "lens_path": str(LENS_PATH),
        "lens_size_bytes": LENS_PATH.stat().st_size,
        "lens_sha256": sha256_file(LENS_PATH),
        "completed_at_unix": time.time(),
    }
    atomic_json(completion, COMPLETE_PATH)
    logging.info("Completed n100 lens: %s", json.dumps(completion, indent=2))


main()
