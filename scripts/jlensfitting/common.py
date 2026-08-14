"""Shared model and prompt loading for LoRA Jacobian-lens fitting."""

import json
import os
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import jlens
import torch
from jlens.examples import load_wikitext_prompts
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
MODEL_ID = "Qwen/Qwen3.5-4B"
ADAPTER_DIR = ROOT / "outputs" / "qwen35-4b-lora-pi-r8-stage1-120"
OUTPUT_DIR = ROOT / "outputs" / "jlens" / "qwen35-4b-lora-pi-r8-stage1-120"


def pick_device_and_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.bfloat16


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize()
    elif device.type == "mps":
        torch.mps.synchronize()


def memory_snapshot(device):
    if device.type == "cuda":
        return {
            "allocated_gib": torch.cuda.memory_allocated() / 1024**3,
            "reserved_gib": torch.cuda.memory_reserved() / 1024**3,
            "peak_allocated_gib": torch.cuda.max_memory_allocated() / 1024**3,
            "device_limit_gib": torch.cuda.get_device_properties(0).total_memory
            / 1024**3,
        }
    if device.type == "mps":
        return {
            "allocated_gib": torch.mps.current_allocated_memory() / 1024**3,
            "driver_allocated_gib": torch.mps.driver_allocated_memory() / 1024**3,
            "device_limit_gib": torch.mps.recommended_max_memory() / 1024**3,
        }
    return {}


def load_merged_lora_lens_model(
    *,
    model_id=MODEL_ID,
    adapter_dir=ADAPTER_DIR,
    local_files_only=False,
):
    """Load Qwen, merge the selected LoRA, and wrap the merged HF model."""
    adapter_dir = Path(adapter_dir)
    adapter_config_path = adapter_dir / "adapter_config.json"
    if not adapter_config_path.exists():
        raise FileNotFoundError(f"LoRA adapter config not found: {adapter_config_path}")
    adapter_config = json.loads(adapter_config_path.read_text(encoding="utf-8"))
    adapter_base = adapter_config.get("base_model_name_or_path")
    if adapter_base != model_id:
        raise ValueError(
            f"Adapter expects base model {adapter_base!r}, not requested {model_id!r}"
        )

    device, dtype = pick_device_and_dtype()
    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    peft_model = PeftModel.from_pretrained(base_model, adapter_dir)
    if not hasattr(peft_model, "merge_and_unload"):
        raise TypeError(f"Loaded PEFT model cannot be merged: {type(peft_model).__name__}")
    merged_model = peft_model.merge_and_unload(safe_merge=True)
    lens_model = jlens.from_hf(merged_model, tokenizer)
    return tokenizer, merged_model, lens_model, device, dtype


def load_or_create_wikitext_prompts(path, n_prompts):
    """Reuse an exact prompt cache, or stream and save one deterministically."""
    path = Path(path)
    if path.exists():
        with path.open(encoding="utf-8") as handle:
            prompts = [json.loads(line)["text"] for line in handle if line.strip()]
        if len(prompts) != n_prompts:
            raise ValueError(
                f"Prompt cache {path} has {len(prompts)} rows, expected {n_prompts}"
            )
        return prompts

    prompts = load_wikitext_prompts(n_prompts=n_prompts)
    if len(prompts) != n_prompts:
        raise ValueError(f"WikiText returned {len(prompts)} prompts, expected {n_prompts}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, text in enumerate(prompts):
            handle.write(json.dumps({"index": index, "text": text}) + "\n")
    return prompts
