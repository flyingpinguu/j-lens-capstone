"""Train or continue a rank-8 LoRA for prompt-injection resistance."""

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parent.parent
DEFAULT_TRAIN_FILE = ROOT / "data" / "training" / "qwen35-4b-pi-r8-stage1-120.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "outputs" / "qwen35-4b-lora-pi-r8-stage1-120"
MODEL_ID = "Qwen/Qwen3.5-4B"
TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "in_proj_qkv",
    "in_proj_z",
    "in_proj_b",
    "in_proj_a",
    "out_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
]


def pick_device_and_dtype():
    if torch.cuda.is_available():
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        return torch.device("cuda"), dtype
    if torch.backends.mps.is_available():
        return torch.device("mps"), torch.bfloat16
    return torch.device("cpu"), torch.bfloat16


def read_jsonl(path):
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def encode_training_example(tokenizer, row, max_length):
    messages = row["messages"]
    prompt_text = tokenizer.apply_chat_template(
        messages[:2],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    full_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False,
        enable_thinking=False,
    )
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
    if full_ids[: len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Chat-template prefix mismatch for {row['template_id']}")
    if len(full_ids) > max_length:
        raise ValueError(
            f"{row['template_id']} needs {len(full_ids)} tokens, above --max-length={max_length}"
        )
    labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
    if not any(token_id != -100 for token_id in labels):
        raise ValueError(f"No assistant tokens remain for {row['template_id']}")
    return {
        "input_ids": torch.tensor([full_ids], dtype=torch.long),
        "attention_mask": torch.ones((1, len(full_ids)), dtype=torch.long),
        "labels": torch.tensor([labels], dtype=torch.long),
    }


def save_checkpoint(model, optimizer, output_dir, epoch, summary, save_optimizer_state):
    checkpoint_dir = output_dir / f"checkpoint-epoch-{epoch}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(checkpoint_dir)
    (checkpoint_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    if save_optimizer_state:
        torch.save(
            {
                "epoch": epoch,
                "optimizer": optimizer.state_dict(),
                "python_random_state": random.getstate(),
                "torch_random_state": torch.random.get_rng_state(),
            },
            checkpoint_dir / "optimizer_state.pt",
        )
    return checkpoint_dir


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-file", type=Path, default=DEFAULT_TRAIN_FILE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--initial-adapter",
        type=Path,
        help="Continue adapter weights as a new phase with a fresh optimizer.",
    )
    parser.add_argument(
        "--resume-checkpoint",
        type=Path,
        help="Exactly resume a checkpoint that includes optimizer_state.pt.",
    )
    parser.add_argument("--save-optimizer-state", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.initial_adapter and args.resume_checkpoint:
        raise ValueError("Use either --initial-adapter or --resume-checkpoint, not both")
    if args.epochs < 1:
        raise ValueError("--epochs must be positive")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device, dtype = pick_device_and_dtype()
    print(f"Device: {device} | dtype: {dtype} | model: {args.model_id}", flush=True)

    rows = read_jsonl(args.train_file)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        local_files_only=args.local_files_only,
    )
    encoded_rows = [encode_training_example(tokenizer, row, args.max_length) for row in rows]
    lengths = [row["input_ids"].shape[1] for row in encoded_rows]
    print(
        f"Training examples: {len(rows)} | tokens min/mean/max: "
        f"{min(lengths)}/{sum(lengths) / len(lengths):.1f}/{max(lengths)}",
        flush=True,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        dtype=dtype,
        local_files_only=args.local_files_only,
        low_cpu_mem_usage=True,
    ).to(device)
    base_model.config.use_cache = False
    if hasattr(base_model, "gradient_checkpointing_enable"):
        base_model.gradient_checkpointing_enable()

    adapter_source = args.resume_checkpoint or args.initial_adapter
    if adapter_source:
        model = PeftModel.from_pretrained(
            base_model,
            adapter_source,
            is_trainable=True,
        )
    else:
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            target_modules=TARGET_MODULES,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(base_model, lora_config)
    model.enable_input_require_grads()
    model.print_trainable_parameters()

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.learning_rate, weight_decay=0.01)
    start_epoch = 1
    if args.resume_checkpoint:
        state_path = args.resume_checkpoint / "optimizer_state.pt"
        if not state_path.exists():
            raise FileNotFoundError(f"Exact-resume state not found: {state_path}")
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        optimizer.load_state_dict(state["optimizer"])
        random.setstate(state["python_random_state"])
        torch.random.set_rng_state(state["torch_random_state"])
        start_epoch = int(state["epoch"]) + 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_config = {
        "model_id": args.model_id,
        "train_file": str(args.train_file),
        "training_examples": len(rows),
        "epochs_requested": args.epochs,
        "start_epoch": start_epoch,
        "learning_rate": args.learning_rate,
        "gradient_accumulation": args.gradient_accumulation,
        "max_length": args.max_length,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": TARGET_MODULES,
        "seed": args.seed,
        "device": str(device),
        "dtype": str(dtype),
        "initial_adapter": str(args.initial_adapter) if args.initial_adapter else None,
        "resume_checkpoint": str(args.resume_checkpoint) if args.resume_checkpoint else None,
    }
    (args.output_dir / "run_config.json").write_text(
        json.dumps(run_config, indent=2) + "\n",
        encoding="utf-8",
    )

    model.train()
    optimizer.zero_grad(set_to_none=True)
    started = time.perf_counter()
    loss_history = []
    optimizer_steps = 0
    final_checkpoint = None
    final_epoch = start_epoch + args.epochs - 1
    for epoch in range(start_epoch, final_epoch + 1):
        order = list(range(len(encoded_rows)))
        random.shuffle(order)
        epoch_losses = []
        for step, row_index in enumerate(order, start=1):
            batch = {key: value.to(device) for key, value in encoded_rows[row_index].items()}
            loss = model(**batch).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite loss at epoch {epoch}, row {step}: {loss.item()}")
            (loss / args.gradient_accumulation).backward()
            loss_value = float(loss.detach().cpu())
            loss_history.append(loss_value)
            epoch_losses.append(loss_value)

            should_update = (
                step % args.gradient_accumulation == 0 or step == len(order)
            )
            if should_update:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                elapsed = time.perf_counter() - started
                recent_count = min(args.gradient_accumulation, len(epoch_losses))
                recent_loss = sum(epoch_losses[-recent_count:]) / recent_count
                total_updates = math.ceil(len(order) / args.gradient_accumulation)
                print(
                    f"epoch {epoch}/{final_epoch} update "
                    f"{math.ceil(step / args.gradient_accumulation)}/{total_updates} "
                    f"loss {recent_loss:.4f} elapsed {elapsed / 60:.1f}m",
                    flush=True,
                )

        summary = {
            **run_config,
            "completed_epoch": epoch,
            "optimizer_steps": optimizer_steps,
            "epoch_mean_loss": sum(epoch_losses) / len(epoch_losses),
            "overall_mean_loss": sum(loss_history) / len(loss_history),
            "elapsed_seconds": time.perf_counter() - started,
        }
        final_checkpoint = save_checkpoint(
            model,
            optimizer,
            args.output_dir,
            epoch,
            summary,
            args.save_optimizer_state,
        )
        print(f"Saved {final_checkpoint}", flush=True)

    model.save_pretrained(args.output_dir)
    final_summary = {
        **summary,
        "final_checkpoint": str(final_checkpoint),
    }
    (args.output_dir / "training_summary.json").write_text(
        json.dumps(final_summary, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(final_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
