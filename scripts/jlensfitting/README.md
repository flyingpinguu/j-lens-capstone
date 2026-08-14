# LoRA Jacobian-lens fitting

These scripts fit a separate Jacobian Lens for the merged
`Qwen/Qwen3.5-4B` + `qwen35-4b-lora-pi-r8-stage1-120` model.

The fitting corpus is generic WikiText-103 text, not prompt-injection
evaluation data. The production fit uses source layers L0-L30, target L31,
and 128-token sequences to match the published base-model lens setup.

## 1. Benchmark memory and throughput

```bash
HF_LOCAL_FILES_ONLY=1 .venv/bin/python \
  scripts/jlensfitting/benchmark_lora_jlens.py \
  --dim-batch 4 \
  --test-passes 4 \
  --overnight-hours 10 \
  --local-files-only
```

`dim_batch` replicates the prompt along the batch axis. Increasing it uses
more accelerator memory and reduces the number of backward launches, while
the total backward FLOPs remain approximately constant.

## 2. Run the resumable fit

Choose `N` from the benchmark and prevent macOS sleep:

```bash
caffeinate -dimsu .venv/bin/python scripts/jlensfitting/fit_lora_jlens.py \
  --n-prompts N \
  --dim-batch 4 \
  --checkpoint-every 1 \
  --local-files-only
```

Re-running the same command resumes from the atomic checkpoint. The prompt
cache and manifest prevent accidentally resuming with different data or
estimator settings.

## Colab A100 capacity test

`colab_a100_dim_batch_test.py` is a capacity-only Colab test. It runs two
real Jacobian backward passes per candidate but deliberately does not fit a
lens. On an NVIDIA A100-SXM4-40GB with BF16 and 128-token WikiText input, the
LoRA-merged Qwen3.5-4B model produced:

| `dim_batch` | peak GPU RAM | headroom | estimated seconds/prompt |
| ---: | ---: | ---: | ---: |
| 12 | 29.70 GiB | 9.80 GiB | 128.8 |
| 14 | 33.26 GiB | 6.23 GiB | 109.6 |
| 16 | 36.83 GiB | 2.66 GiB | 102.3 |
| 20 | OOM | — | — |

Use `dim_batch=14` for a long run on this 40 GB A100 profile. It retains a
reasonable margin while being about 15% faster than 12 in this benchmark.
The exact-shape checkpoint persistence test also succeeded: an 812,656,786
byte synthetic jlens checkpoint was atomically written to mounted Google
Drive, reloaded with `weights_only=True`, and SHA-256 verified. No lens was
fitted during either test.
