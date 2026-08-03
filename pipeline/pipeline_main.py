"""Pipeline entry point -- see docs/pipeline_architecture.md.

Loads all pipeline inputs, holds every adjustable parameter in SETTINGS,
then runs the stages in order. Stage scripts live in numbered directories
(1_data_generation, ...), which are not valid Python package names, so they
are loaded by file path via load_stage() instead of a regular import.

Run from the repo root (or anywhere):  python pipeline/pipeline_main.py
"""

import importlib.util
import json
from pathlib import Path

PIPELINE_DIR = Path(__file__).resolve().parent
ROOT = PIPELINE_DIR.parent

# --------------------------------------------------------------- inputs
INJECTION_FILE = ROOT / "data" / "evaluation" / "injection_corpus.jsonl"
SYSTEM_PROMPTS_FILE = ROOT / "data" / "evaluation" / "system_prompts.jsonl"

# ------------------------------------------------------------- settings
MODEL_CONFIGS = {
    "qwen35-4b": {
        "model_id": "Qwen/Qwen3.5-4B",
        "dtype": "bfloat16",
        "chat_kwargs": {"enable_thinking": False},
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": (
            "qwen3.5-4b/jlens/Salesforce-wikitext/"
            "Qwen3.5-4B_jacobian_lens_n1000.pt"
        ),
        "lens_revision": "qwen-n1000",
    },
    "qwen36-27b": {
        "model_id": "Qwen/Qwen3.6-27B",
        "dtype": "float16",
        "chat_kwargs": {"enable_thinking": False},
        "lens_repo": "neuronpedia/jacobian-lens",
        "lens_file": (
            "qwen3.6-27b/jlens/Salesforce-wikitext/"
            "Qwen3.6-27B_jacobian_lens_n1000.pt"
        ),
        "lens_revision": "qwen-n1000",
    },
}

ACTIVE_MODEL = "qwen35-4b"

SETTINGS = {
    # --- model + lens (a second model is just a different entry here;
    #     it needs its own fitted lens, see AGENTS.md)
    "active_model": ACTIVE_MODEL,
    "model": MODEL_CONFIGS[ACTIVE_MODEL],

    # --- secret (must tokenize to exactly one token; the harness verifies)
    "secret": "banana",

    # --- corpus scope: None runs the full corpus in file order; an int
    #     samples that many rows without replacement using random_seed
    #     (the same sample is paired with every system prompt)
    "max_prompts_per_strictness": None,
    "random_seed": 42,

    # --- readouts. readout_positions is the main run-time lever, readout
    #     cost scales ~linearly with position count:
    #     "last" / "last_n" / "user" / "user_response" / "prompt" / "all"
    "readout_positions": "user_response",
    "readout_last_n": 5,
    "top_k": 10,
    "probe_enabled": True,
    "probe_token": None,  # None -> defaults to the secret

    # --- generation
    "max_new_tokens": 64,

    # --- output (same location and naming scheme as the notebook runs,
    #     so resume files stay compatible)
    "output_dir": ROOT / "outputs" / "j-lens-run",

    # --- stage toggles. Stage 1 loads the full model, so it stays off when
    #     the run data already exists (its resume logic makes re-running
    #     safe but pointlessly slow for analysis-only work).
    "run_stage1": False,

    # --- Stage-2 analysis. Input defaults to the relabeled run file
    #     (corrected attack_successful labels, see
    #     notebooks/analysis_alex/classifier-findings.md); set to None to
    #     use the Stage-1 output file instead. Band/window parameters are
    #     per-model config: for a second model, re-derive them from that
    #     model's own curves (docs/pipeline_architecture.md).
    "analysis_input_file": ROOT / "outputs" / "j-lens-run" / (
        "qwen35-4b-full-corpus-user-response-positions-top10-relabeled.jsonl"
    ),
    "analysis_output_dir": ROOT / "outputs" / "analysis",
    "analysis": {
        "late_band": list(range(27, 32)),   # level/peak band: where the per-layer curves strengthen
        "mid_band": list(range(16, 27)),    # emergence band: where the signal departs from chance
        "user_window_n": 8,                 # fixed end-window of user positions
        "peak_quantile": 0.10,              # near-minimum, robust to outlier cells
        "scaffold_ref_n": 3,                # graph 1 reference: mean over last N scaffolding tokens
        "heatmap_layers": list(range(17, 32)),
    },
}


def load_stage(relative_path):
    """Import a stage script by file path (numbered directories can't be
    imported as packages)."""
    path = PIPELINE_DIR / relative_path
    spec = importlib.util.spec_from_file_location(
        path.stem.replace("-", "_"), path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def read_jsonl(path):
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main():
    injections = read_jsonl(INJECTION_FILE)
    system_prompts = read_jsonl(SYSTEM_PROMPTS_FILE)
    print(f"Inputs: {len(injections)} corpus rows, {len(system_prompts)} system prompts")

    # --- Stage 1: data generation (resume-safe: the harness skips ids
    #     already present in the output file)
    run_data_file = SETTINGS["analysis_input_file"]
    if SETTINGS["run_stage1"]:
        run_harness = load_stage("1_data_generation/run_harness.py")
        stage1_output = run_harness.run(SETTINGS, injections, system_prompts)
        print("Stage 1 complete:", stage1_output)
        if run_data_file is None:
            run_data_file = stage1_output

    # --- Stage 2.1: single-token analysis -- two figures + the F1-F4
    #     feature table (also written to CSV as the hand-off artifact)
    single_token_analysis = load_stage("2_EDA_and_FE/single_token_analysis.py")
    single_token_features = single_token_analysis.run(SETTINGS, run_data_file)
    print("Stage 2.1 complete:", single_token_features.shape[0], "runs,",
          single_token_features.shape[1], "columns")

    # --- Stage 2.2: top-k readout analysis (2_EDA_and_FE/) -- Alex's track,
    #     plugs in here.
    # --- Stage 3: model comparison (3_model_predictions/) -- depends on
    #     both Stage-2 tracks; see docs/pipeline_architecture.md.


if __name__ == "__main__":
    main()
