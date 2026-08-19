"""Run M1-M4 on the Qwen3-14B strict single-turn attack cohort.

The source JSONL also contains 60 authorized requests. They are excluded
before either feature track and the shared category folds are built, so no
authorized row can enter training, tuning, model selection, or evaluation.
Stage 1 remains off because Colab already produced the lens readouts.
"""

from copy import deepcopy
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline import pipeline_main  # noqa: E402


RUN_TAG = "qwen3-14b-singleturn-strict-attacks-only"
INPUT_FILE = ROOT / "outputs" / "j-lens-run" / (
    "qwen3-14b-random-single-token-seed20260812-attack-authorized-"
    "sys_strict-full-corpus-last-16-prompt-plus-response-positions-top10.jsonl"
)
OUTPUT_DIR = ROOT / "outputs" / "pipeline" / RUN_TAG


def main():
    settings = deepcopy(pipeline_main.SETTINGS)
    settings.update({
        "active_model": "qwen3-14b",
        "model": pipeline_main.MODEL_CONFIGS["qwen3-14b"],
        "run_stage1": False,
        "analysis_input_file": INPUT_FILE,
        "analysis_output_dir": ROOT / "outputs" / "analysis" / RUN_TAG,
        "system_ids": ["sys_strict"],
        "analysis_include_labels": ["attack"],
        "run_leakage_audit": False,
    })

    # Qwen3-14B has 40 layers. Preserve the 4B pipeline's relative layer
    # coverage instead of silently reusing its absolute 16-31 indices.
    settings["analysis"] = {
        **settings["analysis"],
        "late_band": list(range(34, 40)),
        "mid_band": list(range(20, 34)),
        "heatmap_layers": list(range(20, 40)),
    }
    settings["multitoken"] = {
        **settings["multitoken"],
        "layers": list(range(20, 40)),
        "output_dir": OUTPUT_DIR,
    }
    # Request metadata is constant once authorized requests are removed;
    # M1 therefore fits only the meaningful lens feature set.
    settings["m1"] = {
        **settings["m1"],
        "feature_sets": {"lens": pipeline_main.LENS_FEATURES},
        "output_dir": OUTPUT_DIR,
    }
    settings["m3"] = {**settings["m3"], "output_dir": OUTPUT_DIR}

    pipeline_main.SETTINGS = settings
    pipeline_main.CLASSIFIER_OUTPUT_DIR = OUTPUT_DIR
    pipeline_main.RUN_TAG = RUN_TAG
    pipeline_main.main()


if __name__ == "__main__":
    main()
