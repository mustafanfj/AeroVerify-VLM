# ── Parameters (override any with environment variables AV_*) ─────────────────
import os, json
from pathlib import Path

def _env(name, default, cast=str):
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    if cast is bool:
        return v.strip().lower() in ("1", "true", "yes", "y")
    if cast is list:
        return [s.strip() for s in v.split(",") if s.strip()]
    return cast(v)

DRY_RUN       = _env("AV_DRY_RUN", False, bool)          # synthetic data + mock VLMs (smoke test)
PROJECT_ROOT  = _env("PROJECT_ROOT", str(Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()))   # repo root
STAGES        = _env("AV_STAGES", ["data", "run", "analyze"], list)
DATASETS      = _env("AV_DATASETS", ["floodnet", "rsvqa_lr"], list)   # primary first
MODELS_TO_RUN = _env("AV_MODELS", ["moondream2", "smolvlm"], list)
MODELS_TO_ANALYZE = _env("AV_ANALYZE_MODELS", ["moondream2", "smolvlm"], list)
RUN_ROBUSTNESS = _env("AV_ROBUSTNESS", True, bool)
ROBUST_DATASETS = ["floodnet"]                               # UAV perturbations are applied to UAV imagery

# Samples (DEV → reliability weights + thresholds only; EVAL → every reported number)
SAMPLE_SIZES = {
    "floodnet": dict(dev=_env("AV_FN_DEV", 60, int), eval=_env("AV_FN_EVAL", 300, int)),
    "rsvqa_lr": dict(dev=40, eval=160),                    # identical to run 1
}
ROBUST_N_IMAGES = _env("AV_N_ROBUST", 40, int)
SEED = 2026

# Evidence
MODEL_SIDE      = 768     # nominal VLM input side (px) used by the resolution gate
GATE_MIN_ZOOM   = 1.0     # native tiles are used only if tile_side / MODEL_SIDE ≥ this
GRID            = 2       # 2×2 tiles
LR_CROP_SIZE    = 512     # E3 baseline on RSVQA-LR upsamples 128-px crops (gate keeps them out of E4)
MAX_BOXES       = 16      # grounded counting: instances verified one by one
DET_MAX_OBJECTS = 100

# Decision rule
MIN_COVERAGE = 0.80       # τ = most accurate threshold on DEV that still answers ≥ 80 %

# E7 robustness (UAV-specific)
PERTURBATIONS = {
    "altitude":    [0.5, 0.25, 0.125],   # down-sample factor (simulates higher flight altitude)
    "motion_blur": [15, 31, 61],         # horizontal blur length at native resolution (px)
    "haze":        [0.3, 0.5, 0.7],      # blend with atmospheric light
    "jpeg":        [30, 15, 5],          # JPEG quality (video link compression)
}

MODEL_SPECS = {
    "moondream2": dict(hf_id="vikhyatk/moondream2", revision="2025-06-21", label="Moondream2"),
    "smolvlm":    dict(hf_id="HuggingFaceTB/SmolVLM-Instruct", revision="main", label="SmolVLM",
                       longest_edge=768),
}
GROUNDER = dict(hf_id="vikhyatk/moondream2", revision="2025-06-21")   # shared grounding module
MAX_NEW_TOKENS = 24
FLOODNET_HF = "takara-ai/FloodNet_2021-Track_2_Dataset_HF"

print(json.dumps(dict(DRY_RUN=DRY_RUN, STAGES=STAGES, DATASETS=DATASETS, MODELS_TO_RUN=MODELS_TO_RUN,
                      SAMPLE_SIZES=SAMPLE_SIZES, RUN_ROBUSTNESS=RUN_ROBUSTNESS), indent=1))
