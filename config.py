# config.py
from pathlib import Path
import os
import torch
import warnings

# === Optional tokens / env ===
GROQ_TOKEN = os.environ.get("GROQ_API_KEY")

# === Base directories (relative to this file) ===
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"

TOOLS_DIR = DATA_DIR / "tools"
ASSET_DIR = DATA_DIR / "assets"

SYMPTOM_JSON = ASSET_DIR / "symptoms.json"
SYMPTOM_EMBEDS = ASSET_DIR / "symptom_embeddings.npy"
DIAG_FEATURE_IMPORTANCE = ASSET_DIR / "diags_feature_importance.json"


XGB_JSON_MODEL = ASSET_DIR / "xgb_model.json"
XGB_SKLEARN_MODEL = ASSET_DIR / "xgb_classifier.joblib"


LABEL_ENCODER = ASSET_DIR / "label_encoder.joblib"


# === Device detection (CUDA → MPS → CPU) ===
def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


DEVICE = get_device()


# === Ensure required directories exist (idempotent) ===
def ensure_dirs():
    for p in [DATA_DIR, ASSET_DIR]:
        p.mkdir(parents=True, exist_ok=True)


ensure_dirs()

# === Import-time log ===
print(f"Config loaded. Base dir: {BASE_DIR}")
print(f"Data: {DATA_DIR}")
print(f"Assets dir: {ASSET_DIR}")
print(f"Device: {DEVICE}")

# Quiet specific warnings (same as before)
warnings.filterwarnings("ignore", category=FutureWarning, module="huggingface_hub")
