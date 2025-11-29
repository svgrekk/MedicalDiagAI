# tools/xgb_tool.py
import xgboost as xgb
import joblib
from langchain.tools import tool
from config import XGB_JSON_MODEL, LABEL_ENCODER
from typing import List, Dict, Any
import numpy as np
from pydantic import BaseModel, Field

# --- Load model and encoder ---
try:
    model = xgb.XGBClassifier()
    model.load_model(XGB_JSON_MODEL)
    print(f"[XGB] Model loaded from {XGB_JSON_MODEL}")
except Exception as e:
    print(f"[XGB ERROR] Failed to load model: {e}")
    model = None

try:
    le = joblib.load(LABEL_ENCODER)
    print(f"[XGB] Encoder loaded from {LABEL_ENCODER}")
    print(f"[XGB] Number of classes: {len(le.classes_)}")
except Exception as e:
    print(f"[XGB ERROR] Failed to load encoder: {e}")
    le = None

print(f"Loaded Booster model: {XGB_JSON_MODEL.name}")
print(f"Loaded LabelEncoder: {LABEL_ENCODER.name}")


class PredictArgs(BaseModel):
    vector: list[list[int]] = Field(...)


@tool("disease_prediction", return_direct=False, args_schema=PredictArgs)
def get_predictions(vector: List[List[int]]) -> Dict[str, Any]:
    """Predict top-5 diseases for a given feature vector."""
    print(f"[TOOL CALL] disease_prediction")
    proba = model.predict_proba(vector)
    top_idx = proba[0].argsort()[::-1][:5]
    print(f"[disease_prediction] top_idx: {top_idx}")
    top_probs = proba[0][top_idx].tolist()
    print(f"[disease_prediction] top_probs: {top_probs}")
    labels = le.inverse_transform(top_idx).tolist()
    print(f"[disease_prediction] labels: {labels}")
    out = {"diseases": labels, "probabilities": top_probs}
    print(f"[TOOL RETURN] disease_prediction : {out}", flush=True)
    return out


