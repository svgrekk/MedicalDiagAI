# chains/predict.py
from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda

from tools.xgb_tool import get_predictions


def build_predict_chain():
    """
    Input:  {"vector": List[List[float]]}  # 2D, как из get_vector
    Output: {"diseases": [...], "probabilities": [...]}
    """
    return RunnableLambda(lambda d: get_predictions.invoke({"vector": d["vector"]}))
