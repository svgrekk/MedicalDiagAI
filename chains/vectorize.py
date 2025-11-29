# chains/vectorize.py
from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda


# твой tool из tools/symptom_tools.py
from tools.symptom_tool import get_vector

def build_vectorize_chain():
    """
    Input:  {"known_indexes": List[int]}
    Output: {"vector": List[List[int]]}
    """

    def _call(inputs: Dict[str, Any]) -> Dict[str, Any]:
        idxs = inputs.get("known_indexes") or inputs.get("indexes", [])
        print(f"[VECTORIZER] Received indexes: {idxs}")

        out = get_vector.invoke({"idx": idxs})
        return {"vector": out["vector"]}
    return RunnableLambda(_call)
