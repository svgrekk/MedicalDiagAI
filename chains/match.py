# chains/match.py
from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.runnables import RunnableLambda


from tools.symptom_tool import match_symptoms, get_vector


def build_match_chain():
    """
    Вход:
      {
        "phrases": List[str],
        "known_symptoms": List[str],
        "known_indexes":  List[int],
      }

    Выход:
      {
        "matched_symptoms": List[str],
        "indexes": List[int],
        "known_symptoms": List[str],
        "known_indexes":  List[int],
      }
    """
    def _call(inputs: Dict[str, Any]) -> Dict[str, Any]:
        phrases: List[str] = inputs.get("symptoms_list", []) or []
        known_syms: List[str] = inputs.get("known_symptoms", []) or []
        known_idxs: List[int] = inputs.get("known_indexes", []) or []


        res = match_symptoms.invoke({"symptoms_list": phrases})
        batch_syms: List[str] = res["matched_symptoms"]
        batch_idxs: List[int] = res["indexes"]


        seen = set(known_syms)
        for s, i in zip(batch_syms, batch_idxs):
            if s not in seen:
                known_syms.append(s)
                known_idxs.append(i)
                seen.add(s)

        return {
            "matched_symptoms": batch_syms,
            "indexes": batch_idxs,
            "known_symptoms": known_syms,
            "known_indexes": known_idxs,
        }

    return RunnableLambda(_call)


def build_vectorize_chain():
    """
    Input:  {"known_indexes": List[int]}
    Output: {"vector": List[List[int]]}
    """
    def _call(inputs: Dict[str, Any]) -> Dict[str, Any]:
        idxs: List[int] = inputs.get("known_indexes", []) or []
        out = get_vector.invoke({"idx": idxs})
        return {"vector": out["vector"]}
    return RunnableLambda(_call)
