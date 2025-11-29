# chains/extract.py
from __future__ import annotations
from typing import List, Dict, Any

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.language_models.chat_models import BaseChatModel

from pydantic import BaseModel, Field
from prompts import EXTRACTION_PROMPT


class ExtractedSymptoms(BaseModel):
    symptoms: List[str] = Field(..., description="short, lowercase phrases")


def build_extract_chain(llm: BaseChatModel):
    """
    Input:  {"complaint": str}
    Output: {"symptoms": List[str]}  # exact user phrases (no normalisation)
    """
    # Create JSON parser with Pydantic schema
    parser = JsonOutputParser(pydantic_object=ExtractedSymptoms)

    # Build prompt (your original EXTRACTION_PROMPT already contains {format_instructions})
    prompt = PromptTemplate(
        template=EXTRACTION_PROMPT,
        input_variables=["complaint"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    # Chain: prompt → LLM → JSON parser
    chain = prompt | llm | parser


    def _normalize(parsed: ExtractedSymptoms | Dict[str, Any]) -> Dict[str, Any]:
        print("\n[DEBUG] Entering _normalize()")
        print(f"[DEBUG] Raw parser output: {repr(parsed)} (type: {type(parsed)})")

        # Convert Pydantic object to dict if needed
        data = parsed if isinstance(parsed, dict) else parsed.dict()
        print(f"[DEBUG] Data after conversion: {data}")

        raw_symptoms = data.get("symptoms", [])
        print(f"[DEBUG] Raw symptoms from LLM: {raw_symptoms}")

        # Keep EXACT user phrases – only strip whitespace and deduplicate
        seen = set()
        out = []
        for s in raw_symptoms:
            if isinstance(s, str):
                t = s.strip()                # keep original case
                if t and t not in seen:
                    seen.add(t)
                    out.append(t)
        print(f"[DEBUG] Final symptoms (deduped, no lowercasing): {out}")
        print("-" * 70)

        return {"symptoms": out}

    # Wrap with RunnableLambda
    from langchain_core.runnables import RunnableLambda as _Lambda
    final_chain = chain | _Lambda(_normalize)
    print("[DEBUG] extract_chain ready – returning final chain\n")
    return final_chain