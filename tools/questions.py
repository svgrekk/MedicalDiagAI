# tools/questions.py

from typing import Dict, Any, List
import traceback
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.tools import tool
from llm.ollama_chat_llama import get_llm
from config import SYMPTOM_JSON
import json
from langchain_community.retrievers import PubMedRetriever
from langchain_core.runnables import RunnablePassthrough
from prompts import RAG_PROMPT

llm = get_llm()

with open(SYMPTOM_JSON, 'r') as f:
    ALL_SYMPTOMS = json.load(f)

retriever = PubMedRetriever(top_k_results=8, doc_content_chars_max=2000)


class QuestionsResponse(BaseModel):
    mapping: Dict[str, str] = Field(
        ...,
        description="Symptom from dataset → natural yes/no question"
    )


parser = JsonOutputParser(pydantic_object=QuestionsResponse)

PROMPT = (ChatPromptTemplate.from_template(RAG_PROMPT).
          partial(format_instructions=parser.get_format_instructions()))

rag_chain = (
        RunnablePassthrough.assign(
            context=lambda x: (
                    print(f"[RAG] retriever с query: {x['query']}") or
                    "\n\n".join(
                        (lambda d: (
                                print(f"[RAG] Document: {d.page_content[:100]}...") or
                                d.page_content
                        )[1])(d) for d in retriever.invoke(x["query"])
                    )
            )
        )
        | PROMPT
        | llm
        | parser
)




class ClarifyingArgs(BaseModel):
    predictions: Dict[str, Any] = Field(...)
    known_symptoms: List[str] = Field(default_factory=list)
    exclude: List[str] = Field(default_factory=list)


def _select_symptoms(preds: Dict[str, Any], known: List[str], denied: List[str]) -> List[str]:
    print(f"[DEBUG] _select_symptoms: diseases={preds.get('diseases', [])[:3]}")
    diseases = preds.get("diseases", [])[:3]
    available = [s for s in ALL_SYMPTOMS if s not in known and s not in denied]
    print(f"[DEBUG] Available symptoms after exclude: {len(available)}")

    candidates = [s for s in available if any(d.lower() in s.lower() for d in diseases)]
    if len(candidates) < 3:
        from random import shuffle
        shuffle(available)
        candidates.extend(available[:3 - len(candidates)])

    selected = candidates[:3]
    print(f"[DEBUG] Chosen: {selected}")
    return selected


# --- @tool ---
@tool("ask_additional_question", return_direct=False, args_schema=ClarifyingArgs)
def ask_additional_symptoms(
        predictions: Dict[str, Any],
        known_symptoms: List[str],
        exclude: List[str],
) -> Dict[str, List[str]]:
    """RAG + real symptoms → questions"""
    print(f"[TOOL CALL] ask_additional_symptoms")
    print(f"[INPUT] diseases: {predictions.get('diseases', [])}")
    print(f"[INPUT] known: {known_symptoms}")
    print(f"[INPUT] exclude: {exclude}")

    selected = _select_symptoms(predictions, known_symptoms, exclude)
    available_str = ", ".join(selected)

    diseases = predictions.get("diseases", [])
    query = (
        "Focus on diseases that are most commonly co-occurring with the provided symptoms according to PubMed context. "
        "Avoid proposing completely unrelated organ systems (e.g., ear bleeding for stomach pain). "
        "If the evidence is unclear, prefer common infectious or inflammatory diseases. "
        f"{' AND '.join(known_symptoms)} differential diagnosis"
    )

    print(f"[RAG] Query PubMed: {query}")

    try:
        print("[RAG]  rag_chain.invoke...")
        response = rag_chain.invoke({
            "query": query,
            "known": ", ".join(known_symptoms),
            "denied": ", ".join(exclude),
            "available": available_str
        })
        print(f"[RAG] raw LLM answer: {response}")
        mapping = response.get("mapping", {}) if isinstance(response, dict) else {}
        symptoms_to_ask = list(mapping.keys())
        questions = list(mapping.values())
        print(f"[RAG] Questions: {questions}")
    except Exception as e:
        print(f"[RAG ERROR] {e}")
        traceback.print_exc()
        symptoms_to_ask = selected
        questions = [f"Do you have {s.replace('_', ' ')}?" for s in selected]
        print(f"[FALLBACK] Questions: {questions}")

    result = {
        "symptoms_to_ask": symptoms_to_ask,
        "questions": questions
    }
    print(f"[TOOL RETURN] {result}")
    return result


# Тест
if __name__ == "__main__":
    demo = {
        "diseases": ['lyme disease', 'aphthous ulcer', 'flu', 'gastritis', 'mononucleosis'],
        "probabilities": [0.2129063755273819, 0.0957469791173935, 0.06782759726047516, 0.04641585052013397,
                          0.03959854692220688]}
    out = ask_additional_symptoms.invoke({
        "predictions": demo,
        "known_symptoms": ["headache", "fever"],
        "exclude": []
    })
    print(out)
