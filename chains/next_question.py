# chains/next_question.py
from __future__ import annotations
from typing import Dict, Any, List
from pydantic import BaseModel, Field
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.runnables import RunnableLambda
from langchain_core.language_models.chat_models import BaseChatModel
from prompts import FOLLOWUP_QUESTIONS_PROMPT
from tools.questions import ask_additional_symptoms


class QuestionsResponse(BaseModel):
    """Mapping from symptom to yes/no question"""
    mapping: Dict[str, str] = Field(
        ...,
        description="Exact symptom name → clear yes/no question. Must include ALL symptoms."
    )


def build_next_question_chain(llm: BaseChatModel, limit: int = 3):
    parser = PydanticOutputParser(pydantic_object=QuestionsResponse)

    prompt = PromptTemplate.from_template(FOLLOWUP_QUESTIONS_PROMPT).partial(
        format_instructions=parser.get_format_instructions())

    def _pick_symptoms(d: Dict[str, Any]) -> List[str]:
        preds = d.get("preds", {})
        known = d.get("known_symptoms", []) or []
        exclude = d.get("exclude", []) or []

        out = ask_additional_symptoms.invoke({
            "predictions": {
                "diseases": preds.get("diseases", []),
                "probabilities": preds.get("probabilities", [])
            },
            "known_symptoms": known,
            "exclude": exclude
        }) or {}

        candidates = out.get("symptoms_to_ask", [])
        return [s for s in candidates if s not in exclude][:limit]

    def _run(d: Dict[str, Any]) -> Dict[str, Any]:
        symptoms = _pick_symptoms(d)
        if not symptoms:
            return {"symptoms_to_ask": [], "questions": []}

        symptoms_list = "\n".join(f"- {s}" for s in symptoms)
        chain = prompt | llm | parser

        try:
            response: QuestionsResponse = chain.invoke({"symptoms_list": symptoms_list})
            mapping = response.mapping

            questions = []
            missing = [s for s in symptoms if s not in mapping]
            if missing:
                print(f"[NEXT_QUESTION FALLBACK] Missing symptoms in LLM response: {missing}")

            for s in symptoms:
                if s in mapping:
                    questions.append(mapping[s])
                else:
                    # Custom handling for tricky symptoms
                    if s == "spotting urination":
                        questions.append("Have you noticed any spotting or blood in your urine?")
                    else:
                        questions.append(f"Do you have {s.replace('_', ' ')}?")

        except Exception as e:
            print(f"[NEXT_QUESTION FALLBACK] {e}")
            questions = []
            for s in symptoms:
                if s == "spotting urination":
                    questions.append("Have you noticed any spotting or blood in your urine?")
                else:
                    questions.append(f"Do you have {s.replace('_', ' ')}?")

        return {
            "symptoms_to_ask": symptoms,
            "questions": questions,
        }

    return RunnableLambda(_run)