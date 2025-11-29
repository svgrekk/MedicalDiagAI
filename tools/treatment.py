# tools/treatment.py

import traceback
from typing import List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from langchain.tools import tool
from llm.ollama_chat_llama import get_llm
from langchain_community.retrievers import PubMedRetriever
from langchain_core.runnables import RunnablePassthrough

llm = get_llm()

retriever = PubMedRetriever(top_k_results=8, doc_content_chars_max=2000)


class TreatmentArgs(BaseModel):
    disease: str = Field("", description="The predicted disease (optional)")
    symptoms: List[str] = Field(..., description="List of known symptoms")


class TreatmentResponse(BaseModel):
    recommendations: str = Field(
        ...,
        description="Treatment recommendations as a formatted string"
    )


treatment_parser = JsonOutputParser(pydantic_object=TreatmentResponse)

TREATMENT_PROMPT = """
Based on PubMed literature, provide general treatment recommendations{disease_prompt} given symptoms: {symptoms}.
Focus on evidence-based treatments, home remedies if applicable, and when to seek medical help.
Output as JSON with key 'recommendations' containing a concise paragraph.
{format_instructions}
"""

treatment_prompt = (ChatPromptTemplate.from_template(TREATMENT_PROMPT).
                    partial(format_instructions=treatment_parser.get_format_instructions()))

treatment_chain = (
        RunnablePassthrough.assign(
            context=lambda x: (
                    print(f"[TREATMENT RAG] retriever with query: {x['query']}") or
                    "\n\n".join(
                        (lambda d: (
                                print(f"[TREATMENT RAG] Document: {d.page_content[:100]}...") or
                                d.page_content
                        )[1])(d) for d in retriever.invoke(x["query"])
                    )
            ),
            disease_prompt=lambda x: f" for {x['disease']}" if x["disease"] else ""
        )
        | treatment_prompt
        | llm
        | treatment_parser
)


@tool("get_treatment_recommendation", return_direct=False, args_schema=TreatmentArgs)
def get_treatment_recommendation(disease: str, symptoms: List[str]) -> str:
    """Fetches treatment recommendations using PubMed RAG."""
    print(f"[TOOL CALL] get_treatment_recommendation for {disease}")
    if disease:
        query = f"{disease} treatment guidelines symptoms {' AND '.join(symptoms)}"
    else:
        query = f"treatment guidelines for symptoms {' AND '.join(symptoms)}"

    try:
        response = treatment_chain.invoke({
            "query": query,
            "disease": disease,
            "symptoms": ", ".join(symptoms)
        })
        recommendations = response.get("recommendations", "No specific recommendations found.")
    except Exception as e:
        print(f"[TREATMENT ERROR] {e}")
        traceback.print_exc()
        recommendations = "General advice: Rest, hydrate, and consult a doctor for personalized treatment."

    print(f"[TOOL RETURN] {recommendations}")
    return recommendations


# Тест
if __name__ == "__main__":
    # Test treatment
    treatment_out = get_treatment_recommendation.invoke({"disease": "flu", "symptoms": ["fever", "headache"]})
    print(treatment_out)

    # Test without disease
    treatment_out_no_disease = get_treatment_recommendation.invoke({"disease": "", "symptoms": ["fever", "headache"]})
    print(treatment_out_no_disease)