# chains/router.py
from langchain_core.tools import tool
from typing import Literal


@tool("intent_classifier")
def classify_intent(message: str) -> Literal["symptom", "question", "command"]:
    """
    Classify user message into one of three intents:
    - symptom: user describes symptoms
    - question: user asks for explanation or info
    - command: exit, restart, etc.
    """
    message_lower = message.lower().strip()

    # === COMMAND ===
    if message_lower in ["exit", "quit", "stop", "restart"]:
        return "command"

    # === QUESTION ===
    if any(q in message_lower for q in ["what is", "что такое", "explain", "why", "how", "?"]):
        return "question"

    # === SYMPTOM ===
    if any(s in message_lower for s in ["i have", "pain", "hurt", "feel", "headache", "chest", "fever", "cough"]):
        return "symptom"

    # fallback
    return "symptom"


# === BUILD FUNCTION ===
def build_router(llm=None):
    """
    Returns a LangChain Tool that can be used with .invoke()
    """
    return classify_intent
