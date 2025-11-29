# llm/ollama_chat_llama.py
from langchain_ollama import ChatOllama


def get_llm(model_name: str = "llama3.1", temperature: float = 0.0) -> ChatOllama:
    return ChatOllama(model=model_name, temperature=temperature)
