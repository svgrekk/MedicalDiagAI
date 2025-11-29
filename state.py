# state.py

from dataclasses import dataclass, field
from typing import List, Dict, Any


@dataclass
class DiagnosticState:
    known_symptoms: List[str] = field(default_factory=list)
    known_indexes: List[int] = field(default_factory=list)
    known_negatives: List[str] = field(default_factory=list)

    pending_question: Dict[str, Any] = field(default_factory=dict)

    predictions: Dict[str, Any] = field(default_factory=dict)

    asked_history: List[Dict[str, Any]] = field(default_factory=list)
    chat_history: List[Dict[str, str]] = field(default_factory=list)  # для RAG

    top_probability: float = 0.0
    finished: bool = False
