from config import SYMPTOM_EMBEDS, SYMPTOM_JSON, DIAG_FEATURE_IMPORTANCE
import numpy as np
from tools.embeddings import get_embedding_model
import json
from langchain.tools import tool

from pydantic import BaseModel, Field


class MatchSymptomsArgs(BaseModel):
    symptoms_list: list[str] = Field(...)


class VectorizeArgs(BaseModel):
    idx: list[int] = Field(...)


class AskAdditionalArgs(BaseModel):
    predictions: dict = Field(...)
    known_symptoms: list[str] | None = None
    exclude: list[str] | None = None


hf = get_embedding_model()

symptom_embeddings = np.load(SYMPTOM_EMBEDS)
symptoms = json.load(open(SYMPTOM_JSON))
assert symptom_embeddings.shape[0] == len(symptoms), \
    "Mismatch between embeddings and symptom names!"


def cosine_similarity(vec1, vec2):
    vec2 = np.array(vec2).reshape(-1)
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1, axis=1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


@tool("symptom_matcher", return_direct=False, args_schema=MatchSymptomsArgs)
def match_symptoms(symptoms_list):
    """Matches user-described symptoms to the closest known medical symptoms using embeddings.

    Args:
        symptoms_list (list):
            A list of symptom phrases provided by the user in natural language.
            Example:
                ["stomach hurts", "feeling dizzy"]

    Returns:
        dict: A dictionary containing:
            - "matched_symptoms" (list of str): The standardized medical symptom names
              that best match each user-provided symptom.
            - "indexes" (list of int): The corresponding indexes of matched symptoms
              in the stored symptom embeddings.

            Example:
                {
                    "matched_symptoms": ["stomach_pain", "dizziness"],
                    "indexes": [12, 57]
                }

    Description:
        This tool converts each user symptom into an embedding vector using the
        HuggingFace embedding model, computes cosine similarity with all stored
        symptom embeddings, and selects the most similar known symptom for each input.
        It is typically the first step in the diagnostic reasoning pipeline.
    """
    print(f"[TOOL CALL] symptom_matcher with symptoms : {symptoms_list}")
    results, idxs = [], []
    for symptom in symptoms_list:
        em_sym = hf.embed_query(symptom)
        sim_res = cosine_similarity(symptom_embeddings, em_sym)
        idx = sim_res.argmax()
        idxs.append(int(idx))
        results.append(symptoms[idx])
    out = {"matched_symptoms": results, "indexes": idxs}
    print(f"[TOOL RETURN] : {out}")
    return out


@tool("symptom_vectorizer", return_direct=False, args_schema=VectorizeArgs)
def get_vector(idx):
    """Converts a list of matched symptom indexes into a binary feature vector for prediction.

    Args:
        idx (list):
            A list of integer indexes corresponding to matched symptoms
            (as returned by the `symptom_matcher` tool).
            Example:
                [12, 57, 88]

    Returns:
        dict: A dictionary containing:
            - "vector" (list of int): A binary feature vector (1 for present symptoms, 0 otherwise),
              ready to be passed into the `disease_prediction` model.
            Example:
                {
                    "vector": [[0, 0, 0, 1, 0, ..., 1]]
                }

    Description:
        This tool constructs a binary vector representation of symptoms based on their indexes
        in the master symptom list. Each index in the vector corresponds to a symptom
        in the database. The resulting vector is typically used as input for
        the XGBoost diagnostic model (`disease_prediction` tool).
    """
    print(f"[TOOL CALL] symptom_vectorizer with IDs = {idx}")
    vector = np.zeros(len(symptoms))
    vector[idx] = 1
    out = {"vector": vector.reshape(1, -1).tolist()}
    print(f"[TOOL RETURN] : {out}")

    return out


@tool("ask_additional_question", return_direct=False, args_schema=AskAdditionalArgs)
def ask_additional_symptoms(predictions, known_symptoms=None, exclude=None):
    """Selects the most relevant additional symptoms to ask."""
    print(f"[TOOL CALL] ask_additional_question with prediction:{predictions}")

    with open(DIAG_FEATURE_IMPORTANCE, "r") as f:
        data = json.load(f)

    MIN_PROB = 0.10
    diseases = predictions["diseases"]
    probs = predictions["probabilities"]
    filtered = [(d, p) for d, p in zip(diseases, probs) if p >= MIN_PROB]
    if not filtered:
        return {"symptoms_to_ask": []}

    names = [d for d, p in filtered]

    known_symptoms = {s.replace("_", " ") for s in (known_symptoms or [])}
    exclude = {s.replace("_", " ") for s in (exclude or [])}
    forbidden = known_symptoms.union(exclude)

    candidates = []
    for name in names:
        if name not in data:
            continue
        for symptom_raw, value in data[name].items():
            symptom = symptom_raw.replace("_", " ")
            if symptom not in forbidden:
                candidates.append((name, symptom, value))

    top = sorted(candidates, key=lambda x: x[2], reverse=True)[:5]
    top_symptoms = list(dict.fromkeys([symptom for _, symptom, _ in top]))
    out = {"symptoms_to_ask": top_symptoms}
    print(f"[TOOL RETURN] ask_additional_question : {out}")
    return out
