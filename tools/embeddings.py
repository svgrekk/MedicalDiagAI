# tools/embeddings.py
from langchain_huggingface import HuggingFaceEmbeddings
from config import DEVICE


def get_embedding_model():
    """
    Creates and returns a Hugging Face embedding model configured for the project.
    Uses DEVICE defined in config.py ('cuda', 'mps', or 'cpu').
    """
    model_name = "sentence-transformers/all-mpnet-base-v2"
    model_kwargs = {"device": DEVICE}
    encode_kwargs = {"normalize_embeddings": False}

    print(f"Embedding model loaded on {DEVICE}")

    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs=model_kwargs,
        encode_kwargs=encode_kwargs,
    )

