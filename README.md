# MedicalDiagAI

Hybrid medical assistant prototype for the course  
**ACIT4620 – Computational Intelligence: Theory and Applications (OsloMet)**.

MedicalDiagAI combines:

- a **Large Language Model (LLM)** for natural-language dialogue, symptom extraction and question generation;
- an **XGBoost classifier** trained on structured symptom–disease tables (496 symptom features, 762 diagnosis labels);
- a **retrieval component** (RAG-style) to generate more clinically plausible follow-up questions and high-level treatment / safety advice.

> ⚠️ **Important disclaimer**  
> This project is a **research and teaching prototype only**.  
> It has **not** been clinically validated and **must not** be used for real medical decision-making or triage.

---

## Repository Structure

```text
.
├── app.py                  # Main entrypoint (Gradio web app / chat interface)
├── config.py               # Global configuration (paths, thresholds, etc.)
├── orchestrator.py         # Dialogue loop & control logic
├── prompts.py              # LLM prompts for chains
├── state.py                # Diagnostic state object and helpers
│
├── chains/                 # LangChain-based pipelines ("chains")
│   ├── extract.py          # Free-text symptom extraction
│   ├── match.py            # Mapping extracted phrases to internal symptom vocab
│   ├── next_question.py    # Next-question generation (RAG-based)
│   ├── predict.py          # XGBoost prediction chain
│   ├── router.py           # Intent routing (symptom / question / command)
│   └── vectorize.py        # Build symptom vector from state
│
├── tools/                  # Tools used from within chains
│   ├── embeddings.py       # Symptom embeddings and similarity helpers
│   ├── questions.py        # ask_additional_symptoms tool (RAG next-question)
│   ├── symptom_tool.py     # Symptom matcher & vectorizer wrappers
│   ├── treatment.py        # Simple treatment / safety advice generator
│   └── xgb_tool.py         # XGBoost prediction wrapper
│
├── llm/
│   └── ollama_chat_llama.py  # LLM wrapper (e.g. local Ollama model)
│
├── data/
│   ├── assets/
│   │   ├── symptoms.json                # Symptom vocabulary
│   │   ├── symptom_embeddings.npy       # Embeddings for symptom similarity
│   │   ├── label_encoder.joblib         # Label encoder for diagnoses
│   │   ├── xgb_classifier.joblib        # Trained XGBoost model (large)
│   │   ├── xgb_model.json               # Full XGBoost model dump (large)
│   │   ├── diags_feature_importance*.json
│   │   └── *_old.*                      # Older versions of the above
│   └── __init__.py
│
├── data_checking.ipynb     # Notebook for dataset sanity checks
└── vector_store.ipynb      # Notebook for retrieval / vector store experiments
```

## Large Model Files

GitHub does not allow files larger than 100 MB.  
The trained XGBoost assets are therefore stored externally.

📂 **Download the assets here:**

https://drive.google.com/drive/folders/1i2hbJEg0yoqqAFyMvVw5urrM6qmKInyo?usp=sharing

Place the downloaded files into:

```text
data/assets/
```

## Installation

1. Clone the repository

    git clone https://github.com/svgrekk/MedicalDiagAI.git
    cd MedicalDiagAI

2. Create and activate a virtual environment (recommended)
    ```text
    python -m venv .venv
     ```
    ### Linux / macOS
   ```text
    source .venv/bin/activate
     ```
    ### Windows (PowerShell / cmd)
    ```text
    .venv\Scripts\activate
    ```

3. Install dependencies
    ```text
    pip install -r requirements.txt
    ```
4. Install Ollama and pull the model

    ### Download and install Ollama from https://ollama.com/download
    ### Then pull the model used by this project:
    ```text
    ollama pull llama3.1
    ```
5. Run the app
    ```text
    python app.py
    ```
