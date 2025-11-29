# prompts.py
from typing import Sequence

EXTRACTION_PROMPT = """You are a medical triage assistant.

TASK: Extract concise symptom phrases from the patient's complaint.

GUIDELINES:
- Use short, general medical phrases (e.g., "fever", "dizziness", "chest pain", "nausea").
- Lowercase only. Remove duplicates.
- No diagnoses. No explanations. Do not invent symptoms.

Return ONLY valid JSON with a single key "symptoms": a list of strings.

Complaint:
{complaint}

{format_instructions}
"""

REPORT_PROMPT = """You are a careful medical triage assistant. Write in clear, plain English for a user in Norway.

Inputs:
- Extracted symptom phrases: {user_symptoms}
- Standardized symptoms considered: {matched_symptoms}
- Top-5 predicted conditions with probabilities: {diseases_probs}

Write a concise message to the patient:
1) Briefly restate the symptoms considered (1 sentence).
2) State the most likely condition with its probability and mention 1–2 alternatives (1–2 sentences).
3) Give safe, general next-step advice and red-flag guidance (1–2 sentences). Do NOT make a diagnosis.
4) If uncertainty is meaningful, ask up to 3 targeted yes/no follow-up questions, each on a new line starting with "- ".

Return plain English prose only (no JSON).
"""

UPDATE_REPORT_PROMPT = """Continue the triage conversation in clear English.

Context:
- Current standardized symptoms: {known_symptoms}
- Newly confirmed symptoms: {new_positive}
- Updated top-5 predicted conditions with probabilities: {diseases_probs}

Write a short update:
1) Acknowledge the new information (1 sentence).
2) State the current most likely condition with its probability and briefly mention one alternative (1–2 sentences).
3) Update advice or red flags if needed (1–2 sentences). Do NOT make a diagnosis.
4) If the top probability remains below {threshold:.2f}, ask up to 2 targeted yes/no questions, each on a new line starting with "- ". Otherwise, conclude politely.

Return plain English prose only.
"""

FOLLOWUP_QUESTIONS_PROMPT = """You are a medical assistant. Ask one clear yes/no question for each symptom.

{format_instructions}

Symptoms (use EXACT names, do NOT skip any):
{symptoms_list}

Return JSON with ALL symptoms.
"""

YESNO_NORMALIZER_PROMPT = """You are standardizing patient replies to yes/no triage questions.

Return ONLY valid JSON with key "normalized": a list matching the number/order of questions.
Each value must be:
- true  (clearly yes)
- false (clearly no)
- null  (unknown/unclear)

Questions:
{questions}

Answers:
{answers}

Return JSON like:
{{"normalized": [true, false, null]}}
"""
INTENT_PROMPT = """
You are a medical diagnostic assistant. Your ONLY job is to classify the user's message into ONE of three categories:

1. "symptom" — if the user:
   - describes symptoms
   - answers "yes", "no", "sometimes"
   - says "I have...", "my head hurts", "pain in chest", "feeling tired"
   - even if poorly written

2. "question" — if the user:
   - asks "what is...", "why", "how", "explain", "tell me"
   - uses "?"
   - wants explanation or clarification

3. "command" — if the user:
   - says "stop", "exit", "restart", "show", "summary", "save"

CRITICAL RULES:
- If the message contains ANY description of pain, discomfort, or health issue → "symptom"
- If it looks like a complaint → "symptom"
- If unsure → choose "symptom" (better to ask than miss a disease)

EXAMPLES (correct answers):
"i have a pain in chest and in head" → symptom
"yes" → symptom
"no, not really" → symptom
"my stomach hurts" → symptom
"headache and fever" → symptom
"what is GERD?" → question
"show me what you know" → question
"stop" → command
"restart" → command

User message: {message}

Return ONLY one word: symptom / question / command
"""


QA_MINI_PROMPT = """
Answer the user's question briefly and clearly in simple English, using the context if relevant.
Do NOT provide a medical diagnosis. 1–2 sentences max.

Context: {context}
Question: {question}
"""
RAG_PROMPT = """
You are an experienced clinical assistant using reasoning supported by PubMed evidence.

Known symptoms (already confirmed): {known}
Denied symptoms (explicitly ruled out): {denied}
Available standard symptoms (from dataset): {available}

PubMed context:
{context}

Your task (3 stages):

1. **Diagnosis reasoning:**  
   Based on the known symptoms and the PubMed context, infer the single most likely disease
   that could explain the observed symptoms.
   - You may name a new disease not present in the dataset if PubMed evidence supports it.

2. **Symptom discovery:**  
   Identify exactly 3 additional symptoms that would most strongly confirm or rule out your inferred disease.
   - Use PubMed context for guidance.
   - Prefer symptoms that are clinically specific, discriminative, or typical of the suspected disease.

3. **Symptom alignment:**  
   For each of the 3 discovered symptoms:
   - If a similar or identical symptom exists in the AVAILABLE list, use that dataset name.
   - Otherwise, include it as a *new* symptom (keep it medically realistic).

Return ONLY JSON in the following format:
{format_instructions}

Each key must be the final symptom (from dataset if possible),
and each value must be a natural yes/no question a clinician would ask a patient about it.

Example output:
{{
  "mapping": {{
    "sore throat": "Does the patient have a sore throat?",
    "fatigue": "Has the patient been feeling unusually tired?",
    "swollen lymph nodes": "Are the glands in the neck swollen?"
  }}
}}
"""


def format_diseases_probs(diseases: Sequence[str], probabilities: Sequence[float]) -> str:
    """Format predictions as: 'DiseaseA (0.72), DiseaseB (0.19), ...'"""
    return ", ".join(f"{d} ({p:.2f})" for d, p in zip(diseases, probabilities))