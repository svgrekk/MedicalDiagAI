# orchestrator.py
from __future__ import annotations
from typing import Dict, Any, List

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser

from chains.extract import build_extract_chain
from chains.match import build_match_chain
from chains.vectorize import build_vectorize_chain
from chains.next_question import build_next_question_chain
from chains.predict import build_predict_chain
from chains.router import build_router
from state import DiagnosticState
from prompts import REPORT_PROMPT, UPDATE_REPORT_PROMPT, YESNO_NORMALIZER_PROMPT, QA_MINI_PROMPT, INTENT_PROMPT
from tools.treatment import get_treatment_recommendation


class Orchestrator:
    """
    Orchestrates the iterative medical diagnosis flow in a loop:
    - User complaint → extraction → matching (standard symptoms & indexes) → vectorization → prediction
    - If low confidence: generate questions for additional symptoms → ask user one by one
    - Process user answers (y/n) → update symptoms (add positives, exclude negatives) → repeat cycle
    Maintains state across iterations. Negative answers are preserved to avoid re-asking.
    No recommendations during diagnosis; only final report when confident.
    """

    def __init__(self, llm: BaseChatModel, confidence_threshold: float = 0.9):
        self.llm = llm
        self.confidence_threshold = confidence_threshold

        # Chains for the cycle
        self.extract_chain = build_extract_chain(llm)
        self.match_chain = build_match_chain()
        self.vectorize_chain = build_vectorize_chain()
        self.predict_chain = build_predict_chain()
        self.next_question_chain = build_next_question_chain(llm, limit=1)  # One question at a time


        # Intent router
        self.router = build_router()

        # QA for non-symptom questions
        self.qa_chain = PromptTemplate.from_template(QA_MINI_PROMPT) | llm

        # State
        self.reset_state()

    def reset_state(self):
        self.state = DiagnosticState()
        self.state.question_count = 0  # New counter for questions asked
        self.state.consecutive_no = 0  # New: counter for consecutive "no" answers

    def process_message(self, user_message: str) -> str:
        """
        Processes each user message in the iterative cycle.
        Handles intents, but focuses on symptom cycle: extract/match/vectorize/predict → questions if needed → update on answers.
        """
        print(f"> {user_message}")
        lower_msg = user_message.strip().lower()

        # Handle commands first
        if lower_msg in ["exit", "quit", "stop"]:
            return "Goodbye! Take care."
        elif lower_msg in ["restart", "reset"]:
            self.reset_state()
            return "Restarted. Describe your symptoms."

        # Special handling for answers when pending question
        if self.state.pending_question:
            # Treat as answer
            answer = lower_msg
            ans = None
            if answer in ['y', 'yes', 'true', 'da', 'yeah']:
                ans = True
            elif answer in ['n', 'no', 'false', 'net', 'nah']:
                ans = False

            if ans is not None:
                symptoms_to_ask = self.state.pending_question.get("symptoms_to_ask", [])
                new_positive = []
                for sym in symptoms_to_ask:  # Only one, but loop for consistency
                    if ans is True:
                        new_positive.append(sym)
                        self.state.consecutive_no = 0  # Reset on "yes"
                    elif ans is False:
                        self.state.known_negatives.append(sym)
                        self.state.consecutive_no += 1  # Increment on "no"

                self.state.pending_question = {}  # Clear for next cycle
                print(f"[ORCH] Processed answers: positives={new_positive}, negatives={self.state.known_negatives}")

                # Proceed with update
                if new_positive:
                    matched = self.match_chain.invoke({
                        "symptoms_list": new_positive,
                        "known_symptoms": self.state.known_symptoms,
                        "known_indexes": self.state.known_indexes,
                    })
                    self.state.known_symptoms = matched["known_symptoms"]
                    self.state.known_indexes = matched["known_indexes"]
                    print(f"[STATE] Updated known_symptoms = {self.state.known_symptoms}")
                    print(f"[STATE] known_indexes = {self.state.known_indexes}")

                # Check for 3 consecutive "no"
                if self.state.consecutive_no >= 3:
                    self.state.finished = True

                # Then predict etc.
            else:
                # Not a yes/no, check if it's a question intent for explanation
                intent = self.router.invoke({"message": user_message})
                print(f"[ORCH] Intent: {intent}")
                if intent == "question":
                    context = f"Current symptoms: {', '.join(self.state.known_symptoms)}. Pending question: {self.state.pending_question.get('questions', [''])[0]}"
                    qa_response = self.qa_chain.invoke({"question": user_message, "context": context}).content
                    return qa_response + "\n\nPlease answer with yes or no to the previous question: " + self.state.pending_question.get("questions", [""])[0]
                else:
                    # Remind
                    return "Please answer with yes or no to the previous question."

        else:
            # Normal intent detection
            intent = self.router.invoke({"message": user_message})
            print(f"[ORCH] Intent: {intent}")

            if intent == "command":
                return "Unknown command. Please describe symptoms or ask a question."

            if intent == "question":
                context = self.state.pending_question.get("questions", [""])[0] if self.state.pending_question else ""
                return self.qa_chain.invoke({"question": user_message, "context": context}).content

            # Symptom intent: extract new symptoms
            extracted = self.extract_chain.invoke({"complaint": user_message})
            symptoms_list = extracted["symptoms"]
            print(f"[DEBUG] Extracted symptoms: {symptoms_list}")

            if not symptoms_list:
                if self.state.finished:
                    return f"Diagnosis complete. No new symptoms detected. Final diagnosis: {self.state.predictions['diseases'][0]} (confidence {self.state.top_probability * 100:.0f}%). Consult a doctor."
                else:
                    return "No new symptoms. Please provide more details."

            matched = self.match_chain.invoke({
                "symptoms_list": symptoms_list,
                "known_symptoms": self.state.known_symptoms,
                "known_indexes": self.state.known_indexes,
            })
            new_positive = matched["matched_symptoms"]

            if new_positive:
                matched = self.match_chain.invoke({
                    "symptoms_list": new_positive,
                    "known_symptoms": self.state.known_symptoms,
                    "known_indexes": self.state.known_indexes,
                })
                self.state.known_symptoms = matched["known_symptoms"]
                self.state.known_indexes = matched["known_indexes"]
                print(f"[STATE] Updated known_symptoms = {self.state.known_symptoms}")
                print(f"[STATE] known_indexes = {self.state.known_indexes}")

        # 6. Vectorize → Predict (after answer or new symptoms)
        response = ""
        if self.state.known_indexes:
            print(f"[ORCH] Building vector from indexes: {self.state.known_indexes}")
            vector = self.vectorize_chain.invoke({"known_indexes": self.state.known_indexes})["vector"]

            prediction = self.predict_chain.invoke({"vector": vector})
            self.state.predictions = prediction
            print(f"[TOOL RETURN] disease_prediction : {prediction}")

            top_diseases = prediction["diseases"][:3]
            top_probs = prediction["probabilities"][:3]
            self.state.top_probability = top_probs[0]

            diagnoses_str = "\n".join(f"- {disease} ({prob * 100:.0f}%)" for disease, prob in zip(top_diseases, top_probs))

            if self.state.finished and self.state.top_probability < self.confidence_threshold:
                response = f"Symptoms: {', '.join(self.state.known_symptoms)}. Low confidence in diagnosis, providing symptom-based recommendations."
                disease_for_rec = ""  # No disease if low confidence
            else:
                response = f"Recognized symptoms: {', '.join(self.state.known_symptoms)}.\nPossible diagnoses:\n{diagnoses_str}"
                disease_for_rec = top_diseases[0]

            # 7. If low confidence: Get additional symptoms → Generate questions (one at a time) → Set pending for next iteration
            if self.state.top_probability < self.confidence_threshold and not self.state.finished:
                questions_data = self.next_question_chain.invoke({
                    "predictions": prediction,
                    "known_symptoms": self.state.known_symptoms,
                    "exclude": self.state.known_symptoms + self.state.known_negatives,
                })

                symptoms_to_ask = questions_data["symptoms_to_ask"]
                questions = questions_data["questions"]

                if symptoms_to_ask:
                    self.state.pending_question = {
                        "symptoms_to_ask": symptoms_to_ask,
                        "questions": questions
                    }
                    response += "\nI need more info:\n" + "\n".join(f"• {q} (y/n)" for q in questions)
                    print(f"[ORCH] Pending questions set for next cycle")
                    self.state.question_count += 1  # Increment question count
            else:
                # Finished: Simple final message
                if not (self.state.finished and self.state.top_probability < self.confidence_threshold):
                    response = f"Final possible diagnoses:\n{diagnoses_str}\nSymptoms: {', '.join(self.state.known_symptoms)}. Consult a doctor for confirmation."
                self.state.finished = True

            # Check if diagnosis is finished and provide treatment recommendation
            if self.state.finished:
                # Call treatment recommendation tool
                treatment_rec = get_treatment_recommendation.invoke({"disease": disease_for_rec, "symptoms": self.state.known_symptoms})
                response += f"\n\nTreatment Recommendations (based on general knowledge; consult a professional):\n{treatment_rec}"

            # Update history
            self.state.chat_history.append({"user": user_message, "assistant": response})

            return response

        # Fallback
        return "Please provide symptoms to start diagnosis."


# Example usage (for testing)
if __name__ == "__main__":
    from llm.ollama_chat_llama import get_llm

    llm = get_llm(model_name="llama3.1")
    orch = Orchestrator(llm)

    print("Welcome to the medical assistant. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        resp = orch.process_message(user_input)
        print(f"[Assistant]: {resp}")