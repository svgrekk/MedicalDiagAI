# app.py
import gradio as gr
from orchestrator import Orchestrator
from llm.ollama_chat_llama import get_llm

# Initialize the LLM and Orchestrator
llm = get_llm(model_name="llama3.1")
orch = Orchestrator(llm)


def chat_response(message, history):
    """
    Process user message through Orchestrator and return response.
    History is maintained in Orchestrator's state.
    """
    if message.lower() == "restart":
        orch.reset_state()
        return "", []  # Clear chat

    response = orch.process_message(message)
    updated_history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": response}]
    return "", updated_history


# Create Gradio Chat Interface
with gr.Blocks(title="Medical Assistant") as demo:
    gr.Markdown("# Medical Assistant")
    gr.Markdown(
        "Describe your symptoms or ask questions. Type 'restart' to start over, 'exit' to quit (but in chat, it just responds).")

    chatbot = gr.Chatbot(height=500, type="messages")
    msg = gr.Textbox(label="Your Message", placeholder="Type here...")
    clear = gr.Button("Clear Chat")

    # Submit on enter
    msg.submit(chat_response, [msg, chatbot], [msg, chatbot])

    # Clear button resets chat UI and state
    clear.click(lambda: (orch.reset_state(), ""), None, [chatbot, msg], queue=False)

demo.launch(share=True)