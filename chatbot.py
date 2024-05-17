import gc
import os

import gradio as gr
import torch.cuda
from  transformers import AutoModelForCausalLM, AutoTokenizer


MODELS_DIR = "D:/works/big/LLaMA-Factory/models"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
current_model = None
current_tokenizer = None
def load_model(model_name):
    global current_model, current_tokenizer
    if current_model is not None:
        del current_model
        del current_tokenizer
        torch.cuda.empty_cache()
        gc.collect()
        current_model = None
        current_tokenizer = None

    model_path = os.path.join(MODELS_DIR, model_name)
    current_model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
    current_tokenizer = AutoTokenizer.from_pretrained(model_path)
    return current_model, current_tokenizer

def chat(input_text, history, model_name):
    model, tokenizer = load_model(model_name)
    history = history or []
    history.append(f"User: {input_text}")
    inputs = tokenizer("\n".join(history) + "\nAI:", return_tensors="pt").to(device)
    outputs = model.generate(inputs.input_ids, max_length=500, pad_token_id=tokenizer.eos_token_id)
    response = tokenizer.decode(outputs[:, inputs.input_ids.shape[-1]:][0], skip_special_tokens=True)
    history.append(f"AI: {response}")
    chat_history = [(history[i], history[i+1]) for i in range(0, len(history)-1, 2)]
    return chat_history, history
def get_model_list(model_dir):
    return [name for name in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, name))]
def change_model(model_name):
    load_model(model_name)
    return f"Model changed to {model_name}"


model_list = get_model_list(MODELS_DIR)
with gr.Blocks() as demo:
    gr.Markdown("# Dynamic Model Swithcing Chatbot")
    model_name = gr.Dropdown(
        choices = model_list,
        value=model_list[0],
        label="Select Model"
    )
    chat_input = gr.Textbox(label="Your Input")
    chat_output = gr.Chatbot(label="Chat History")
    state = gr.State([])

    model_name.change(change_model, inputs=model_name, outputs=None)
    chat_input.submit(chat, inputs=[chat_input, state, model_name], outputs=[chat_output, state])


demo.launch()

