import gradio as gr
import torch
import json
from day2_model import MedMiniGPT, device

# Load vocabulary and metadata
with open('metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

chars = meta['vocab']
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char_to_id.get(c, 0) for c in s]

def decode(l):
    return ''.join([id_to_char.get(i, '') for i in l])

# Load PyTorch model
model = MedMiniGPT().to(device)
model.load_state_dict(torch.load('med_model.pt', map_location=device))
model.eval()

def answer_query(question, temperature=0.35, max_tokens=180):
    if not question.strip():
        return "Please enter a valid medical question."
    
    prompt = f"Question: {question.strip()}\nAnswer:"
    encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(encoded, max_new_tokens=int(max_tokens), temperature=float(temperature))
    
    full_text = decode(generated[0].tolist())
    answer_raw = full_text[len(prompt):].strip()
    
    # Truncate stop condition
    if "\nQuestion:" in answer_raw:
        answer_raw = answer_raw.split("\nQuestion:")[0].strip()
    if "Question:" in answer_raw:
        answer_raw = answer_raw.split("Question:")[0].strip()
        
    return answer_raw

demo = gr.Interface(
    fn=answer_query,
    inputs=[
        gr.Textbox(
            label="Medical Question",
            placeholder="Type a clinical question or select an example below...",
            lines=3
        ),
        gr.Slider(
            minimum=0.1,
            maximum=1.0,
            value=0.35,
            step=0.05,
            label="Sampling Temperature",
            info="Lower values yield more deterministic clinical facts."
        ),
        gr.Slider(
            minimum=50,
            maximum=300,
            value=180,
            step=10,
            label="Maximum Output Length (Tokens)"
        )
    ],
    outputs=gr.Textbox(label="Generated Clinical Answer", lines=5),
    title="MedMini-LLM: Domain-Specific Transformer from Scratch",
    description="A 468K-parameter Decoder-Only Transformer built and trained entirely from scratch in PyTorch using RMSNorm and SwiGLU activations (LLaMA-3 architecture).",
    examples=[
        ["What is Hypertension and how is it clinically defined?", 0.35, 180],
        ["What are the hallmark symptoms and diagnostic criteria for Type 2 Diabetes?", 0.35, 180],
        ["What is Asthma and how is an acute bronchospasm managed?", 0.35, 180],
        ["How do you recognize an acute ischemic stroke using the FAST protocol?", 0.30, 180]
    ]
)

if __name__ == "__main__":
    demo.launch()
