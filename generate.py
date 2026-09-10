import torch
import torch.nn.functional as F
import json
import argparse
from model import MedMiniGPT, device, block_size

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

# Load the trained model
model = MedMiniGPT().to(device)
model.load_state_dict(torch.load('med_model.pt', map_location=device))
model.eval()

def answer_medical_question(question, max_tokens=200, temperature=0.5):
    prompt = f"Question: {question.strip()}\nAnswer:"
    encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(encoded, max_new_tokens=max_tokens, temperature=temperature)
    
    result = decode(generated[0].tolist())
    return result

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="MedMini-LLM Interactive Inference")
    parser.add_argument('--question', type=str, default="What is Hypertension and how is it clinically defined?", help="Medical question to ask")
    parser.add_argument('--tokens', type=int, default=200, help="Max tokens to generate")
    parser.add_argument('--temperature', type=float, default=0.5, help="Sampling temperature (lower = more deterministic)")
    args = parser.parse_args()

    print("=" * 65)
    print("MEDMINI-LLM: CLINICAL INFERENCE")
    print("=" * 65)
    output = answer_medical_question(args.question, max_tokens=args.tokens, temperature=args.temperature)
    print(output)
    print("=" * 65)
