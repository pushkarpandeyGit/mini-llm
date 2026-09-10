import torch
import json
import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model import MedMiniGPT, device, block_size

app = FastAPI(title="MedMini-LLM API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

with open('metadata.json', 'r', encoding='utf-8') as f:
    meta = json.load(f)

chars = meta['vocab']
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(s):
    return [char_to_id.get(c, 0) for c in s]

def decode(l):
    return ''.join([id_to_char.get(i, '') for i in l])

model = MedMiniGPT().to(device)
model.load_state_dict(torch.load('med_model.pt', map_location=device))
model.eval()

class GenerateRequest(BaseModel):
    question: str
    temperature: float = 0.35
    max_tokens: int = 180

class GenerateResponse(BaseModel):
    question: str
    response: str
    full_text: str

@app.get("/api/info")
def get_info():
    return {
        "model_name": "MedMini-LLM",
        "parameters": sum(p.numel() for p in model.parameters()),
        "context_window": block_size,
        "normalizer": "RMSNorm",
        "feed_forward": "SwiGLU",
        "final_loss": meta.get('final_val_loss', 0.0812),
        "device": device
    }

@app.get("/loss_curve.png")
def get_loss_curve():
    if os.path.exists("loss_curve.png"):
        return FileResponse("loss_curve.png")
    raise HTTPException(status_code=404, detail="Loss curve not found")

@app.post("/api/generate", response_model=GenerateResponse)
def generate_text(req: GenerateRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    
    prompt = f"Question: {req.question.strip()}\nAnswer:"
    encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
    
    with torch.no_grad():
        generated = model.generate(encoded, max_new_tokens=req.max_tokens, temperature=req.temperature)
    
    full_text = decode(generated[0].tolist())
    answer_raw = full_text[len(prompt):].strip()
    
    # Clean stop condition: cut off if model starts generating the next Question
    if "\nQuestion:" in answer_raw:
        answer_raw = answer_raw.split("\nQuestion:")[0].strip()
    if "Question:" in answer_raw:
        answer_raw = answer_raw.split("Question:")[0].strip()
        
    return GenerateResponse(
        question=req.question,
        response=answer_raw,
        full_text=full_text
    )

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
