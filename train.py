import torch
import torch.nn as nn
from torch.nn import functional as F
import json
import time
import matplotlib.pyplot as plt
from model import MedMiniGPT, vocab_size, block_size, device
from dataset import train_data, val_data, encode, decode, chars

print("=" * 65)
print("TRAINING ENHANCED MedMiniGPT (Context Window = 128)")
print("=" * 65)
print(f"Device: {device}")

batch_size = 16
max_iters = 2500      # 2500 steps for deep convergence
eval_interval = 250
eval_iters = 40
learning_rate = 1e-3

def get_train_val_batch(split):
    data_source = train_data if split == 'train' else val_data
    ix = torch.randint(len(data_source) - block_size, (batch_size,))
    x = torch.stack([data_source[i : i + block_size] for i in ix])
    y = torch.stack([data_source[i + 1 : i + block_size + 1] for i in ix])
    return x.to(device), y.to(device)

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_train_val_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

model = MedMiniGPT().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-2)

print(f"Total model parameters: {sum(p.numel() for p in model.parameters()):,}")
print("Starting training loop...\n")

train_loss_history = []
val_loss_history = []
step_history = []

start_time = time.time()

for iter in range(max_iters + 1):
    if iter % eval_interval == 0:
        losses = estimate_loss(model)
        train_loss_history.append(losses['train'].item())
        val_loss_history.append(losses['val'].item())
        step_history.append(iter)
        elapsed = time.time() - start_time
        print(f"Step {iter:4d}/{max_iters} | Train Loss: {losses['train']:.4f} | Val Loss: {losses['val']:.4f} | Elapsed: {elapsed:.1f}s")

    xb, yb = get_train_val_batch('train')
    logits, loss = model(xb, yb)

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    optimizer.step()

total_duration = time.time() - start_time
print("\n" + "=" * 65)
print(f"TRAINING COMPLETE in {total_duration:.1f} seconds!")
print(f"Final Validation Loss: {val_loss_history[-1]:.4f}")
print("=" * 65)

# Save checkpoint
torch.save(model.state_dict(), 'med_model.pt')
metadata = {
    'vocab': chars,
    'vocab_size': vocab_size,
    'block_size': block_size,
    'final_val_loss': float(val_loss_history[-1])
}
with open('metadata.json', 'w', encoding='utf-8') as f:
    json.dump(metadata, f, indent=2)

print("[+] Model checkpoint saved as: 'med_model.pt'")
print("[+] Model metadata saved as:   'metadata.json'")

# Save plot
plt.figure(figsize=(9, 5))
plt.plot(step_history, train_loss_history, label='Training Loss', color='#2563eb', linewidth=2)
plt.plot(step_history, val_loss_history, label='Validation Loss', color='#dc2626', linewidth=2, linestyle='--')
plt.title('MedMiniGPT (Context 128): Training & Validation Loss', fontsize=14, fontweight='bold')
plt.xlabel('Steps', fontsize=12)
plt.ylabel('Cross-Entropy Loss', fontsize=12)
plt.grid(True, linestyle=':', alpha=0.6)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('loss_curve.png', dpi=150)
print("[+] Loss curve saved as: 'loss_curve.png'")
