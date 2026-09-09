import torch
import torch.nn as nn
from torch.nn import functional as F
import math

# =====================================================================
# DAY 2 (UPGRADED): MODERN LLaMA-STYLE ARCHITECTURE 
# Components:
# 1. Causal Multi-Head Self-Attention
# 2. RMSNorm (Root Mean Square Normalization - LLaMA 3 standard)
# 3. SwiGLU Feed-Forward Network (Gated Linear Unit - LLaMA 3 standard)
# 4. Pre-LayerNorm & Residual Connections
# =====================================================================

vocab_size = 70       # 70 unique characters from Medical Q&A
n_embd = 64           # Embedding dimension
block_size = 32       # Context window
n_head = 4            # 4 attention heads (64 / 4 = 16 dimensions per head)
n_layer = 4           # 4 Transformer blocks
dropout = 0.1
device = 'cuda' if torch.cuda.is_available() else 'cpu'


# =====================================================================
# 1. RMSNorm (Root Mean Square Normalization)
# Used in: LLaMA 3, Gemma, Mistral, DeepSeek
# Why better than standard LayerNorm?
# - Skips mean-centering (saving 20% compute time)
# - Only normalizes by the root-mean-square of inputs.
# =====================================================================
class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization (LLaMA-3 standard)."""

    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


# =====================================================================
# 2. CAUSAL MULTI-HEAD ATTENTION
# =====================================================================
class Head(nn.Module):
    """One single head of self-attention."""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)    # (B, T, head_size)
        q = self.query(x)  # (B, T, head_size)

        # Scaled dot-product attention
        wei = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        v = self.value(x)
        out = wei @ v
        return out


class MultiHeadAttention(nn.Module):
    """Multiple heads of self-attention in parallel."""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out


# =====================================================================
# 3. SwiGLU FEED-FORWARD NETWORK (Gated Linear Unit)
# Used in: LLaMA 1/2/3, PaLM, Gemma
# Why better than standard ReLU/GELU?
# - Uses a multiplicative "gate" that lets the network dynamically decide
#   how much information should pass through each neuron.
# Formula: SwiGLU(x) = (SiLU(x @ W1) * (x @ W3)) @ W2
# =====================================================================
class SwiGLUFeedForward(nn.Module):
    """SwiGLU Gated Feed-Forward Network (Modern LLaMA-style)."""

    def __init__(self, n_embd):
        super().__init__()
        hidden_dim = int(2 * (4 * n_embd) / 3)  # Standard LLaMA dimension scaling
        self.w1 = nn.Linear(n_embd, hidden_dim, bias=False)  # Gate projection
        self.w2 = nn.Linear(hidden_dim, n_embd, bias=False)  # Down projection
        self.w3 = nn.Linear(n_embd, hidden_dim, bias=False)  # Up projection
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # SiLU(x @ W1) * (x @ W3), then projected through W2
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        out = self.w2(gate * up)
        return self.dropout(out)


# =====================================================================
# 4. MODERN TRANSFORMER BLOCK (RMSNorm + Attention + SwiGLU)
# =====================================================================
class Block(nn.Module):
    """Transformer block with RMSNorm and SwiGLU."""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = SwiGLUFeedForward(n_embd)
        self.norm1 = RMSNorm(n_embd)
        self.norm2 = RMSNorm(n_embd)

    def forward(self, x):
        # Pre-Norm residual connections
        x = x + self.sa(self.norm1(x))
        x = x + self.ffwd(self.norm2(x))
        return x


# =====================================================================
# 5. THE COMPLETE MEDICAL-MINI-GPT MODEL
# =====================================================================
class MedMiniGPT(nn.Module):

    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.norm_f = RMSNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)                                # (B, T, n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, n_embd)
        x = tok_emb + pos_emb                                                   # (B, T, n_embd)

        x = self.blocks(x)                                                      # (B, T, n_embd)
        x = self.norm_f(x)                                                      # (B, T, n_embd)
        logits = self.lm_head(x)                                                # (B, T, vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens, temperature=1.0):
        """Autoregressive text generation with temperature control."""
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-5)  # Scale by temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# =====================================================================
# VERIFICATION RUN
# =====================================================================
if __name__ == '__main__':
    from day1_data import get_batch, decode

    print("=" * 65)
    print("DAY 2: MODERN LLaMA-STYLE ARCHITECTURE (MedMiniGPT)")
    print("=" * 65)

    model = MedMiniGPT().to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Device: {device}")
    print(f"Total Parameters: {total_params:,}")
    print("Architecture Upgrades:")
    print("  [+] Normalization: RMSNorm (Root Mean Square Normalization)")
    print("  [+] Non-Linearity: SwiGLU Gated Feed-Forward (LLaMA-3 standard)")
    print("  [+] Causal Multi-Head Attention: 4 heads, block_size=32")

    # Pass a medical batch
    xb, yb = get_batch('train')
    xb, yb = xb.to(device), yb.to(device)
    logits, loss = model(xb, yb)

    print(f"\nForward Pass Successful!")
    print(f"Logits shape (B*T, vocab_size): {logits.shape}")
    print(f"Initial untrained loss: {loss.item():.4f}")
    expected = -math.log(1.0 / vocab_size)
    print(f"Theoretical expected random loss (-ln(1/70)): {expected:.4f}")

    print("\n" + "=" * 65)
    print("SAMPLE GENERATION BEFORE TRAINING (RANDOM GIBBERISH):")
    print("-" * 65)
    start = torch.zeros((1, 1), dtype=torch.long, device=device)
    print(decode(model.generate(start, max_new_tokens=100)[0].tolist()))
    print("=" * 65)
