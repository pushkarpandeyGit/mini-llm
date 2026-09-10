import torch
import tiktoken

# =====================================================================
# DATASET AND TOKENIZATION PIPELINE
# =====================================================================

# 1. Load the Medical Q&A text
with open('medical_data.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("=" * 65)
print(f"Medical Q&A Dataset loaded: {len(text):,} characters")
print("First 250 characters preview:")
print("-" * 65)
print(text[:250])
print("-" * 65)

# 2. Build the Vocabulary (Character-Level for fast local training)
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"\nTotal unique characters (Vocab Size): {vocab_size}")
print(f"Vocabulary: {''.join(chars)}")

char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(string):
    return [char_to_id[c] for c in string]

def decode(indices):
    return ''.join([id_to_char[i] for i in indices])

# 3. Modern Subword Tokenization Demo (tiktoken / GPT-4 / LLaMA style)
enc_bpe = tiktoken.get_encoding("gpt2")
sample_medical = "Question: What is Hypertension and how is it clinically defined?"

bpe_tokens = enc_bpe.encode(sample_medical)
char_tokens = encode(sample_medical)

print("\n" + "=" * 65)
print("TOKENIZER COMPARISON (FOR YOUR RESUME):")
print(f"Original Text: '{sample_medical}'")
print(f"- Character Tokenizer: {len(char_tokens)} tokens (1 number per character)")
print(f"- Modern BPE Tokenizer (tiktoken): {len(bpe_tokens)} tokens (1 number per subword/word!)")
print(f"  BPE tokens: {bpe_tokens[:8]}...")
print("=" * 65)

# 4. Convert Full Medical Dataset to PyTorch Tensor (90% Train / 10% Val)
full_data_tensor = torch.tensor(encode(text), dtype=torch.long)
split_idx = int(0.9 * len(full_data_tensor))
train_data = full_data_tensor[:split_idx]
val_data = full_data_tensor[split_idx:]

print(f"\nTraining set size:   {len(train_data):,} tokens (90%)")
print(f"Validation set size: {len(val_data):,} tokens (10%)")

# 5. Batch Generator
torch.manual_seed(1337)
batch_size = 4
block_size = 128  # 32 characters of medical context

def get_batch(split='train'):
    dataset = train_data if split == 'train' else val_data
    random_starts = torch.randint(len(dataset) - block_size, (batch_size,))
    x = torch.stack([dataset[i : i + block_size] for i in random_starts])
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in random_starts])
    return x, y

xb, yb = get_batch('train')

print("\n" + "=" * 65)
print("SAMPLE MEDICAL TRAINING BATCH (x and y):")
print("-" * 65)
print(f"Input Context x[0]:  '{decode(xb[0].tolist())}'")
print(f"Target Label  y[0]:  '{decode(yb[0].tolist())}'")
print("=" * 65)
print("SUCCESS: DATASET AND TOKENIZATION PIPELINE
