import torch

# =====================================================================
# STEP 1: LOAD RAW DATA
# In natural language processing, everything starts with raw text.
# We are using Shakespeare's plays (about 1.1 million characters).
# =====================================================================
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("=" * 60)
print(f"Dataset loaded: {len(text):,} total characters")
print("First 150 characters preview:")
print("-" * 60)
print(text[:150])
print("-" * 60)


# =====================================================================
# STEP 2: BUILD THE VOCABULARY (THE ALPHABET OF OUR LLM)
# Neural networks don't understand letters; they only understand numbers.
# We find all unique characters in the text and assign each a number (0, 1, 2...).
# =====================================================================
chars = sorted(list(set(text)))
vocab_size = len(chars)

print(f"\nTotal unique characters (Vocab Size): {vocab_size}")
print(f"Vocabulary list:\n{''.join(chars)}")


# =====================================================================
# STEP 3: BUILD THE TOKENIZER (ENCODER & DECODER)
# - Encoder: takes text like 'hello' and turns it into [46, 43, 50, 50, 53]
# - Decoder: takes [46, 43, 50, 50, 53] and turns it back into 'hello'
# =====================================================================
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}

def encode(string):
    """Converts a string of characters into a list of integers."""
    return [char_to_id[c] for c in string]

def decode(indices):
    """Converts a list of integers back into a readable string."""
    return ''.join([id_to_char[i] for i in indices])

# Let's test our tokenizer with a sample sentence!
sample_sentence = "Hello world!"
encoded_sample = encode(sample_sentence)
decoded_sample = decode(encoded_sample)

print("\n" + "=" * 60)
print("TOKENIZER TEST:")
print(f"Original text: '{sample_sentence}'")
print(f"Encoded tokens (numbers): {encoded_sample}")
print(f"Decoded back to text:     '{decoded_sample}'")
print("=" * 60)


# =====================================================================
# STEP 4: PREPARE THE TRAINING AND VALIDATION DATASETS
# We convert the entire 1.1 million characters into a giant 1D PyTorch tensor.
# 90% is used for training, and 10% is held out for validation (testing).
# =====================================================================
full_data_tensor = torch.tensor(encode(text), dtype=torch.long)

split_idx = int(0.9 * len(full_data_tensor))
train_data = full_data_tensor[:split_idx]
val_data = full_data_tensor[split_idx:]

print(f"\nTraining set size:   {len(train_data):,} tokens (90%)")
print(f"Validation set size: {len(val_data):,} tokens (10%)")


# =====================================================================
# STEP 5: HOW THE LLM TRAINS (INPUTS 'x' AND TARGETS 'y')
# LLMs are trained to predict the NEXT token in a sequence.
#
# If the text is 'hello', and context window is 4:
# When input is 'h', target is 'e'
# When input is 'he', target is 'l'
# When input is 'hel', target is 'l'
# When input is 'hell', target is 'o'
# Notice target 'y' is always input 'x' shifted to the right by 1!
# =====================================================================
torch.manual_seed(1337)

batch_size = 4  # How many independent sequences do we process at the same time?
block_size = 8  # What is the maximum context length (how far back can the model look)?

def get_batch(split='train'):
    """
    Grabs a small batch of data of inputs (x) and targets (y).
    """
    dataset = train_data if split == 'train' else val_data
    
    # Pick random starting spots in the text
    random_starts = torch.randint(len(dataset) - block_size, (batch_size,))
    
    # x is a slice of length block_size
    x = torch.stack([dataset[i : i + block_size] for i in random_starts])
    
    # y is the exact same slice shifted by 1 position forward
    y = torch.stack([dataset[i + 1 : i + block_size + 1] for i in random_starts])
    
    return x, y

xb, yb = get_batch('train')

print("\n" + "=" * 60)
print("BATCH DETAILS (What goes into the neural network):")
print(f"Batch Size (B): {batch_size} sequences in parallel")
print(f"Block Size (T): {block_size} characters of context per sequence")
print(f"xb (inputs) shape:  {xb.shape}  --> (Batch Size, Context Length)")
print(f"yb (targets) shape: {yb.shape}  --> (Batch Size, Context Length)")
print("-" * 60)
print("Input Tensor (xb):\n", xb)
print("Target Tensor (yb - next character to predict):\n", yb)
print("=" * 60)

# =====================================================================
# STEP 6: VISUALIZE THE TEACHING PROCESS
# Let's inspect the very first sequence in this batch token-by-token.
# =====================================================================
print("\n" + "=" * 60)
print("HOW THE MODEL LEARNS (Sequence 0 in the batch):")
print("-" * 60)
first_x = xb[0].tolist()
first_y = yb[0].tolist()

for t in range(block_size):
    context = first_x[: t + 1]
    target = first_y[t]
    print(f"Step {t+1}: Context: {context} ('{decode(context)}')  -->  Predict next: {target} ('{decode([target])}')")

print("=" * 60)
