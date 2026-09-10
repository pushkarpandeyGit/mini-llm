import streamlit as st
import torch
import json
import os
from day2_model import MedMiniGPT, device, block_size

# Page configuration
st.set_page_config(
    page_title="MedMini-LLM | Transformer from Scratch",
    page_icon="🩺",
    layout="wide"
)

# Header Section
st.title("🩺 MedMini-LLM: Medical Transformer from Scratch")
st.markdown("""
**A domain-specific Decoder-Only Transformer built and trained entirely from scratch in PyTorch.**
Features modern 2026 architecture enhancements (**RMSNorm** + **SwiGLU activation** as used in **LLaMA 3**).
""")

# Load Model & Vocabulary Cached
@st.cache_resource
def load_med_model():
    with open('metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    
    chars = meta['vocab']
    char_to_id = {ch: i for i, ch in enumerate(chars)}
    id_to_char = {i: ch for i, ch in enumerate(chars)}
    
    model = MedMiniGPT().to(device)
    model.load_state_dict(torch.load('med_model.pt', map_location=device))
    model.eval()
    
    return model, char_to_id, id_to_char, meta

model, char_to_id, id_to_char, meta = load_med_model()

def encode(s):
    return [char_to_id.get(c, 0) for c in s]

def decode(l):
    return ''.join([id_to_char.get(i, '') for i in l])

# Model Specs Metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Parameters", "207,680", "Trained on CPU")
col2.metric("Architecture", "Decoder-Only GPT", "LLaMA-3 Style")
col3.metric("Normalizer", "RMSNorm", "Pre-Norm")
col4.metric("Final Val Loss", f"{meta['final_val_loss']:.4f}", "-3.84 from init")

st.divider()

# Sidebar: Generation Settings
st.sidebar.header("⚙️ Generation Parameters")
temperature = st.sidebar.slider("Sampling Temperature", min_value=0.1, max_value=1.2, value=0.5, step=0.05,
                               help="Lower temperature = more deterministic / strictly learned answers. Higher = more creative.")
max_tokens = st.sidebar.slider("Max Generation Length (Tokens)", min_value=50, max_value=300, value=180, step=10)

st.sidebar.subheader("Presets (Click to try):")
preset = st.sidebar.radio(
    "Choose an example question:",
    [
        "What is Hypertension and how is it clinically defined?",
        "What are the hallmark symptoms and diagnostic criteria for Type 2 Diabetes?",
        "What is Asthma and how is an acute bronchospasm managed?",
        "How do you recognize an acute ischemic stroke using the FAST protocol?"
    ]
)

# Tabs
tab1, tab2, tab3 = st.tabs(["💬 Interactive Q&A", "📈 Training & Loss Curve", "🧠 Architecture Deep Dive"])

with tab1:
    st.subheader("Ask the Medical Model")
    user_question = st.text_input("Enter your medical inquiry:", value=preset)
    
    if st.button("🚀 Generate Clinical Answer", type="primary"):
        with st.spinner("Model is autoregressively predicting next tokens..."):
            prompt = f"Question: {user_question.strip()}\nAnswer:"
            encoded = torch.tensor([encode(prompt)], dtype=torch.long, device=device)
            
            with torch.no_grad():
                generated = model.generate(encoded, max_new_tokens=max_tokens, temperature=temperature)
            
            raw_text = decode(generated[0].tolist())
            
            st.success("Generation Complete!")
            st.markdown("### Generated Response:")
            st.code(raw_text, language="text")

with tab2:
    st.subheader("Training Convergence & Loss Curve")
    st.markdown("""
    The graph below shows the cross-entropy loss dropping across **1,500 training steps**.
    Notice the rapid descent from random initialization (**4.38**) down to (**0.54**), demonstrating that the self-attention heads successfully learned the clinical relationships.
    """)
    if os.path.exists("loss_curve.png"):
        st.image("loss_curve.png", caption="Training vs Validation Loss (MedMiniGPT)", use_container_width=True)
    else:
        st.info("Run `python day3_train.py` to generate the loss curve.")

with tab3:
    st.subheader("Inside the Architecture")
    st.markdown("""
    ### Why this model is modern (2026 Resume Ready):
    
    1. **RMSNorm (Root Mean Square Normalization):**
       $$\\text{RMSNorm}(x) = \\frac{x}{\\sqrt{\\frac{1}{d}\\sum_{i=1}^d x_i^2 + \\epsilon}} \\odot \\gamma$$
       - Replaced traditional LayerNorm. Skips mean-centering, saving ~20% compute while preserving stability.
    
    2. **SwiGLU Gated Feed-Forward Network:**
       $$\\text{SwiGLU}(x) = \\big(\\text{SiLU}(xW_1) \\otimes xW_3\\big) W_2$$
       - Modern replacement for standard ReLU/GELU. Dynamic gating allows higher representation density with fewer parameters.
    
    3. **Causal Attention Masking:**
       - Strictly enforces an upper-triangular $-\\infty$ mask to prevent tokens from cheating by looking ahead into future tokens.
    """)

st.caption("Developed as part of the From-Scratch LLM Engineering Project.")
