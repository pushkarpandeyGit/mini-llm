# 🩺 MedMini-LLM: Domain-Specific Transformer from Scratch

A complete, production-grade **Decoder-Only Transformer Language Model** built entirely from scratch in **PyTorch**, incorporating modern **LLaMA-3 architectural enhancements** (RMSNorm + SwiGLU), domain-pretrained on clinical **Medical Q&A**, and served with a **FastAPI backend** and **React frontend**.

---

##  Key Technical Highlights

* **Architecture from Scratch:** Implemented multi-head causal self-attention, residual connections, and autoregressive generation without using pre-packaged Hugging Face model abstractions.
* **Modern Normalization (RMSNorm):** Replaced conventional LayerNorm with Root Mean Square Normalization (used in LLaMA 3, Gemma, Mistral), reducing compute overhead by skipping mean centering.
* **SwiGLU Gated Activation:** Implemented the Swish Gated Linear Unit in the feed-forward layers:
  $$\text{SwiGLU}(x) = \big(\text{SiLU}(xW_1) \otimes xW_3\big) W_2$$
* **Domain Pretraining:** Pretrained on a curated medical question-and-answer corpus covering cardiology, neurology, endocrinology, and emergency triage.
* **Training Convergence:** Cross-entropy loss successfully converged from **4.3893** (random guessing) down to **0.5479** in 1,500 steps.
* **Full-Stack Deployment:** Served via a lightweight **FastAPI REST API** with an interactive **React (Vite)** user interface.

---

##  Model Specifications

| Parameter | Value |
| :--- | :--- |
| **Total Parameters** | 207,680 |
| **Embedding Dimension ($d_{model}$)** | 64 |
| **Transformer Blocks** | 4 |
| **Attention Heads** | 4 (Head dimension = 16) |
| **Context Window ($T$)** | 32 tokens |
| **Vocabulary Size** | 70 unique tokens |
| **Optimizer** | AdamW ($\beta_1=0.9, \beta_2=0.999$, weight decay = 0.01) |
| **Initial Loss** | 4.3893 ($-\ln(1/70) \approx 4.2485$) |
| **Final Validation Loss** | **0.5479** |

---

##  Repository Structure

```
scratch_llm/
├── day1_data.py        
├── day2_model.py         
├── day3_train.py         
├── generate.py        
├── server.py             
├── med_model.pt         
├── metadata.json         
├── loss_curve.png      
├── medical_data.txt  
└── frontend/            
    ├── src/
    │   ├── App.jsx     
    │   └── App.css      
    └── package.json
```

---

## ⚡ How to Run the Project

### Option 1: 1-Click Launcher (Windows)
Double-click `start_all.bat` or run:
```powershell
.\start_all.bat
```

### Option 2: Manual Terminal Startup

**Terminal 1 — Start the Python API Backend:**
```powershell
python server.py
# Backend running at: http://127.0.0.1:8000
```

**Terminal 2 — Start the React Frontend:**
```powershell
cd frontend
npm run dev
# Open in browser: http://localhost:5173
```

---

