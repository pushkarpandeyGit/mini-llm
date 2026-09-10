import React, { useState, useEffect } from "react";
import "./App.css";

const PRESET_QUERIES = [
  "What is Hypertension and how is it clinically defined?",
  "What are the hallmark symptoms and diagnostic criteria for Type 2 Diabetes?",
  "What is Asthma and how is an acute bronchospasm managed?",
  "How do you recognize an acute ischemic stroke using the FAST protocol?"
];

function App() {
  const [question, setQuestion] = useState(PRESET_QUERIES[0]);
  const [temperature, setTemperature] = useState(0.35);
  const [maxTokens, setMaxTokens] = useState(180);
  const [response, setResponse] = useState("");
  const [loading, setLoading] = useState(false);
  const [copied, setCopied] = useState(false);
  const [error, setError] = useState(null);
  const [showLossPlot, setShowLossPlot] = useState(false);
  const [modelInfo, setModelInfo] = useState({
    parameters: 462490,
    context_window: 128,
    normalizer: "RMSNorm",
    feed_forward: "SwiGLU"
  });

  useEffect(() => {
    fetch("http://127.0.0.1:8000/api/info")
      .then((res) => res.json())
      .then((data) => setModelInfo(data))
      .catch(() => {
        // Fallback defaults if backend is initializing
      });
  }, []);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!question.trim()) return;

    setLoading(true);
    setError(null);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:8000/api/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          question: question.trim(),
          temperature: parseFloat(temperature),
          max_tokens: parseInt(maxTokens)
        })
      });

      if (!res.ok) {
        throw new Error("Unable to connect to the model backend. Please verify server.py is running.");
      }

      const data = await res.json();
      setResponse(data.response || data.full_text);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = () => {
    if (!response) return;
    navigator.clipboard.writeText(response);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="page-wrapper">
      <div className="main-content">
        {/* Header */}
        <header className="page-header">
          <div className="project-tag">Custom PyTorch Model</div>
          <h1 className="title">MedMini-LLM</h1>
          <p className="description">
            A domain-specific Decoder-Only Transformer for clinical question answering.
            Built from scratch with modern RMSNorm and SwiGLU feed-forward layers.
          </p>
        </header>

        {/* Technical Specs Bar */}
        <div className="spec-row">
          <div className="spec-item">
            <span className="spec-title">Parameters</span>
            <span className="spec-value">{modelInfo.parameters ? modelInfo.parameters.toLocaleString() : "462,490"}</span>
          </div>
          <div className="spec-item">
            <span className="spec-title">Context Window</span>
            <span className="spec-value">{modelInfo.context_window || 128} tokens</span>
          </div>
          <div className="spec-item">
            <span className="spec-title">Normalization</span>
            <span className="spec-value">RMSNorm</span>
          </div>
          <div className="spec-item">
            <span className="spec-title">Architecture</span>
            <span className="spec-value">LLaMA-3 Style</span>
          </div>
        </div>

        {/* Query Form */}
        <div className="form-card">
          <form onSubmit={handleSubmit}>
            <div className="field-group">
              <label htmlFor="query-input" className="field-label">
                Medical Question
              </label>
              <textarea
                id="query-input"
                className="text-input"
                rows={3}
                value={question}
                onChange={(e) => setQuestion(e.target.value)}
                placeholder="Enter a medical question or select one of the suggested prompts below."
                disabled={loading}
              />
            </div>

            {/* Quick Presets */}
            <div className="presets-wrapper">
              <span className="presets-caption">Preset Prompts:</span>
              <div className="preset-buttons">
                {PRESET_QUERIES.map((item, i) => (
                  <button
                    type="button"
                    key={i}
                    className={`preset-btn ${question === item ? "preset-btn-selected" : ""}`}
                    onClick={() => setQuestion(item)}
                  >
                    {item}
                  </button>
                ))}
              </div>
            </div>

            {/* Control Sliders */}
            <div className="parameters-grid">
              <div className="param-box">
                <div className="param-header">
                  <span className="param-name">Sampling Temperature</span>
                  <span className="param-val">{temperature}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(e.target.value)}
                  className="slider"
                />
                <span className="param-caption">Lower values produce more deterministic medical facts.</span>
              </div>

              <div className="param-box">
                <div className="param-header">
                  <span className="param-name">Maximum Length</span>
                  <span className="param-val">{maxTokens} tokens</span>
                </div>
                <input
                  type="range"
                  min="50"
                  max="300"
                  step="10"
                  value={maxTokens}
                  onChange={(e) => setMaxTokens(e.target.value)}
                  className="slider"
                />
                <span className="param-caption">Maximum number of characters to generate.</span>
              </div>
            </div>

            {/* Submit */}
            <button type="submit" className="action-button" disabled={loading}>
              {loading ? "Generating response..." : "Generate Answer"}
            </button>
          </form>

          {/* Error Display */}
          {error && (
            <div className="alert-box">
              <p className="alert-text">{error}</p>
              <p className="alert-subtext">Ensure `python server.py` is active in the project root directory.</p>
            </div>
          )}

          {/* Response Container */}
          {response && (
            <div className="output-section">
              <div className="output-header">
                <span className="output-title">Generated Clinical Answer</span>
                <button type="button" className="copy-action" onClick={handleCopy}>
                  {copied ? "Copied" : "Copy text"}
                </button>
              </div>
              <div className="output-content">{response}</div>
            </div>
          )}
        </div>

        {/* Training Diagnostics Toggle */}
        <div className="diagnostics-card">
          <button
            type="button"
            className="toggle-button"
            onClick={() => setShowLossPlot(!showLossPlot)}
          >
            {showLossPlot ? "Hide Training Loss Curve" : "View Training Loss Curve"}
          </button>

          {showLossPlot && (
            <div className="plot-container">
              <img
                src="http://127.0.0.1:8000/loss_curve.png"
                alt="Training Loss Curve"
                className="loss-image"
                onError={(e) => {
                  e.target.style.display = "none";
                }}
              />
              <p className="plot-caption">
                Training and validation cross-entropy loss tracking convergence across training iterations.
              </p>
            </div>
          )}
        </div>

        <footer className="page-footer">
          <p>MedMini-LLM &bull; PyTorch Neural Architecture &bull; React User Interface</p>
        </footer>
      </div>
    </div>
  );
}

export default App;
