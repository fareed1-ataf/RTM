# RTM: RAG-Based Threat Memory

> **Adaptive and Explainable Defense Against Indirect Prompt Injection in LLM Agents**  
> IEEE Transactions on Dependable and Secure Computing (TDSC) / IEEE S&P Submission  
> Authors: Fareed Ataf & Mazen Al-Bahri (Sana'a University)

---

## 🌟 Overview

**RTM (RAG-Based Threat Memory)** is a defense middleware designed to protect autonomous LLM agents against Indirect Prompt Injection (IPI). By unifying dense semantic retrieval (FAISS + all-MiniLM-L6-v2) over verified threat experiences with an LLM-based boundary detector through a **Weighted Soft-OR fusion rule**, RTM achieves zero-shot generalization to unseen threat families and sub-millisecond mitigation latency without requiring LLM retraining.

### Key Empirical Results (Audited 849 Executions across E1–E6)
- **Robust Attack Mitigation:** Outperforms single-modality baselines on held-out threat families (ASR: 8.3% vs. 25.0% for detector-only and 33.3% for memory-only).
- **Near-Zero False Alarms:** 0.0% False Positive Rate on verified benign test inputs.
- **Fast Local Decisioning:** 10.7 ms median inference latency for local retrieval and calibrated fusion (compared to 3263.7 ms for remote LLM calls).
- **Formal Invariance:** Passes all paired McNemar hypothesis tests (H1–H6) with Holm-Bonferroni correction ($\alpha = 0.05$).

---

## 🚀 Quick Start & Reproduction

You can replicate all empirical tables and statistical tests in the paper directly from the audited execution log (`logs/cases.csv`) without needing external API keys.

### 1. Installation

Clone the repository and install the verified dependencies:

```bash
git clone https://github.com/your-username/rtm.git
cd rtm/rtm_project
pip install -r requirements.txt
```

### 2. Verify Mathematical Integrity & Generate All Paper Tables

To verify dataset invariants, execute paired McNemar significance tests, and regenerate all LaTeX tables (Tables II–VIII) and figures (Figure 3):

```bash
python src/eval.py
```

Outputs will be populated in:
- `outputs/tables/` (LaTeX source code for Tables II, III, IV, V, VI, VII, and VIII)
- `outputs/figures/` (Vector PDF and PNG plots for ASR vs. query budget)
- `outputs/statistical_report.md` (Holm-Bonferroni adjusted $p$-values and hypothesis verification)

### 3. Run Test Suite

To run all 170 unit and integration tests across detector, retriever, middleware, and calibrations:

```bash
pytest
```

---

## ⚙️ Configuration & Live Evaluation (Optional)

If you wish to run live evaluations with Google Gemini or local Ollama endpoints:

1. Copy the secrets template:
   ```bash
   cp config/secrets.yaml.template config/secrets.yaml
   ```
2. Open `config/secrets.yaml` and provide your Gemini API key or local Ollama address.
3. Configure models and endpoints in `config/models.yaml` (default uses `gemini-1.5-flash` and `all-MiniLM-L6-v2`).
4. Run live evaluation:
   ```bash
   python src/run_e1.py
   ```

---

## 📂 Repository Structure

```
rtm_project/
├── config/
│   ├── experiment.yaml          # Experiment hyperparameters (tau*=0.60, w_sim=0.75)
│   ├── models.yaml              # Model endpoints and prompt paths
│   ├── paths.yaml               # Canonical data and logging paths
│   ├── prereg.yaml              # Pre-registered frozen parameters and hypotheses
│   └── secrets.yaml.template    # Template for API secrets (never commit secrets.yaml)
├── data/
│   ├── memory.jsonl             # Curated, verified threat memory records (>= 150 items)
│   ├── calibration_models.pkl   # Pre-trained isotonic calibration models
│   ├── faiss_index.bin          # Indexed dense vector embeddings of memory items
│   └── partitions/              # Exact splits for calibration, testing, and adaptive seeds
├── logs/
│   ├── cases.csv                # Canonical execution log (849 audited decision rows)
│   └── data_inventory.csv       # Line-item audit of all dataset sources and licenses
├── outputs/
│   ├── figures/                 # Generated plots (fig_3_asr_vs_budget.png, .pdf)
│   ├── tables/                  # Generated LaTeX tables (table_II.tex through table_VIII.tex)
│   ├── statistical_report.json  # Full machine-readable statistical test results
│   └── statistical_report.md    # Formatted statistical summary
├── src/
│   ├── rtm_middleware.py        # Core RTM middleware with Weighted Soft-OR fusion
│   ├── detector.py              # LLM boundary detector and parser
│   ├── ingest.py                # Embedding generator and FAISS indexer
│   ├── calibrate.py             # Isotonic calibration mapping
│   ├── eval.py                  # Master evaluation and table generator script
│   ├── fuzz.py                  # Adaptive fuzzing engine (E3 evaluation)
│   └── poison.py                # Poisoning robustness evaluation (E4 evaluation)
├── tests/                       # 170 comprehensive unit tests
├── manuscript.tex               # Full LaTeX manuscript
└── requirements.txt             # Pinned Python package dependencies
```

---

## 🛡️ Mathematical Formulation: Weighted Soft-OR Fusion

RTM computes fused risk score $R \in [0, 1]$ via the continuous relaxation:
$$R = 1 - (1 - S_{\text{det}}) \cdot (1 - w_{\text{sim}} \cdot S_{\text{sim}})$$

Where:
- $S_{\text{det}} = f_{\text{cal}}(\text{raw\_det}) \in [0, 1]$ is the isotonically calibrated boundary detector confidence.
- $S_{\text{sim}} = g_{\text{cal}}(\max_k \text{sim}(q, m_k)) \in [0, 1]$ is the calibrated semantic retrieval similarity score.
- $w_{\text{sim}} = 0.75$ is the memory discounting weight.
- Decision rule: $\text{Block if } R \ge \tau^* \ (0.60), \text{ else Allow}$.

---

## 📜 Citation & License

This research is released for scientific peer review and reproducibility.
- **Code License:** Apache 2.0
- **Dataset License:** See `logs/data_inventory.csv` for upstream dataset attributions (AgentDojo, BIPIA, SecTool, JailbreakHub).
