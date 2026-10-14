# RTM — RAG-Based Threat Memory

> Adaptive and Explainable Defense Against Indirect Prompt Injection in LLM Agents
> IEEE Research Paper Implementation | v4

---

## Quick Start

### 1. Set up secrets
```bash
cp config/secrets.yaml.template config/secrets.yaml
# Edit config/secrets.yaml and fill in your API keys
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run B0 smoke test (Phase 0 verification)
```bash
python src/smoke_test_b0.py
```

---

## Project Structure

```
rtm_project/
├── config/              # All configuration — never hardcode anything
│   ├── models.yaml      # Model names and endpoints (dynamic switching)
│   ├── experiment.yaml  # Experiment parameters
│   ├── paths.yaml       # All file paths
│   └── secrets.yaml     # API keys — NEVER commit (git-ignored)
├── prereg.yaml          # Pre-registration — frozen before any test
├── data/
│   ├── raw/             # Downloaded datasets (read-only)
│   ├── memory.jsonl     # Verified threat records (>= 150)
│   ├── partitions/      # Audited, split datasets
│   └── verification_log.csv
├── src/                 # All source code
├── logs/cases.csv       # Unified raw log (all experiments)
├── outputs/             # Auto-generated tables and figures
└── tests/               # Unit tests
```

---

## Execution Phases

| Phase | Goal | Gate |
|-------|------|------|
| 0 | Infrastructure + B0 smoke test | Human approval |
| 1 | Data audit + partitioning | Human approval |
| 2 | Baselines B5 + B6 verified | Human approval |
| 3 | Memory + calibration FROZEN | Human approval |
| 4 | Core experiments E1 + E2 | Human approval |
| 5 | Advanced experiments E3–E6 | Human approval |
| 6 | Table + figure generation | Project complete |

---

## Critical Rules

1. **No data fabrication** — every number comes from `logs/cases.csv` via `eval.py`
2. **100% English** — all code, files, variables, comments
3. **API keys** in `config/secrets.yaml` only — never in code
4. **Models are dynamic** — swap by editing `config/models.yaml`
5. **α and τ frozen** after Phase 3 — `calibrate.py` enforces this
6. **leakage_check.py and consistency_check.py** must both exit 0 before tables are generated

---

## Authors

Fareed Ataf & Mazen Al-Bahri — Sana'a University
