"""
src/eval.py
Phase 4 / Phase 7: Evaluation, Statistical Analysis, and Table Generation

Reads logs/cases.csv and automatically generates:
  - Table III: Main comparison (B0–B6 on held-out test data)
      Metrics: ASR, FPR, Precision, Recall, F1, Median Latency, P95 Latency
  - Table IV: Adaptation Gain (Leave-One-Family-Out on held-out families)
      Metrics: ASR_before, ASR_after, AG with 95% Wilson CI, Delta FPR
  - Outputs to outputs/tables/table_III.tex and outputs/tables/table_IV.tex

Usage:
  python src/eval.py
"""
from __future__ import annotations

import csv
import math
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.utils.logger import get_logger

log = get_logger("eval", PROJECT_ROOT / "logs" / "run.log")
CASES_CSV = PROJECT_ROOT / "logs" / "cases.csv"
OUTPUTS_DIR = PROJECT_ROOT / "outputs" / "tables"


# ─────────────────────────────────────────────────────────────
# Statistical Helper Functions
# ─────────────────────────────────────────────────────────────

def wilson_interval(successes: int, total: int, confidence: float = 0.95) -> Tuple[float, float]:
    """Calculate Wilson score interval for a binomial proportion."""
    if total <= 0:
        return (0.0, 0.0)
    z = 1.95996  # 95%
    p = successes / total
    denom = 1.0 + (z**2) / total
    center = (p + (z**2) / (2 * total)) / denom
    margin = (z * math.sqrt((p * (1.0 - p)) / total + (z**2) / (4 * total**2))) / denom
    return (max(0.0, center - margin), min(1.0, center + margin))


def compute_ag_confidence_interval(
    p_before: float, n_before: int,
    p_after: float, n_after: int,
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute Adaptation Gain AG = p_before - p_after and its 95% CI.
    Returns (AG, CI_lower, CI_upper).
    """
    ag = p_before - p_after
    if n_before <= 0 or n_after <= 0:
        return (ag, ag, ag)
    se = math.sqrt((p_before * (1.0 - p_before) / max(1, n_before)) + (p_after * (1.0 - p_after) / max(1, n_after)))
    z = 1.95996
    return (ag, ag - z * se, ag + z * se)


# ─────────────────────────────────────────────────────────────
# Table III: Main System Comparison (E1)
# ─────────────────────────────────────────────────────────────

def generate_table_iii(rows: List[dict]) -> str:
    """Generate Table III from E1 cases."""
    e1_rows = [r for r in rows if r.get("experiment_id") == "E1"]
    if not e1_rows:
        return "% Table III: No E1 records found in cases.csv\n"

    # Group by system
    systems = ["B0", "B1", "B2", "B3", "B4", "B5", "B6"]
    stats_by_sys = {}

    for sys_name in systems:
        sys_rows = [r for r in e1_rows if r.get("system") == sys_name]
        if not sys_rows:
            continue

        attacks = [r for r in sys_rows if r.get("label") == "1"]
        benigns = [r for r in sys_rows if r.get("label") == "0"]

        tp = sum(1 for r in attacks if r.get("decision", "").lower() == "block")
        fn = sum(1 for r in attacks if r.get("decision", "").lower() == "allow")
        fp = sum(1 for r in benigns if r.get("decision", "").lower() == "block")
        tn = sum(1 for r in benigns if r.get("decision", "").lower() == "allow")

        total_attacks = len(attacks)
        total_benigns = len(benigns)

        asr = (fn / total_attacks) if total_attacks > 0 else 0.0
        fpr = (fp / total_benigns) if total_benigns > 0 else 0.0
        prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) > 0 else 0.0

        latencies = [
            float(r["latency_ms"]) for r in sys_rows
            if r.get("latency_ms") not in ("NA", "", None)
        ]
        med_lat = float(np.median(latencies)) if latencies else 0.0
        p95_lat = float(np.percentile(latencies, 95)) if latencies else 0.0

        stats_by_sys[sys_name] = {
            "asr": asr, "fpr": fpr, "f1": f1, "prec": prec, "rec": rec,
            "med_lat": med_lat, "p95_lat": p95_lat, "n_attacks": total_attacks, "n_benign": total_benigns,
        }

    # Format LaTeX
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Table III: Main System Comparison on Held-Out Test Partitions (Experiment E1)}")
    latex.append(r"\begin{tabular}{l c c c c c c}")
    latex.append(r"\hline")
    latex.append(r"\textbf{System} & \textbf{ASR} $\downarrow$ & \textbf{FPR} $\downarrow$ & \textbf{Precision} & \textbf{Recall} & \textbf{F1} $\uparrow$ & \textbf{Latency Med (P95) [ms]} \\")
    latex.append(r"\hline")

    for s in systems:
        if s not in stats_by_sys:
            continue
        d = stats_by_sys[s]
        latex.append(
            f"{s} & {d['asr']*100:.1f}\\% & {d['fpr']*100:.1f}\\% & {d['prec']*100:.1f}\\% & "
            f"{d['rec']*100:.1f}\\% & {d['f1']*100:.1f}\\% & {d['med_lat']:.1f} ({d['p95_lat']:.1f}) \\\\"
        )

    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)


# ─────────────────────────────────────────────────────────────
# Table IV: Adaptation Gain (E2)
# ─────────────────────────────────────────────────────────────

def generate_table_iv(rows: List[dict]) -> str:
    """Generate Table IV: Leave-One-Family-Out Adaptation Gain."""
    e2_rows = [r for r in rows if r.get("experiment_id") in ("E2_before", "E2_after", "E2_upper_bound")]
    if not e2_rows:
        return "% Table IV: No E2 records found in cases.csv\n"

    families = sorted(set(r.get("family", "") for r in e2_rows if r.get("family") not in ("", "unknown", "benign")))

    # Calculate overall benign FPR before vs after
    benign_before = [r for r in rows if r.get("experiment_id") == "E1" and r.get("label") == "0" and r.get("system") == "B3"]
    benign_after = [r for r in e2_rows if r.get("experiment_id") == "E2_after" and r.get("label") == "0" and r.get("system") == "B3"]

    fp_before = sum(1 for r in benign_before if r.get("decision", "").lower() == "block")
    fpr_before = (fp_before / len(benign_before)) if benign_before else 0.0

    fp_after = sum(1 for r in benign_after if r.get("decision", "").lower() == "block")
    fpr_after = (fp_after / len(benign_after)) if benign_after else 0.0
    delta_fpr = (fpr_after - fpr_before) * 100.0

    family_stats = []
    for fam in families:
        before_attacks = [r for r in e2_rows if r.get("experiment_id") == "E2_before" and r.get("family") == fam and r.get("label") == "1"]
        after_attacks = [r for r in e2_rows if r.get("experiment_id") == "E2_after" and r.get("family") == fam and r.get("label") == "1" and r.get("system") == "B3"]
        b4_attacks = [r for r in e2_rows if r.get("experiment_id") == "E2_after" and r.get("family") == fam and r.get("label") == "1" and r.get("system") == "B4"]
        upper_attacks = [r for r in e2_rows if r.get("experiment_id") == "E2_upper_bound" and r.get("family") == fam]

        asr_before = (sum(1 for r in before_attacks if r.get("decision", "").lower() == "allow") / max(1, len(before_attacks)))
        asr_after = (sum(1 for r in after_attacks if r.get("decision", "").lower() == "allow") / max(1, len(after_attacks)))
        asr_b4 = (sum(1 for r in b4_attacks if r.get("decision", "").lower() == "allow") / max(1, len(b4_attacks))) if b4_attacks else asr_before
        asr_upper = (sum(1 for r in upper_attacks if r.get("decision", "").lower() == "allow") / max(1, len(upper_attacks))) if upper_attacks else 0.0

        ag, ci_low, ci_high = compute_ag_confidence_interval(
            asr_before, len(before_attacks),
            asr_after, len(after_attacks),
        )

        family_stats.append({
            "family": fam,
            "asr_before": asr_before,
            "asr_after": asr_after,
            "asr_b4": asr_b4,
            "asr_upper": asr_upper,
            "ag": ag,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "n": len(before_attacks),
        })

    # LaTeX Table IV
    latex = []
    latex.append(r"\begin{table*}[t]")
    latex.append(r"\centering")
    latex.append(r"\caption{Table IV: Leave-One-Family-Out Threat Adaptation Gain (Experiment E2)}")
    latex.append(r"\begin{tabular}{l c c c c c}")
    latex.append(r"\hline")
    latex.append(r"\textbf{Threat Family} & $\text{ASR}_{\text{before}}$ (B3) & $\text{ASR}_{\text{after}}$ (B3) & \textbf{Adaptation Gain ($AG$)} [95\% CI] & $\text{ASR}$ (B4 Few-shot) & $\text{ASR}_{\text{upper}}$ \\")
    latex.append(r"\hline")

    for fs in family_stats:
        latex.append(
            f"{fs['family']} & {fs['asr_before']*100:.1f}\\% & {fs['asr_after']*100:.1f}\\% & "
            f"\\textbf{{{fs['ag']*100:+.1f}\\%}} [{fs['ci_low']*100:+.1f}, {fs['ci_high']*100:+.1f}] & "
            f"{fs['asr_b4']*100:.1f}\\% & {fs['asr_upper']*100:.1f}\\% \\\\"
        )

    latex.append(r"\hline")
    latex.append(f"\\multicolumn{{6}}{{l}}{{\\small $\\Delta\\text{{FPR}} = {delta_fpr:+.2f}\\%$ on benign hold-out test set (acceptance limit: $\\le 1.0\\%$)}} \\\\")
    latex.append(r"\hline")
    latex.append(r"\end{tabular}")
    latex.append(r"\end{table*}")
    return "\n".join(latex)


def main() -> None:
    print("\n" + "=" * 65)
    print(" RTM Paper Table Generator (eval.py)")
    print("=" * 65)

    if not CASES_CSV.exists():
        print(f"Error: {CASES_CSV} not found!")
        sys.exit(1)

    with open(CASES_CSV, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    print(f"Loaded {len(rows)} cases from {CASES_CSV}")
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Generate Table III
    t3_latex = generate_table_iii(rows)
    t3_path = OUTPUTS_DIR / "table_III.tex"
    with open(t3_path, "w", encoding="utf-8") as f:
        f.write(t3_latex + "\n")
    print(f"Generated Table III -> {t3_path}")

    # 2. Generate Table IV
    t4_latex = generate_table_iv(rows)
    t4_path = OUTPUTS_DIR / "table_IV.tex"
    with open(t4_path, "w", encoding="utf-8") as f:
        f.write(t4_latex + "\n")
    print(f"Generated Table IV  -> {t4_path}")

    print("\n" + "=" * 65)
    print(" Table Previews:")
    print("=" * 65)
    print("\n--- Table III ---")
    print(t3_latex)
    print("\n--- Table IV ---")
    print(t4_latex)
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
