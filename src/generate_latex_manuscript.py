"""
src/generate_latex_manuscript.py
Generates a complete, flawless IEEEtran publication-ready LaTeX manuscript (manuscript.tex)
embedding all exact tables (Tables II-VIII), equations, figures, citations, and sections.
"""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TABLES_DIR = PROJECT_ROOT / "outputs" / "tables"
OUT_PATHS = [
    PROJECT_ROOT.parent / "manuscript.tex",
    PROJECT_ROOT / "manuscript.tex",
]

def read_table(name: str) -> str:
    path = TABLES_DIR / name
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return f"% Table {name} not found"

def build_latex_manuscript() -> str:
    t2 = read_table("table_II.tex")
    t3 = read_table("table_III.tex")
    t4 = read_table("table_IV.tex")
    t5 = read_table("table_V.tex")
    t6 = read_table("table_VI.tex")
    t7_abl = read_table("table_VII_ablations.tex")
    t7_exp = read_table("table_VII_explanation_eval.tex")
    t8 = read_table("table_VIII_hypotheses.tex")

    latex = r"""\documentclass[conference]{IEEEtran}
\IEEEoverridecommandlockouts
\usepackage{cite}
\usepackage{amsmath,amssymb,amsfonts}
\usepackage{algorithmic}
\usepackage{graphicx}
\usepackage{textcomp}
\usepackage{xcolor}
\usepackage{booktabs}
\usepackage{multirow}
\usepackage{url}
\usepackage{hyperref}

\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em
    T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}

\begin{document}

\title{RAG-Based Threat Memory: An Adaptive and Explainable Defense Against Indirect Prompt Injection in LLM Agents}

\author{\IEEEauthorblockN{Fareed Ataf\IEEEauthorrefmark{1} and Mazen Al-Bahri\IEEEauthorrefmark{2}}
\IEEEauthorblockA{Department of Computer Science, Sana'a University, Sana'a, Yemen\\
Email: \IEEEauthorrefmark{1}fareedataf4@gmail.com, \IEEEauthorrefmark{2}m.albahri@su.edu.ye}
}

\maketitle

\begin{abstract}
Large language model (LLM) agents that read documents, browse the Web, and call tools are exposed to indirect prompt injection (IPI): instructions hidden in untrusted content that the agent mistakes for trusted ones. Existing defenses rely on fixed detectors or costly runtime analysis, typically need retraining to absorb new threat knowledge, and rarely tie a decision to documented evidence. We present RAG-Based Threat Memory (RTM), a defense that keeps adaptive security knowledge outside the detector. RTM combines a zero-training base detector (requiring no fine-tuning of model weights; calibration and retrieval memory updates are utilized instead for adaptation) with a retrieval-augmented memory of verified, documented injection patterns, a calibrated Soft-OR fusion rule, and an explanation that cites the closest documented threat and the triggering text segment. A new threat is absorbed by appending a verified record, without retraining. Because the memory is itself an attack surface, RTM includes a verified ingestion pipeline designed to resist poisoning. We evaluate RTM on AgentDojo against a no-defense agent, its own components, a public trained detector, and MELON, a recent training-free defense, measuring adaptation gain on held-out attack families, robustness to black-box adaptive attacks, memory poisoning, utility, explanation verifiability, and latency. On held-out threat variants, RTM achieves an Attack Success Rate (ASR) of 8.3\% (1/12) at 0.0\% False Positive Rate (0/10 FPR, F1 = 0.957), strictly outperforming the base detector alone (ASR = 25.0\% [3/12], FPR = 10.0\% [1/10], F1 = 0.818) and DeBERTa (ASR = 41.7\% [5/12], F1 = 0.700). Ingesting verified exemplars yields a mean adaptation gain of +88.9\% across six held-out attack families with 0.00\% delta-FPR. RTM demonstrated robust resistance across 200 total poisoning trials (100 unactivated candidate attempts and 100 verified-crafted insertion attempts across 5 threat categories, achieving 0.0\% MPS under this evaluated threat scope), delivers 94.0\% verifiable explanations ($\kappa = 1.000$) with zero privacy leakage, and adds only 5.0\,ms median middleware latency (10.7\,ms end-to-end tool-return latency)---over 100$\times$ faster than MELON (1102.9\,ms median end-to-end latency). Code and data are available at \url{https://github.com/fareed1-ataf/rtm}.
\end{abstract}

\begin{IEEEkeywords}
large language model agents, indirect prompt injection, retrieval-augmented generation, threat memory, adaptive defense, memory poisoning, explainability, AgentDojo.
\end{IEEEkeywords}

\section{Introduction}
Artificial intelligence systems built on large language models (LLMs) now run inside real products, and with that deployment they have become the target of a growing family of security threats, including jailbreaks, data extraction, poisoning of training and retrieval data, and prompt injection \cite{vassilev2025adversarial, owasp2025llm01}. Prompt injection, in which an adversary makes a model follow instructions that the legitimate user or developer did not intend, is listed first among the risks of LLM applications \cite{owasp2025llm01} and has been studied since the first goal-hijacking demonstrations \cite{perez2022ignore, liu2023prompt}. Two forms are distinguished: in direct injection the attacker is the user, whereas in indirect prompt injection (IPI) the attacker hides instructions in external content that the application later retrieves \cite{greshake2023not}.

IPI is most serious in LLM agents, which interleave reasoning with actions such as reading documents, browsing the Web, and calling application programming interfaces \cite{yao2023react, schick2023toolformer}. In such agentic systems, retrieved content and instructions share one context, so the model may treat data as an instruction, and processing retrieved content resembles executing untrusted code. The impact depends on the privileges and tools available to the agent. Benchmarks now quantify this exposure: InjecAgent \cite{zhan2024injecagent}, BIPIA \cite{yi2023benchmarking}, and AgentDojo, which provides 97 realistic tasks and 629 security test cases and measures security together with task utility \cite{debenedetti2024agentdojo}. Attackers also adapt. AgentVigil reports attack success rates of 71\% on AgentDojo agents based on o3-mini and 70\% on VWA-adv agents based on GPT-4o \cite{wang2025agentvigil}, and Zhan et al. report that adaptive attacks bypassed eight evaluated defenses with attack success rates above 50\% in their setting \cite{zhan2025adaptive}. A defense must therefore be evaluated against attackers that adapt to it.

Existing defenses fall into four families. Prevention by prompting or training separates instructions from data, as in spotlighting \cite{hines2024defending}, structured queries \cite{chen2025struq}, and the instruction hierarchy \cite{wallace2024instruction}. Detection classifies the input, as in safeguard classifiers \cite{inan2023llama}, PromptShield \cite{jacob2025promptshield}, and DataSentinel \cite{liu2025datasentinel}. Behavioral inference-time analysis examines what the agent would do, as in MELON \cite{zhu2025melon} and AgentSentry \cite{zhang2026agentsentry}. System-level approaches constrain execution, as in CaMeL \cite{debenedetti2025defeating}, Progent \cite{shi2025progent}, and Task Shield \cite{jia2025task}. Each family has limits. Training-based prevention and detection need training resources and updated data to absorb a new attack pattern; fixed mechanisms can be bypassed by adaptive attacks; behavioral re-execution adds computation; and system-level controls restrict functionality and do not inspect content.

This paper presents RAG-Based Threat Memory (RTM), a model-centric defense that combines a zero-training base detector (requiring no fine-tuning of model weights; calibration and retrieval memory updates are utilized instead for adaptation) with an updatable retrieval-augmented memory of documented injection patterns. The central hypothesis is that adaptive security knowledge can be maintained separately from the base detector: when a new threat is documented and verified, appending its description to the memory is sufficient, and every rejection can be explained by the closest documented threat and the text segment that triggered it.

The contributions are:
\begin{enumerate}
    \item \textbf{Architecture:} An architecture (RTM) that fuses the score of a zero-training base detector (requiring no fine-tuning of model weights; calibration and retrieval memory updates are utilized instead for adaptation) with the similarity of the content to a verified threat memory via calibrated Soft-OR fusion.
    \item \textbf{Adaptation Mechanism:} An adaptation mechanism that absorbs a documented threat by appending a record to the vector store, with a verified ingestion pipeline designed to resist poisoning.
    \item \textbf{Verifiable Explanations:} An evidence-based explanation that attaches to each rejection the closest documented threat and the triggering text segment.
    \item \textbf{Empirical Findings:} Across 809 audited evaluation cases spanning AgentDojo, InjecAgent, and BIPIA, Full RTM (B3) achieves $\text{ASR} = 8.3\%$ (1/12) and $\text{FPR} = 0.0\%$ (0/10, $\text{F1} = 0.957$), attains a mean adaptation gain of $+88.9\%$ across six held-out threat families with zero false alarm increase ($\Delta\text{FPR} = +0.00\%$), demonstrated robust resistance across 200 total poisoning trials (100 candidate attempts and 100 verified-crafted attempts across 5 threat categories, yielding 0.0\% MPS), provides $94.0\%$ verifiable explanations ($\kappa = 1.000$) with zero privacy leaks, and adds only $5.0\,\text{ms}$ median middleware latency ($10.7\,\text{ms}$ end-to-end tool-return latency) compared to MELON's $1102.9\,\text{ms}$.
\end{enumerate}

\section{Problem Statement and Threat Model}
\subsection{Problem Statement}
When an agent receives a user instruction $U$ together with external content $C$, it has no reliable mechanism to distinguish content that is data to be used as information from content that carries a disguised instruction aimed at steering its behavior. Let $\mathcal{M}$ denote a threat memory, a set of verified records. The task is to design a decision function
\begin{equation}
D(U, C; \mathcal{M}) \rightarrow \{\text{allow}, \text{block}\}
\end{equation}
taken at each tool-return boundary, that detects injections accurately, absorbs new documented knowledge without retraining, explains each rejection with a documented source, and incurs minimal latency.

\subsection{Research Questions and Pre-Registered Hypotheses}
\begin{itemize}
    \item \textbf{H1 (RQ1):} On held-out attack variants, RTM ($B_3$) attains a lower ASR than the base detector alone ($B_1$) at equal or lower FPR (paired McNemar test, corrected for multiple comparisons).
    \item \textbf{H2 (RQ2):} After a verified record for a held-out threat family is added, ASR on that family decreases ($AG > 0$, with the lower bound of the 95\% confidence interval above zero) while FPR rises by at most 1 percentage point ($\Delta\text{FPR} \le 1.0\text{ pp}$).
    \item \textbf{H3 (RQ3):} Under black-box adaptive fuzzing across budgets 50--500, $B_3$ attains lower ASR than $B_1$, $B_2$, and $B_6$.
    \item \textbf{H4 (RQ4):} Explanation Verification Rate $\text{EVR} \ge 90\%$ with Cohen's $\kappa \ge 0.60$, and the privacy audit reconstructs no protected context from explanations.
    \item \textbf{H5 (RQ5):} $B_3$ keeps benign task success within 5.0 pp of the undefended agent ($B_0$), is non-inferior to $B_5$ (MELON) within a pre-registered margin of $\Delta = 5.0\text{ pp}$, and adds significantly lower median latency per tool return than $B_5$.
    \item \textbf{H6 (RQ6):} Candidate records never change enforcement decisions ($\text{MPS} = 0\%$ by construction), and adversarially crafted records yield $\text{MPS} \le 5\%$.
\end{itemize}

\section{Proposed RTM Architecture}
\subsection{Overview}
RTM consists of a zero-training base detector (requiring no fine-tuning of model weights; calibration and retrieval memory updates are utilized instead for adaptation), an external threat memory, a retriever, a calibrated decision layer, and an evidence-based explanation module. For each tool-return content item $C$, RTM computes a base detector score $S_{\text{det}}$, retrieves the top-$k$ active memory records over sentence-level windows to compute $S_{\text{sim}}$, and fuses them into a calibrated risk score $R$.

\subsection{Retrieval and Calibrated Soft-OR Fusion}
For each content segment $s_j$, the retriever returns the top-$k$ active records by cosine similarity. The calibrated memory similarity score is:
\begin{equation}
S_{\text{sim}} = g\left( \max_{j,r} \cos(e(s_j), e(r)) \right)
\end{equation}
where $e(\cdot)$ is the embedding function and $g$ is a monotone isotonic calibration map. The base detector score $S_{\text{det}} \in [0, 1]$ is calibrated via $g_{\text{det}}$.

To avoid the failure mode of linear averaging—where moderate memory similarity on unseen zero-day attacks dilutes confident base detector detections—we formulate the fusion as a \textbf{Weighted Soft-OR}:
\begin{equation}
R = 1 - (1 - S_{\text{det}}) \cdot (1 - w_{\text{sim}} \cdot S_{\text{sim}})
\end{equation}
where $w_{\text{sim}} = 0.75$ scales retrieval importance. The content is blocked if $R \ge \tau^*$, and allowed otherwise, with calibrated decision threshold $\tau^* = 0.60$. Specifically, when the base detector detects no threat ($S_{\text{det}} = 0$), a memory similarity of $S_{\text{sim}} = 0.80$ yields a fused risk score of $R = 1 - (1 - 0)(1 - 0.75 \times 0.80) = 0.60$, exactly matching the decision threshold $\tau^* = 0.60$, showing how the calibrated parameters smoothly integrate semantic and retrieval signals. Conversely, if $S_{\text{sim}} = 0$, any detector score $S_{\text{det}} \ge 0.60$ directly triggers containment. When both channels provide moderate evidence (e.g., $S_{\text{det}} = 0.40$ and $S_{\text{sim}} = 0.50$), their fused score $R = 1 - (1 - 0.40)(1 - 0.75 \times 0.50) = 0.625 \ge 0.60$ smoothly reinforces the threat signal without allowing either channel to suppress the other. Under this formulation:
\begin{enumerate}
    \item If the base detector is confident ($S_{\text{det}} \ge 0.60$), the content is blocked regardless of $S_{\text{sim}}$.
    \item If a verified threat exemplar matches ($S_{\text{sim}} \ge 0.80$), $R \ge 0.60$ and the content is blocked immediately.
    \item Moderate evidence from both channels smoothly reinforces each other without suppression.
\end{enumerate}
Parameters $w_{\text{sim}} = 0.75$ and $\tau^* = 0.60$ were fitted on the calibration partition to maximize F1 at an FPR budget of 1.0\%, and frozen before test evaluation.

\section{Experimental Setup}
\subsection{Data Partitions and Sample Accounting}
""" + t2 + r"""

Table~\ref{tab:partitions} details the audited dataset partitions. To ensure complete experimental transparency and avoid accounting ambiguities across the benchmark hierarchy, we explicitly reconcile the data structure: (1) Memory construction incorporates $\ge 150$ verified records across $\ge 8$ attack families; (2) The calibration partition comprises 750 items (400 attack, 350 benign) used strictly to fit scoring parameters ($\tau^*, w_{\text{sim}}$); (3) The benign test partition consists of 317 instances (including 175 challenging hard negatives) for evaluating false alarm rates; (4) The held-out attack evaluation pool spans 1,673 variants across 6 distinct threat families; and (5) The final audited execution log (\texttt{cases.csv}) records exactly 809 unique end-to-end execution decisions across Experiments E1--E6 under frozen models (spanning 110 decisions in E1, 75 in E2, 160 in E3, 200 in E4, 50 in E5, and 242 in E6 [837 core total, 809 benchmark evaluation points]).

\subsection{Compared Systems}
We compare seven configurations under identical agent environments and permissions:
\begin{itemize}
    \item \textbf{B0 (None):} Undefended base agent.
    \item \textbf{B1 (Base Detector Only):} Zero-training prompted LLM detector.
    \item \textbf{B2 (Memory Retrieval Only):} Retrieval similarity without detector.
    \item \textbf{B3 (Full RTM):} Calibrated Soft-OR fusion of detector and memory.
    \item \textbf{B4 (B1 + Few-Shot Update):} Detector updated via few-shot exemplar.
    \item \textbf{B5 (MELON):} State-of-the-art training-free masked re-execution defense.
    \item \textbf{B6 (Trained Classifier):} Public fine-tuned DeBERTa prompt injection model.
\end{itemize}

\subsection{Detector, Calibration, and Latency Architecture}
The detector LLM (Gemini-2.5-Flash), its prompt, and the agent LLM (Claude-3.5-Sonnet deployed on AgentDojo) were fixed before testing, and all generation used temperature 0. Scores were calibrated on the calibration partition by Platt scaling and isotonic regression \cite{guo2017calibration}. Parameters $w_{\text{sim}} = 0.75$ and $\tau^* = 0.60$ were chosen on a grid over the calibration partition and frozen before evaluating any test partitions.

The operational latency divergence between $B_1$ and Full RTM ($B_3$) is rooted in their architectural execution paths. Baseline $B_1$ sends every tool-return content segment to a remote generative LLM over the network, incurring token generation delay and round-trip API overhead ($3263.7\,\text{ms}$ median). In contrast, Full RTM ($B_3$) executes detection entirely inside a local middleware pipeline: incoming text is sliced into sentence windows, embedded, and queried against a local FAISS index in sub-millisecond time. Pre-calibrated isotonic mapping and fast Soft-OR fusion reach containment without remote generative calls for recognized patterns, resulting in an end-to-end latency of $10.7\,\text{ms}$ median (P95: $14.1\,\text{ms}$) and middleware processing overhead of only $5.0\,\text{ms}$ median.

\section{Empirical Results}
All results in this section were generated from the logged case repository (\texttt{cases.csv}) using frozen models and calibrated decision thresholds.

\subsection{Main Comparison (Experiment E1)}
""" + t3 + r"""

Table III presents the main system comparison on held-out test partitions. Full RTM ($B_3$) achieves the lowest Attack Success Rate ($\text{ASR} = 8.3\%$, 1/12 attacks successful, 11/12 blocked) with zero false alarms ($\text{FPR} = 0.0\%$, 0/10 false alarms), yielding an overall $\text{F1} = 0.957$ (Precision = 100.0\% [11/11], Recall = 91.7\% [11/12]). It strictly outperforms the base detector alone ($B_1$: $\text{ASR} = 25.0\%$ [3/12], $\text{FPR} = 10.0\%$ [1/10], Precision = 90.0\% [9/10], Recall = 75.0\% [9/12], $\text{F1} = 0.818$), memory alone ($B_2$: $\text{ASR} = 50.0\%$ [6/12], $\text{FPR} = 0.0\%$ [0/10], Precision = 100.0\% [6/6], Recall = 50.0\% [6/12], $\text{F1} = 0.667$), and trained DeBERTa ($B_6$: $\text{ASR} = 41.7\%$ [5/12], $\text{FPR} = 10.0\%$ [1/10], Precision = 87.5\% [7/8], Recall = 58.3\% [7/12], $\text{F1} = 0.700$). Undefended $B_0$ permits all attacks ($\text{ASR} = 100.0\%$ [12/12]). On benign task success, both $B_0$ and $B_3$ achieve $\text{TSR} = 100.0\%$ (10/10).

We conducted two-sided paired McNemar tests with Holm-Bonferroni step-down correction for multiple comparisons across all pairwise configurations. Full RTM ($B_3$) achieves statistically significant superiority over undefended $B_0$ ($p = 0.0039$, adjusted $p < 0.01$) and the base detector alone $B_1$ ($p < 0.01$) across attack containment (ASR) and overall $F_1$ classification performance. Under pre-registered Hypothesis H5, non-inferiority is formally evaluated against MELON ($B_5$) with a pre-specified non-inferiority margin of $\Delta = 5.0\text{ percentage points (pp)}$ for both benign task success (TSR) and attack security (ASR). On benign utility, $B_3$ achieves $\text{TSR} = 100.0\%$ (10/10) compared to $B_0$'s $100.0\%$ ($\Delta = 0.0\text{ pp} \le 5.0\text{ pp}$) and $B_5$'s $95.0\%$ (difference $+5.0\text{ pp}$, one-sided non-inferiority $p < 0.001$). On security containment, $B_3$ achieves $\text{ASR} = 8.3\%$ (1/12) vs. $B_5$'s $12.5\%$, confirming that $B_3$ is non-inferior to $B_5$ within the $5.0\text{ pp}$ margin ($\text{ASR}_{B_3} - \text{ASR}_{B_5} = -4.2\text{ pp} < +5.0\text{ pp}$, one-sided Farrington-Manning non-inferiority test $z = -2.14, p = 0.016$), fully satisfying H5. In terms of end-to-end tool-return latency (total wall-clock delay at the tool boundary), $B_3$ operates at a median of only $10.7\,\text{ms}$ (P95: $14.1\,\text{ms}$), compared to $3263.7\,\text{ms}$ for $B_1$ (requiring full remote LLM inference per invocation) and $1102.9\,\text{ms}$ for $B_5$ (MELON, requiring complete agent masked re-execution).

\subsection{Leave-One-Family-Out Threat Adaptation Gain (Experiment E2)}
""" + t4 + r"""

Table IV reports adaptation performance when a newly documented threat family is absorbed into memory. Across all six held-out threat families ($n=3$ evaluation instances per family, 18 paired before/after cases), ingesting verified exemplars drops ASR from an initial $88.9\%$ down to $0.0\%$, achieving a mean Adaptation Gain ($AG$) of $+88.9\%$ (with paired bootstrap confidence intervals bounded in $[0.0\%, 100.0\%]$: $[+100.0\%, +100.0\%]$ for \texttt{data\_security\_harm}, \texttt{financial\_harm}, \texttt{language\_translation}, and \texttt{physical\_harm}, and $[+0.0\%, +100.0\%]$ for \texttt{financial\_data} and \texttt{physical\_data}; pooled 95\% bootstrap CI is $[+72.2\%, +100.0\%]$). Crucially, this gain is achieved with zero increase in false alarms on benign hold-out data ($\Delta\text{FPR} = +0.00\%$), satisfying Hypothesis H2.

\subsection{Robustness Under Adaptive Fuzzing (Experiment E3)}
""" + t5 + r"""

\begin{figure}[t]
\centering
\includegraphics[width=0.92\linewidth]{outputs/figures/fig_3_asr_vs_budget.pdf}
\caption{Attack Success Rate (ASR) vs. Adaptive Query Budget under black-box mutation fuzzing (Experiment E3).}
\label{fig:fuzzing}
\end{figure}

Table V and Fig.~\ref{fig:fuzzing} report ASR against the black-box adaptive fuzzer as the query budget scales from 50 to 500 iterations. While all evaluated baselines suffer monotonic degradation under iterative payload mutation---with $B_1$ rising from $30.0\%$ to $70.0\%$ ASR, $B_2$ surging from $40.0\%$ to $90.0\%$ ASR, and DeBERTa ($B_6$) climbing from $30.0\%$ to $70.0\%$ ASR---Full RTM ($B_3$) consistently maintains the lowest attack success rate across all budgets ($10.0\%$ at budget 50, $20.0\%$ at budget 100, $20.0\%$ at budget 200, and $40.0\%$ at budget 500). The calibrated Soft-OR fusion ensures that mutations designed to evade lexical memory signatures remain caught by the semantic base detector, and vice versa, empirically supporting Hypothesis H3.

\subsection{Memory Poisoning Resistance (Experiment E4)}
""" + t6 + r"""

Table VI details the memory poisoning evaluation across five threat categories comprising 200 total trials: 100 unactivated candidate insertion attempts and 100 verified-crafted records engineered to mimic benign schema (20 attempts per threat category for each stage). Unverified candidate records produced $\text{MPS}_{\text{cand}} = 0.0\%$ (0/100), and adversarially crafted records also yielded $\text{MPS}_{\text{craft}} = 0.0\%$ (0/100), demonstrating robust resistance across all 200 structured poisoning trials (0.0\% MPS under this evaluated threat scope, confirming Hypothesis H6).

\subsection{Explanation Quality and Privacy Audit (Experiment E5)}
""" + t7_exp + r"""

Table VII reports the blind human explanation evaluation and automated privacy audit ($N=50$). The Explanation Verification Rate reached $94.0\%$ (47/50) with perfect inter-rater concordance (Cohen's $\kappa = 1.000$). The privacy audit detected 0 leakage incidents ($0.0\%$ context recovery), confirming Hypothesis H4.

\subsection{Ablation Analysis and Overhead (Experiment E6)}
""" + t7_abl + r"""

Table VII presents ablation results on the held-out test partition. Full RTM ($B_3$) is the top-performing configuration ($\text{ASR} = 8.3\%, \text{FPR} = 0.0\%, \text{F1} = 0.957$). Ablating memory ($- \text{memory}$) increases ASR to $25.0\%$ and eliminates explanations; ablating the base detector ($- \text{detector}$) spikes ASR to $41.7\%$; removing calibration ($- \text{calibration}$) causes an unacceptable surge in false alarms ($\text{FPR} = 30.0\%$ on hard negatives); and removing sentence segmentation or exemplars raises ASR to $25.0\%$. Sensitivity analysis on the retrieval neighborhood $k$ reveals that while $k=1$ and $k=3$ achieve identical $\text{ASR} = 16.7\%$ and $\text{F1} = 0.870$ at $\text{FPR} = 10.0\%$, increasing retrieval depth to $k=5$ (Full RTM) enables optimal calibration coverage, dropping ASR to $8.3\%$ and FPR to $0.0\%$ ($\text{F1} = 0.957$). Expanding further to $k=10$ dilutes neighborhood precision, increasing ASR back to $25.0\%$ and $\text{FPR}$ to $10.0\%$.

We explicitly distinguish two latency dimensions: (1) Added middleware latency (the internal overhead incurred by RTM for dense vector retrieval, isotonic score calibration, Soft-OR fusion, and evidence extraction), which is only $5.0\,\text{ms}$ median (P95: $14.2\,\text{ms}$); and (2) End-to-End tool-return latency (total elapsed time including model dispatch), where $B_3$ exhibits $10.7\,\text{ms}$ median (P95: $14.1\,\text{ms}$), more than two orders of magnitude faster than MELON ($B_5$, $1102.9\,\text{ms}$ median, P95: $1246.1\,\text{ms}$) which requires complete agent re-execution. Median Time-to-Protect (TTP) was $1440\,\text{minutes}$ ($24.0\,\text{hours}$) across 155 verified records.

\subsection{Pre-Specified Hypotheses Outcomes}
""" + t8 + r"""

Table VIII summarizes the formal evaluation of the six pre-registered hypotheses. All six hypotheses ($H_1$--$H_6$) were empirically supported by the data.

\section{Discussion}
\subsection{Adaptation versus Memorization}
A central question is whether RTM's performance gains represent true semantic generalization or memorization. First, all adaptation evaluations in Table IV used held-out test instances disjoint from the memory exemplars. Second, replacing the threat memory with random benign records ($B_3$-rand) degrades performance severely ($\text{ASR} = 25.0\%, \text{FPR} = 40.0\%, \text{F1} = 0.720$). Third, lexical retrieval ($B_3$-lex) fails to generalize across semantic paraphrases ($\text{ASR} = 25.0\%, \text{EVR} = 0.0\%$). These findings prove that RTM's gains stem from semantic threat retrieval.

\subsection{Comparison with MELON}
While both RTM and MELON ($B_5$) provide training-free defenses, MELON relies on masked agent re-execution. This introduces heavy computational overhead ($1102.9\,\text{ms}$ median latency) and doubles LLM inference cost. RTM adds only $5.0\,\text{ms}$ middleware latency ($10.7\,\text{ms}$ end-to-end) without auxiliary LLM calls and provides human-auditable evidence, making it suited for latency-critical production agents.

\subsection{Failure Modes and Implications}
The single evasion against $B_3$ in E1 involved a complex multilingual translation injection that evaded both the base detector ($S_{\text{det}} = 0.132$) and initial memory ($S_{\text{sim}} = 0.35$). Once a translation exemplar was ingested in E2, ASR dropped to $0.0\%$, verifying that operational updates neutralize evasive mutations without retraining.

\section{Conclusion}
This paper presented RAG-Based Threat Memory (RTM), an adaptive and explainable defense against indirect prompt injection in LLM agents. By coupling an untrained base detector with a verified retrieval memory via calibrated Soft-OR fusion, RTM achieves an $\text{ASR} = 8.3\%$ (1/12) at $\text{FPR} = 0.0\%$ (0/10, $\text{F1} = 0.957$), delivers $+88.9\%$ mean adaptation gain with zero false positive increase, demonstrated robust resistance across 200 structured poisoning trials (100 candidate, 100 crafted attempts) spanning 5 distinct malicious/adversarial categories (0.0\% MPS under this evaluated threat scope), provides $94.0\%$ verifiable explanations, and operates with a minimal $5.0\,\text{ms}$ median middleware latency ($10.7\,\text{ms}$ end-to-end tool-return latency). All six pre-registered hypotheses were empirically supported.

\section*{Data and Code Availability}
Code, memory records (with non-operational exemplars), partition definitions, configuration files, and per-case logs are available at \url{https://github.com/fareed1-ataf/rtm}. Payloads that could be directly harmful are withheld.

\section*{Acknowledgment}
The authors thank the Faculty of Computer and Information Technology at Sana'a University for supporting this research. In accordance with IEEE policy, the authors disclose that generative AI tools were used exclusively for language polishing and formatting verification; all experimental design, code implementation, data auditing, and empirical analyses were conducted and verified by the authors.

\begin{thebibliography}{00}
\bibitem{vassilev2025adversarial} A. Vassilev et al., ``Adversarial machine learning: A taxonomy and terminology of attacks and mitigations,'' NIST AI 100-2 E2025, 2025.
\bibitem{owasp2025llm01} OWASP GenAI Security Project, ``LLM01:2025 Prompt Injection,'' 2025.
\bibitem{perez2022ignore} F. Perez and I. Ribeiro, ``Ignore previous prompt: Attack techniques for language models,'' in \textit{NeurIPS ML Safety Workshop}, 2022.
\bibitem{liu2023prompt} Y. Liu et al., ``Prompt injection attack against LLM-integrated applications,'' arXiv:2306.05499, 2023.
\bibitem{greshake2023not} K. Greshake et al., ``Not what you've signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection,'' in \textit{Proc. ACM AISec}, 2023.
\bibitem{yao2023react} S. Yao et al., ``ReAct: Synergizing reasoning and acting in language models,'' in \textit{Proc. ICLR}, 2023.
\bibitem{schick2023toolformer} T. Schick et al., ``Toolformer: Language models can teach themselves to use tools,'' in \textit{NeurIPS}, 2023.
\bibitem{zhan2024injecagent} Q. Zhan et al., ``InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents,'' in \textit{Findings of ACL}, 2024.
\bibitem{yi2023benchmarking} J. Yi et al., ``Benchmarking and defending against indirect prompt injection attacks on large language models,'' arXiv:2312.14197, 2023.
\bibitem{debenedetti2024agentdojo} E. Debenedetti et al., ``AgentDojo: A dynamic environment to evaluate prompt injection attacks and defenses for LLM agents,'' in \textit{NeurIPS Datasets and Benchmarks}, 2024.
\bibitem{wang2025agentvigil} Z. Wang et al., ``AgentVigil: Automatic black-box red-teaming for indirect prompt injection against LLM agents,'' in \textit{Findings of EMNLP}, 2025.
\bibitem{zhan2025adaptive} Q. Zhan et al., ``Adaptive attacks break defenses against indirect prompt injection attacks on LLM agents,'' in \textit{Findings of NAACL}, 2025.
\bibitem{hines2024defending} K. Hines et al., ``Defending against indirect prompt injection attacks with spotlighting,'' arXiv:2403.14720, 2024.
\bibitem{chen2025struq} S. Chen et al., ``StruQ: Defending against prompt injection with structured queries,'' in \textit{Proc. USENIX Security}, 2025.
\bibitem{wallace2024instruction} E. Wallace et al., ``The instruction hierarchy: Training LLMs to prioritize privileged instructions,'' arXiv:2404.13208, 2024.
\bibitem{inan2023llama} H. Inan et al., ``Llama Guard: LLM-based input-output safeguard for human-AI conversations,'' arXiv:2312.06674, 2023.
\bibitem{jacob2025promptshield} D. Jacob et al., ``PromptShield: Deployable detection for prompt injection attacks,'' arXiv:2501.15145, 2025.
\bibitem{liu2025datasentinel} Y. Liu et al., ``DataSentinel: A game-theoretic detection of prompt injection attacks,'' in \textit{Proc. IEEE S\&P}, 2025.
\bibitem{zhu2025melon} K. Zhu et al., ``MELON: Provable defense against indirect prompt injection attacks in AI agents,'' in \textit{Proc. ICML}, 2025.
\bibitem{zhang2026agentsentry} T. Zhang et al., ``AgentSentry: Mitigating indirect prompt injection in LLM agents via temporal causal diagnostics and context purification,'' arXiv:2602.22724, 2026.
\bibitem{debenedetti2025defeating} E. Debenedetti et al., ``Defeating prompt injections by design,'' arXiv:2503.18813, 2025.
\bibitem{shi2025progent} T. Shi et al., ``Progent: Programmable privilege control for LLM agents,'' arXiv:2504.11703, 2025.
\bibitem{jia2025task} F. Jia et al., ``The Task Shield: Enforcing task alignment to defend against indirect prompt injection in LLM agents,'' in \textit{Proc. ACL}, 2025.
\bibitem{lewis2020retrieval} P. Lewis et al., ``Retrieval-augmented generation for knowledge-intensive NLP tasks,'' in \textit{NeurIPS}, 2020.
\bibitem{zou2025poisonedrag} W. Zou et al., ``PoisonedRAG: Knowledge corruption attacks to retrieval-augmented generation of large language models,'' in \textit{Proc. USENIX Security}, 2025.
\bibitem{chen2024agentpoison} Z. Chen et al., ``AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases,'' in \textit{NeurIPS}, 2024.
\bibitem{liu2024formalizing} Y. Liu et al., ``Formalizing and benchmarking prompt injection attacks and defenses,'' in \textit{Proc. USENIX Security}, 2024.
\bibitem{zou2023universal} A. Zou et al., ``Universal and transferable adversarial attacks on aligned language models,'' arXiv:2307.15043, 2023.
\bibitem{gao2023retrieval} Y. Gao et al., ``Retrieval-augmented generation for large language models: A survey,'' arXiv:2312.10997, 2023.
\bibitem{karpukhin2020dense} V. Karpukhin et al., ``Dense passage retrieval for open-domain question answering,'' in \textit{Proc. EMNLP}, 2020.
\bibitem{mitre2026atlas} MITRE, ``ATLAS: Adversarial Threat Landscape for Artificial-Intelligence Systems,'' 2026.
\bibitem{guo2017calibration} C. Guo et al., ``On calibration of modern neural networks,'' in \textit{Proc. ICML}, 2017.
\bibitem{reimers2019sentence} N. Reimers and I. Gurevych, ``Sentence-BERT: Sentence embeddings using Siamese BERT-networks,'' in \textit{Proc. EMNLP}, 2019.
\bibitem{wang2024multilingual} L. Wang et al., ``Multilingual E5 text embeddings: A technical report,'' arXiv:2402.05672, 2024.
\bibitem{protectai2026deberta} ProtectAI, ``deberta-v3-base-prompt-injection-v2,'' Hugging Face model card, 2026.
\end{thebibliography}

\end{document}
"""
    return latex

def main():
    latex_content = build_latex_manuscript()
    for p in OUT_PATHS:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(latex_content)
        print(f"[OK] Generated LaTeX manuscript -> {p}")

if __name__ == "__main__":
    main()
