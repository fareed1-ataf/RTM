# RTM Final Statistical Evaluation Report

- **Total Rows Analyzed**: 849
- **Mathematical Invariants**: Fully Verified (TP+FN == Total Attacks, FP+TN == Total Benign)

## McNemar's Paired Tests (B3 vs Baselines on E1)

| Comparison | N Pairs | Both Correct | B3 Only | Baseline Only | Neither | McNemar Stat | Raw p | Adj p (Holm) | Significant (alpha=0.05) |
|---|---|---|---|---|---|---|---|---|---|
| B3 vs B0 | 22 | 10 | 11 | 0 | 1 | 11.000 | 0.0010 | 0.0039 | YES |
| B3 vs B1 | 22 | 18 | 3 | 0 | 1 | 3.000 | 0.2500 | 0.2500 | NO |
| B3 vs B2 | 22 | 16 | 5 | 0 | 1 | 5.000 | 0.0625 | 0.1875 | NO |
| B3 vs B6 | 22 | 16 | 5 | 0 | 1 | 5.000 | 0.0625 | 0.1875 | NO |

## Adaptive Fuzzing Summary (Table V & Figure 3)

| System | Budget 50 | Budget 100 | Budget 200 | Budget 500 |
|---|---|---|---|---|
| B1 | 30.0% | 40.0% | 60.0% | 70.0% |
| B2 | 40.0% | 60.0% | 80.0% | 90.0% |
| B3 | 10.0% | 20.0% | 20.0% | 40.0% |
| B6 | 30.0% | 40.0% | 50.0% | 70.0% |
