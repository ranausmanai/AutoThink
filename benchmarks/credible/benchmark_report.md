# Benchmark Report

Generated: 2026-02-16 22:06:25 UTC
Budgets: [10, 30, 60]
Seeds: [42, 1337, 2025]

## Run Health

- Total runs: 81
- Successful runs: 81
- Failed runs: 0

## Summary (mean +- 95% CI)

### Budget = 10s

#### Heart Disease (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.95245 +- 0.00524 | 7.95s +- 0.43s |
| AutoThink V4 | 0.95299 +- 0.00445 | 9.32s +- 0.51s |
| FLAML | 0.95245 +- 0.00596 | 10.07s +- 0.05s |

#### Loan Repayment (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.91165 +- 0.01683 | 10.49s +- 0.08s |
| AutoThink V4 | 0.91236 +- 0.01778 | 16.30s +- 0.47s |
| FLAML | 0.90902 +- 0.02191 | 10.46s +- 0.33s |

#### House Price (regression, 5K) (RMSE)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 30589.44 +- 381.04 | 6.57s +- 0.39s |
| AutoThink V4 | 31627.97 +- 321.55 | 11.39s +- 0.39s |
| FLAML | 30917.18 +- 260.50 | 10.26s +- 0.04s |

### Budget = 30s

#### Heart Disease (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.95245 +- 0.00524 | 8.78s +- 1.36s |
| AutoThink V4 | 0.95299 +- 0.00445 | 9.71s +- 0.38s |
| FLAML | 0.95254 +- 0.00518 | 30.06s +- 0.03s |

#### Loan Repayment (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.91165 +- 0.01683 | 11.23s +- 0.08s |
| AutoThink V4 | 0.91236 +- 0.01778 | 15.18s +- 0.71s |
| FLAML | 0.91191 +- 0.01884 | 30.42s +- 0.42s |

#### House Price (regression, 5K) (RMSE)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 30589.44 +- 381.04 | 6.76s +- 0.12s |
| AutoThink V4 | 31627.97 +- 321.55 | 11.49s +- 0.51s |
| FLAML | 30743.65 +- 474.66 | 30.99s +- 0.33s |

### Budget = 60s

#### Heart Disease (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.95245 +- 0.00524 | 10.02s +- 1.82s |
| AutoThink V4 | 0.95299 +- 0.00445 | 11.20s +- 1.12s |
| FLAML | 0.95335 +- 0.00478 | 60.24s +- 0.11s |

#### Loan Repayment (binary, 10K) (AUC)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 0.91165 +- 0.01683 | 10.97s +- 0.53s |
| AutoThink V4 | 0.91236 +- 0.01778 | 14.46s +- 0.04s |
| FLAML | 0.91387 +- 0.01779 | 60.58s +- 0.50s |

#### House Price (regression, 5K) (RMSE)

| Tool | Quality | Time |
|------|---------|------|
| AutoGluon | 30589.44 +- 381.04 | 6.76s +- 0.37s |
| AutoThink V4 | 31627.97 +- 321.55 | 11.64s +- 0.58s |
| FLAML | 30623.40 +- 476.28 | 61.98s +- 0.63s |
