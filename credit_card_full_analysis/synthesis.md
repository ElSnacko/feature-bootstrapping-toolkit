# Credit Card Default — Bootstrap Stability Analysis

**Date**: 2026-04-05
**Toolkit**: bootstrap_stability
**Mode**: Option A (no model retraining per bootstrap)

## Dataset

- **Source**: UCI Default of Credit Card Clients (Taiwan, 2005)
- **Samples**: 30,000
- **Features**: 23 (numeric + categorical with `support_categorical=True`)
- **Target**: `default payment next month` (event rate = 22.12%)

---

## 1. Marginal (Distributional) Stability

Panel analysis across all 23 features using `k/n^0.5 + floor` learning curves
(fixed alpha=0.5), 25 bootstrap resamples per pool size.

Wasserstein distance is now normalized by IQR, making all metrics scale-invariant
and the overall complexity score directly comparable across features.

### Overall Complexity Ranking

| Feature | Complexity | Wasserstein Floor | KS Floor | Spearman Floor | IV Floor | Censored |
|---------|-----------|------------------|----------|---------------|----------|----------|
| PAY_AMT1 | -0.0061 | -0.0321 | 0.0030 | 0.1509 | 0.0894 | Yes |
| BILL_AMT1 | -0.0055 | -0.0016 | 0.0006 | 0.0147 | -0.0489 | No |
| BILL_AMT4 | -0.0037 | -0.0083 | 0.0008 | 0.0103 | -0.0171 | No |
| BILL_AMT2 | -0.0032 | -0.0008 | 0.0015 | 0.0096 | -0.0378 | No |
| LIMIT_BAL | -0.0023 | -0.0047 | -0.0029 | 0.1724 | 0.1263 | No |
| BILL_AMT3 | -0.0023 | -0.0029 | 0.0009 | 0.0096 | -0.0255 | No |
| AGE | -0.0017 | -0.0033 | -0.0014 | 0.0055 | -0.0212 | No |
| MARRIAGE | -0.0016 | 0.0020 | 0.0029 | 0.0003 | -0.0310 | No |
| PAY_5 | -0.0015 | -0.0034 | -0.0006 | 0.1728 | 0.4053 | Yes |
| BILL_AMT6 | -0.0012 | -0.0033 | 0.0010 | 0.0066 | -0.0355 | No |
| PAY_0 | -0.0004 | -0.0018 | -0.0005 | 0.3123 | 0.7794 | No |
| PAY_3 | 0.0001 | -0.0030 | 0.0002 | 0.2035 | 0.0459 | Yes |
| SEX | 0.0001 | 0.0001 | 0.0001 | 0.0348 | -0.0347 | Yes |
| EDUCATION | 0.0004 | 0.0001 | -0.0005 | 0.0426 | 0.0153 | No |
| BILL_AMT5 | 0.0005 | -0.0052 | 0.0016 | 0.0083 | -0.0161 | No |
| PAY_6 | 0.0022 | 0.0009 | 0.0015 | 0.1515 | 0.3127 | Yes |
| PAY_4 | 0.0026 | 0.0014 | 0.0016 | 0.1812 | 0.0288 | Yes |
| PAY_2 | 0.0034 | 0.0029 | 0.0014 | 0.2307 | 0.0348 | Yes |
| PAY_AMT5 | 0.0035 | -0.0205 | -0.0033 | 0.1110 | 0.0415 | Yes |
| PAY_AMT6 | 0.0116 | 0.0242 | 0.0004 | 0.1254 | 0.0676 | Yes |
| PAY_AMT2 | 0.0135 | 0.1562 | -0.0007 | 0.1570 | 0.1109 | Yes |
| PAY_AMT3 | 0.0163 | 0.0602 | 0.0008 | 0.1405 | 0.0767 | Yes |
| PAY_AMT4 | 0.0265 | 0.0370 | 0.0003 | 0.1406 | 0.0722 | Yes |

All complexity scores are now in the range **[-0.006, +0.027]** — directly comparable.

### Target-Dependent Stability (Spearman Floor)

Features ranked by irreducible instability of the feature-target relationship:

| Tier | Features | Spearman Floor Range |
|------|----------|---------------------|
| **Very stable** (< 0.02) | MARRIAGE, AGE, BILL_AMT1–6 | 0.0003 – 0.0147 |
| **Stable** (0.02–0.05) | SEX, EDUCATION | 0.035 – 0.043 |
| **Moderate** (0.05–0.18) | PAY_AMT1–6, LIMIT_BAL, PAY_4–6 | 0.111 – 0.173 |
| **Elevated** (0.18–0.25) | PAY_2, PAY_3 | 0.204 – 0.231 |
| **High** (> 0.25) | PAY_0 | 0.312 |

### Censoring Flags

11 of 23 features flagged for potential policy truncation:
- **PAY_AMT1–6**: Boundary spikes (many zeros / minimum payments)
- **SEX**: Binary (trivially flagged)
- **PAY_2–6**: Payment status boundary density spikes

---

## 2. SHAP (Model Decision) Stability

SHAP stability on 3,000 samples, LightGBM (100 trees, depth 6), Option A.

### Learning Curve Fit Results

| Metric | Floor | R² | Anomalous | Weight |
|--------|-------|-----|-----------|--------|
| rank_stability (inverted) | 0.0205 | 0.767 | No | 0.30 |
| wasserstein | 0.0073 | 0.983 | No | 0.15 |
| js_divergence | 0.0002 | 0.989 | No | 0.10 |
| magnitude_cv | -0.0079 | 0.975 | No | 0.15 |
| direction_consistency (inverted) | 0.4093 | 0.074 | **Yes** | 0.30 (excluded) |
| rank_stability_global (inverted) | -0.0027 | 0.830 | No | 0.00 |
| magnitude_iqr | -0.0110 | 0.985 | No | 0.00 |
| topk_overlap (inverted) | -0.0851 | 0.866 | No | 0.00 |

7/8 fit successfully. Weight coverage: **70%**.
**Overall SHAP complexity: 0.4197**

### Per-Feature SHAP Complexity

| Feature | SHAP Complexity | Direction Consistency | Rank Stability |
|---------|----------------|----------------------|----------------|
| PAY_2 | 0.114 | 0.673 | 0.976 |
| BILL_AMT5 | 0.119 | 0.664 | 0.969 |
| PAY_6 | 0.132 | 0.604 | 0.995 |
| MARRIAGE | 0.139 | 0.569 | 0.995 |
| SEX | 0.139 | 0.572 | 1.000 |
| PAY_AMT1 | 0.142 | 0.633 | 0.926 |
| EDUCATION | 0.142 | 0.606 | 0.976 |
| PAY_AMT4 | 0.145 | 0.569 | 0.982 |
| PAY_0 | 0.146 | 0.626 | 0.937 |
| PAY_AMT3 | 0.150 | 0.559 | 0.980 |
| PAY_AMT2 | 0.154 | 0.587 | 0.937 |
| BILL_AMT2 | 0.157 | 0.591 | 0.919 |
| PAY_AMT6 | 0.160 | 0.586 | 0.913 |
| BILL_AMT3 | 0.164 | 0.570 | 0.926 |
| BILL_AMT6 | 0.167 | 0.600 | 0.871 |
| PAY_5 | 0.168 | 0.540 | 0.942 |
| BILL_AMT4 | 0.171 | 0.601 | 0.858 |
| LIMIT_BAL | 0.176 | 0.589 | 0.856 |
| PAY_3 | 0.179 | 0.507 | 0.940 |
| BILL_AMT1 | 0.180 | 0.578 | 0.857 |
| AGE | 0.181 | 0.579 | 0.851 |
| PAY_AMT5 | 0.186 | 0.582 | 0.840 |
| PAY_4 | 0.201 | 0.563 | 0.807 |

---

## 3. Train/Holdout Drift Detection

70/30 stratified split, LightGBM, SHAP compared.

| Metric | Value |
|--------|-------|
| **Drift Grade** | **A** |
| **Overall Drift Score** | **0.057** |
| Rank Correlation | 0.998 |
| Direction Flip Rate | 17.4% |
| Top-10 Overlap | 1.000 |
| Magnitude Drift | 0.020 |

### Per-Feature Drift (top 10)

| Feature | Drift Score | Rank Change | Direction Consistent |
|---------|-------------|-------------|---------------------|
| PAY_AMT2 | 0.403 | +0 | **No** |
| SEX | 0.403 | +0 | **No** |
| BILL_AMT2 | 0.402 | +0 | **No** |
| MARRIAGE | 0.401 | +0 | **No** |
| PAY_4 | 0.016 | -1 | Yes |
| PAY_5 | 0.015 | +1 | Yes |
| BILL_AMT4 | 0.015 | -1 | Yes |
| BILL_AMT3 | 0.014 | +1 | Yes |
| BILL_AMT5 | 0.009 | +0 | Yes |
| EDUCATION | 0.007 | +0 | Yes |

The 4 direction-flipping features are all weak predictors with near-zero mean SHAP.

---

## 4. Reliability Scoring

| Feature | Reliability | Feature | Reliability |
|---------|------------|---------|------------|
| PAY_0 | 0.988 | PAY_AMT6 | 0.772 |
| BILL_AMT3 | 0.973 | PAY_6 | 0.757 |
| PAY_AMT2 | 0.964 | BILL_AMT5 | 0.757 |
| PAY_AMT1 | 0.843 | PAY_3 | 0.754 |
| PAY_AMT3 | 0.839 | EDUCATION | 0.728 |
| PAY_AMT4 | 0.838 | PAY_5 | 0.728 |
| BILL_AMT1 | 0.838 | PAY_2 | 0.719 |
| PAY_AMT5 | 0.830 | MARRIAGE | 0.703 |
| BILL_AMT2 | 0.818 | SEX | 0.703 |
| PAY_4 | 0.815 | | |
| BILL_AMT4 | 0.798 | | |
| LIMIT_BAL | 0.796 | | |
| AGE | 0.781 | | |
| BILL_AMT6 | 0.775 | | |

All features now score 0.703+. PAY_AMT4 (previously an outlier at 0.449 due to
Wasserstein scale effects) is now 0.838.

---

## 5. Key Insights

1. **Payment history (PAY_0–6) dominates prediction but has the highest marginal instability.**
   PAY_0's Spearman floor of 0.31 means its rank correlation with default fluctuates
   substantially across resamples. This is structural — more data won't help. However,
   the model's use of PAY_0 is stable (SHAP complexity 0.146, direction consistency 0.63).

2. **Bill amount features (BILL_AMT1–6) are the most target-stable.**
   Spearman floors < 0.015. Good candidates for stable scorecards.

3. **Marginal vs SHAP stability diverge meaningfully.**
   PAY_2 is marginally unstable (Spearman 0.23) but SHAP-stable (complexity 0.114,
   ranked 1st). The model captures its signal consistently despite distributional
   fluctuation. Conversely, AGE is marginally stable (Spearman 0.006) but
   SHAP-unstable (complexity 0.181, ranked 21st).

4. **The model generalizes well.** Drift grade A (0.057), rank correlation 0.998.

5. **Censoring affects payment amounts.** PAY_AMT1–6 have boundary spikes (many zeros).
   Spearman floors (~0.11–0.16) reveal moderate instability that marginal Wasserstein
   partially conceals.

6. **IQR-normalized Wasserstein produces sensible cross-feature comparisons.**
   Before normalization, LIMIT_BAL had overall complexity -398 and PAY_AMT4 had +49.5,
   both dominated by raw Wasserstein in dollars. After normalization, the range is
   [-0.006, +0.027] — all features directly comparable.

---

## 6. Bugs Found & Fixed

| Bug | Location | Status |
|-----|----------|--------|
| `aggregate_shap_metrics` crashes on None pool entries | `shap_metrics.py:656` | **Fixed** |
| Stability metrics not inverted before SHAP curve fitting | `shap_stability.py:804` | **Fixed** |
| Wasserstein distance not scale-invariant | `core.py:MetricRunner` | **Fixed** (÷ IQR) |
| `direction_consistency` flat under Option A | `shap_stability.py` | Design limitation |

---

## Files Generated

| File | Description |
|------|-------------|
| `marginal_panel_summary.csv` | Marginal stability scores for all 23 features |
| `marginal_panel_chart.png` | Bar chart of complexity scores |
| `marginal_LIMIT_BAL.png/.csv` | Deep dive: credit limit |
| `marginal_PAY_0.png/.csv` | Deep dive: payment status |
| `marginal_AGE.png/.csv` | Deep dive: age |
| `marginal_BILL_AMT1.png/.csv` | Deep dive: bill amount |
| `marginal_all_results.json` | Full marginal results (floors, flags, types) |
| `shap_results.json` | SHAP stability scores and per-metric floors |
| `drift_results.json` | Train/holdout drift metrics per feature |
| `reliability_results.json` | Reliability scores per feature |
| `synthesis.md` | This file |
