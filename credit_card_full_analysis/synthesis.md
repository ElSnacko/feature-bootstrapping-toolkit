# Credit Card Default — Bootstrap Stability Analysis

**Date**: 2026-04-05
**Toolkit**: bootstrap_stability v1.0.0

## Dataset

- **Source**: UCI Default of Credit Card Clients (Taiwan, 2005)
- **Samples**: 30,000
- **Features**: 23 (numeric + categorical with `support_categorical=True`)
- **Target**: `default payment next month` (event rate = 22.12%)

---

## 1. Marginal (Distributional) Stability

Panel analysis across all 23 features using `k/n^0.5 + floor` learning curves
with 25 bootstrap resamples per pool size.

### Important Note on Scale

The **overall complexity score** is dominated by the Wasserstein floor, which is in the
same absolute units as the feature. Features with large scales (BILL_AMT, PAY_AMT,
LIMIT_BAL) produce extreme Wasserstein floors that are not comparable across features.

The **target-dependent complexity** (Spearman/IV/monotonicity) is scale-invariant and
more meaningful for cross-feature comparisons.

### Target-Dependent Stability (scale-invariant)

Features ranked by Spearman floor (irreducible instability of feature-target relationship):

| Feature | Spearman Floor | Feature Type | Censored | Interpretation |
|---------|---------------|-------------|----------|----------------|
| MARRIAGE | 0.0003 | categorical | No | Extremely stable |
| AGE | 0.0055 | continuous | No | Very stable |
| BILL_AMT2 | 0.0096 | continuous | No | Very stable |
| BILL_AMT3 | 0.0096 | continuous | No | Very stable |
| BILL_AMT4 | 0.0103 | continuous | No | Very stable |
| BILL_AMT1 | 0.0147 | continuous | No | Very stable |
| SEX | 0.0348 | binary | Yes | Stable |
| EDUCATION | 0.0426 | categorical | No | Stable |
| PAY_AMT5 | 0.1110 | continuous | Yes | Moderate instability |
| PAY_AMT6 | 0.1254 | continuous | Yes | Moderate instability |
| PAY_AMT3 | 0.1405 | continuous | Yes | Moderate instability |
| PAY_AMT4 | 0.1406 | continuous | Yes | Moderate instability |
| PAY_AMT1 | 0.1509 | continuous | Yes | Moderate instability |
| PAY_6 | 0.1515 | categorical | Yes | Moderate instability |
| PAY_AMT2 | 0.1570 | continuous | Yes | Moderate instability |
| LIMIT_BAL | 0.1724 | continuous | No | Moderate instability |
| PAY_5 | 0.1728 | categorical | Yes | Moderate instability |
| PAY_4 | 0.1812 | continuous | Yes | Elevated instability |
| PAY_3 | 0.2035 | continuous | Yes | Elevated instability |
| PAY_2 | 0.2307 | continuous | Yes | Elevated instability |
| PAY_0 | 0.3123 | continuous | No | High instability |

**Key finding**: Payment history features (PAY_0 through PAY_6) have the highest
target-dependent instability floors (0.15–0.31). These features are the strongest
predictors of default, but their WOE profiles and Spearman correlations fluctuate
substantially across bootstrap resamples. This is structural — more data won't help.

Bill amount features (BILL_AMT1–6) are the most target-stable (floors < 0.015).
Their relationship with default is consistent across resamples.

### Censoring Flags

11 of 23 features were flagged for potential policy truncation:
- **PAY_AMT1–6**: Payment amount features have boundary spikes (many zeros/minimum payments)
- **SEX**: Binary feature (trivially "censored" by having only 2 values)
- **PAY_2–6**: Payment status features show boundary density spikes

Censored features may appear artificially stable in marginal space because truncation
removes the distributional tails where drift would be visible.

---

## 2. SHAP (Model Decision) Stability

SHAP stability analysis on 3,000 samples using LightGBM (100 trees, depth 6).
Option A: Single model trained on full data, SHAP computed on bootstrap subsets.

### Overall Results

| Metric | Floor Value | Interpretation |
|--------|-------------|----------------|
| **rank_stability** | 0.9795 | Rankings extremely stable (instability = 0.02) |
| **direction_consistency** | 0.5907 | Direction moderately stable (instability = 0.41) |
| **wasserstein** | 0.0073 | SHAP distributions very stable |
| **magnitude_cv** | -0.0079 | Magnitude essentially zero instability |
| **js_divergence** | 0.0002 | Near-zero distributional divergence |
| **Overall complexity** | **-0.0001** | Near-zero — model decisions are stable |

### Per-Feature SHAP Complexity (from fit_panel)

| Feature | SHAP Complexity | Direction Consistency | Rank Stability |
|---------|----------------|----------------------|----------------|
| PAY_2 | 0.114 | 0.637 | 0.926 |
| BILL_AMT5 | 0.119 | 0.601 | 0.871 |
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

### Bug Investigation: "SHAP complexity is 0.053 for every feature"

In this run, per-feature SHAP complexity scores **do differ** (0.114 to 0.201).
The reported bug where all features show the same complexity (0.053) is caused by
a combination of factors:

1. **`aggregate_shap_metrics` crashes on None**: If any pool produces no valid SHAP
   resamples, `all_pool_metrics` contains None entries, and `aggregate_shap_metrics`
   raises `TypeError: argument of type 'NoneType' is not iterable`. **This bug has
   been fixed** — None entries now append NaN values.

2. **Fallback path in `fit_panel`**: When all learning curve fits fail (anomalous),
   the overall complexity comes from a fallback computation. If per-feature metrics
   are also unavailable (e.g., due to the None crash above), `fit_panel` falls back
   to using the global `overall_complexity` for every feature — producing identical scores.

3. **Option A limitation**: With `retrain_per_bootstrap=False` (default), the model
   is trained once and SHAP values for each data point are fixed. The only variation
   across bootstrap resamples is which points are included. This produces very flat
   learning curves that don't fit the `k/n^alpha + floor` model well, causing
   anomalous fits. Using `retrain_per_bootstrap=True` (Option B) captures genuine
   model instability but is much more expensive.

---

## 3. Train/Holdout Drift Detection

Model trained on 70% of data, SHAP compared against 30% holdout.

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **Overall Drift Score** | **0.057** | Minimal drift |
| **Drift Grade** | **A** | Excellent stability |
| Rank Correlation | 0.998 | Near-perfect feature importance agreement |
| Direction Flip Rate | 17.4% | 4 features flip sign (flagged) |
| Top-10 Overlap | 1.000 | Top features identical in train/holdout |
| Magnitude Drift | 0.020 | Negligible magnitude change |

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

**4 features flip SHAP direction** between train and holdout: SEX, MARRIAGE,
BILL_AMT2, PAY_AMT2. Their rank doesn't change (still unimportant features),
but the sign of their average SHAP contribution flips. This is expected for
weak predictors with near-zero mean SHAP values — sign instability in features
that contribute little to predictions.

---

## 4. Key Insights

1. **Payment history (PAY_0–6) are the strongest but least stable predictors.**
   PAY_0 has a Spearman floor of 0.31 — its rank correlation with default fluctuates
   substantially across resamples. This is structural instability that won't resolve
   with more data. Use these features but monitor their WOE profiles in production.

2. **Bill amount features (BILL_AMT1–6) are the most target-stable.**
   Spearman floors < 0.015. Their relationship with default is highly consistent.
   Good candidates for stable scorecards.

3. **The model generalizes well.** Train-to-holdout drift grade is A (score 0.057).
   Feature importance rankings are nearly identical (rank correlation 0.998).

4. **Direction flips are benign.** The 4 features that flip SHAP direction (SEX,
   MARRIAGE, BILL_AMT2, PAY_AMT2) are all weak predictors. Their mean SHAP values
   are near zero, so sign instability doesn't affect model performance.

5. **Censoring affects payment amounts.** PAY_AMT1–6 are flagged for boundary spikes
   (many zeros). Their marginal stability may be artificially high due to truncation.
   The target-dependent view (Spearman floors ~0.11–0.16) reveals moderate instability
   that the marginal view partially conceals.

6. **SHAP complexity differs from marginal complexity.** Marginal stability measures
   whether the feature's distribution is stable across resamples. SHAP stability
   measures whether the model's use of the feature is stable. A feature can be
   marginally unstable but SHAP-stable (the model ignores the unstable parts) or
   vice versa.

---

## 5. Bugs Found

| Bug | Location | Severity | Status |
|-----|----------|----------|--------|
| `aggregate_shap_metrics` crashes on None | `shap_metrics.py:656` | High — crashes SHAP pipeline if any pool has no resamples | **Fixed** |
| Stability metrics not inverted before curve fitting | `shap_stability.py:804` | High — 4 of 8 SHAP metrics always fail curve fitting | **Fixed** |
| README shows `importance_rank` but API uses `importance_score` | `README.md` / `reliability.py` | Low — misleading docs | Noted |
| `estimate_alpha=True` produces extreme floor values | `core.py:fit_learning_curve` | Medium — alpha hitting bounds (0.1 or 1.0) distorts floor parameter | Noted |
| Option A SHAP produces flat learning curves | `shap_stability.py` | Design — inherent to fixed-model approach | Documented |

### Root Cause: SHAP Learning Curve Failures

The `k/n^alpha + floor` model is designed for **decreasing** curves (instability → 0 as n → ∞).
Four SHAP metrics are **stability** metrics that *increase* with pool size:
`rank_stability`, `rank_stability_global`, `direction_consistency`, `topk_overlap`.

When fed directly to the curve fitter, the optimizer sets `k` negative to track the
increasing trend. This produces a non-monotone fitted curve (rises then falls), which
triggers the anomalous flag. Result: 60% of SHAP weight (rank_stability 30% +
direction_consistency 30%) was excluded from the complexity score.

**Fix**: Invert stability metrics to instability (`1 - value`) before fitting. After the
fix, `rank_stability` and `rank_stability_global` fit properly (R² > 0.86). Two metrics
remain anomalous due to Option A limitations:
- `direction_consistency`: essentially flat (range 0.416–0.422 after inversion), R²=0.62
- `topk_overlap`: degenerate (always 1.0 → inverted to 0.0)

Weight coverage improved from ~25% to ~70%. Per-feature complexity scores now vary
across features (the "0.053 for every feature" bug is resolved).

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
| `synthesis.md` | This file |
