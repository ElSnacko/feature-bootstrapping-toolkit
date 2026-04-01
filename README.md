# bootstrap_stability

Feature stability analysis for credit risk modeling using bootstrap learning curves.

Instead of computing a single IV or correlation on the full development sample, this library treats **feature stability as a learning curve problem**. It varies pool size (not resample fraction), runs bootstrap resamples at each pool size, and fits `k/sqrt(n) + floor` to the resulting instability curve.

The **floor** parameter is the key output: it separates structural instability (won't resolve with more data) from volume instability (will resolve with more data). A high floor means the feature cannot stabilize within your observed data — a strong signal it won't behave in deployment.

This is a diagnostic tool, not a pass/fail gate.

---

## Installation

```bash
pip install numpy scipy pandas matplotlib joblib scikit-learn
```

Clone the repo and import directly — no package installation required:

```bash
git clone https://github.com/ElSnacko/feature-bootstrapping-toolkit.git
cd feature-bootstrapping-toolkit
python demo.py
```

---

## Quick start

```python
import pandas as pd
from bootstrap_stability import BootstrapStability, plot_results, print_report

df = pd.read_csv("your_data.csv")

bs = BootstrapStability(n_resamples=20, random_state=42)
results = bs.fit(df, feature_col="debt_to_income", target_col="default_flag")

print_report(results)
fig = plot_results(results, save_path="dti_stability.png")
```

---

## How it works

### The learning curve approach

Traditional stability checks run a metric once on the full sample. That misses the question that actually matters in production: **will this feature behave the same way when the model is trained on a different subset?**

This library answers it by constructing an instability curve:

1. Generate a sequence of pool sizes from `min_pool` up to `n` (linear spacing at small n, log spacing at large n where the curve changes fastest)
2. At each pool size, draw multiple bootstrap resamples (with replacement, fixed at 80% of pool size)
3. Compute distributional and target-dependent metrics on each resample vs the full-sample reference
4. Fit `k/sqrt(n) + floor` to the resulting means across pool sizes

The two parameters tell you different things:
- **k** — how fast instability decays with more data (volume instability). Large k means you just need more observations.
- **floor** — the irreducible instability that remains even at large n (structural instability). A positive floor means something in the feature itself is unstable, not just the sample size.

### Complexity score

A single summary score: weighted average of floor parameters across all metrics with valid fits. Lower is better. Negative scores indicate the feature's instability converges cleanly to zero.

Features with high scores warrant investigation before using them in production.

---

## Metrics

### Distributional (always computed)

| Metric | Description |
|---|---|
| Wasserstein | Earth mover's distance between reference KDE and resample KDE |
| KS | Kolmogorov-Smirnov statistic between reference and resample |
| JS divergence | Jensen-Shannon divergence between reference and resample KDEs |

### Target-dependent (when `target_col` is provided)

| Metric | Description |
|---|---|
| Spearman ρ | Rank correlation between feature and target per resample |
| IV | Information value via quantile-binned WOE (bins recomputed per resample) |
| Monotonicity | Rate of non-monotone WOE profiles across resamples |

> **Why bins are recomputed per resample:** Fixed bins anchor WOE to the full-sample distribution and suppress variance artificially. Per-resample bins are the honest measure — they reflect what actually happens when a model is trained on a subset.

---

## API reference

### `BootstrapStability`

```python
BootstrapStability(
    resample_frac=0.8,      # Fixed fraction per bootstrap draw — do not vary this
    n_resamples=20,         # Draws per pool size
    n_bins=5,               # Quantile bins for WOE/IV
    min_events=20,          # Minimum events required per pool
    imbalance_threshold=0.05,
    allow_imbalance=False,  # If False, raises ImbalanceError when minority < threshold
    metric_weights=None,    # Defaults to DEFAULT_WEIGHTS
    min_pool=50,
    linear_threshold=1000,
    n_points=25,
    r2_threshold=0.70,      # Below this R², a fit is flagged as anomalous
    extrapolate_to=None,    # Default: [500, 1000]
    store_raw=True,         # Store per-resample values in results dict
    n_jobs=-1,              # Parallel pool computation
    random_state=42,
)
```

#### `.fit(df, feature_col, target_col=None) -> dict`

Analyze a single feature. Returns a results dict with learning curves, fitted parameters, WOE profiles, percentile stability, and metadata.

```python
results = bs.fit(df, "ltv_ratio", target_col="charged_off")
```

Without a target column, only distributional metrics are computed:

```python
results = bs.fit(df, "income_band")  # distributional only
```

#### `.fit_panel(df, target_col=None, feature_cols=None) -> dict`

Analyze multiple features. Defaults to all numeric columns. Returns per-feature results and a summary DataFrame sorted by complexity score.

```python
panel = bs.fit_panel(df, target_col="default_flag")
print(panel["summary"][["feature", "complexity_score", "censoring_flag"]])
```

---

### Output functions

#### `print_report(results)`

Prints a structured terminal report: metadata, complexity score, metric table with floors and R², extrapolations, WOE bin table with stability status.

```
================================================================
Bootstrap Stability Report — 'debt_to_income'
Version: 1.0.0  |  2024-11-14T09:33:12
================================================================
Observations        : 12450
Feature type        : continuous
Event rate          : 0.0842
Imbalance flag      : No
Censoring flag      : No

Complexity score : 0.0183

Metric                            Floor       R²   Anomalous
--------------------------------------------------------------
Wasserstein   [primary]          0.0412    0.951          no
KS            [secondary]        0.0183    0.977          no
JS divergence [secondary]        0.0091    0.944          no
Spearman ρ    [primary]          0.2871    0.031         yes
...
```

#### `plot_results(results, save_path=None, figsize=(15, 11), dpi=150) -> Figure`

Four-panel figure:
1. **Learning curves** — instability vs pool size per metric, with fitted curve overlay and floor markers
2. **Floor decomposition** — horizontal bar chart of floor values (the structural instability per metric)
3. **WOE profile stability** — mean WOE per bin with SD error bars and sign flip rate
4. **Percentile stability** — how feature percentiles shift across pool sizes

```python
fig = plot_results(results, save_path="dti_stability.png")
```

#### `plot_panel(panel_results, save_path=None, top_n=30) -> Figure`

Horizontal bar chart of complexity scores across all features. Red bars indicate censoring detected.

```python
fig = plot_panel(panel, save_path="feature_scores.png")
```

#### `to_csv(results, save_path) -> pd.DataFrame`

One row per pool size. Metadata written as commented header lines. Columns include per-metric means, standard errors, floors, k values, R², anomalous flags, and extrapolations.

#### `panel_to_csv(panel_results, save_path) -> pd.DataFrame`

Writes the panel summary DataFrame directly.

---

## Precondition checks

The library runs two checks before analysis and surfaces the results in `results["meta"]`:

### Imbalance check

```python
bs = BootstrapStability(
    imbalance_threshold=0.05,  # raises ImbalanceError if minority class < 5%
    allow_imbalance=True,      # downgrade to warning instead
)
```

When imbalance is detected, the minimum event threshold is automatically scaled up (1.5x moderate, 2x severe, 3x critical) to ensure reliable WOE/IV estimates.

### Truncation / censoring check

Detects policy-censored features: hard boundary walls, and density spikes at the outer edges of the distribution (which indicate the feature was clipped by upstream rules). A feature that looks stable may only be stable because it was truncated. The flag is surfaced in the report and shown in red in panel charts.

---

## Examples

### Example 1 — single feature deep dive

```python
import pandas as pd
from bootstrap_stability import BootstrapStability, plot_results, print_report, to_csv

df = pd.read_csv("loans.csv")
bs = BootstrapStability(n_resamples=30, random_state=0)

results = bs.fit(df, "debt_to_income", target_col="default_flag")
print_report(results)
fig = plot_results(results, save_path="dti_stability.png")
to_csv(results, "dti_stability.csv")
```

### Example 2 — no-target mode (distributional only)

Useful when you want to check whether a feature's distribution is stable across time periods or population segments, independent of any outcome.

```python
# Does this feature look the same in OOT as in development?
oot_results = bs.fit(oot_df, "bureau_score")
dev_results = bs.fit(dev_df, "bureau_score")

print(f"Dev floor:  {dev_results['per_metric_floors']['wasserstein']:.4f}")
print(f"OOT floor:  {oot_results['per_metric_floors']['wasserstein']:.4f}")
```

### Example 3 — panel analysis and feature selection

```python
from bootstrap_stability import BootstrapStability, plot_panel, panel_to_csv

bs = BootstrapStability(n_resamples=15, n_jobs=-1, random_state=42)
panel = bs.fit_panel(df, target_col="default_flag")

summary = panel["summary"]
print(summary[["feature", "complexity_score", "censoring_flag"]].to_string(index=False))

# Flag structurally unstable features
unstable = summary[summary["complexity_score"] > 0.05]
print(f"\n{len(unstable)} features with complexity > 0.05:")
print(unstable["feature"].tolist())

fig = plot_panel(panel, save_path="panel_scores.png")
panel_to_csv(panel, "panel_scores.csv")
```

### Example 4 — handling imbalanced targets

```python
from bootstrap_stability import BootstrapStability, ImbalanceError

bs = BootstrapStability(
    allow_imbalance=True,   # warn instead of raise
    min_events=10,          # lower absolute threshold for rare events
    n_resamples=40,         # more resamples to compensate for noisy rare-event draws
)

results = bs.fit(df, "payment_velocity", target_col="fraud_flag")
# results["meta"]["imbalance_severity"] will show "severe" or "critical"
```

### Example 5 — accessing raw results

```python
results = bs.fit(df, "income", target_col="default_flag")

# Fitted curve parameters
fit = results["learning_curves"]["wasserstein"]["fit"]
print(f"k={fit['k']:.4f}, floor={fit['floor']:.4f}, R²={fit['r2']:.3f}")

# WOE bin stability
for bin_name, stats in results["woe_profiles"].items():
    print(f"{bin_name}: mean_woe={stats['mean_woe']:.3f}, flip_rate={stats['sign_flip_rate']:.1%}")

# Extrapolated instability at larger hypothetical sample sizes
extrap = fit["extrapolations"]
print(f"Predicted Wasserstein at n=1000: {extrap.get(1000, 'not computed'):.4f}")

# Raw per-resample values at each pool size (requires store_raw=True)
if results["raw_bootstrap"]:
    raw = results["raw_bootstrap"]  # {pool_size: {metric: [values]}}
```

---

## Interpreting results

### Complexity score

| Score | Interpretation |
|---|---|
| Negative | Feature stabilizes cleanly — structural floor is effectively zero |
| ~0 | On the boundary — converges to stability but has no margin |
| Positive, small (< 0.05) | Mild structural instability — worth monitoring |
| Positive, large (> 0.10) | Significant structural instability — investigate before production use |

### Anomalous fits

An anomalous fit (low R² or non-monotone fitted curve) means the `k/sqrt(n)` model doesn't describe the instability pattern well. For Spearman/IV/Monotonicity on small datasets this is expected — those metrics converge quickly and there's no meaningful decay to fit. Anomalous fits are excluded from the complexity score.

### WOE bin status

| Status | Condition |
|---|---|
| stable | SD < 0.15 and sign flip rate < 10% |
| noisy | Between stable and unstable |
| unstable | Sign flip rate > 30% — bin polarity reverses frequently across resamples |

### Censoring warning

A censored feature may appear artificially stable because policy truncation removes the tails where distributional drift would show up. Treat censoring-flagged features with caution even if their complexity score is low.

---

## Running the demo

```bash
python demo.py
```

Uses the sklearn breast cancer dataset as a credit risk proxy (malignant = event). Runs:
1. Deep dive on `mean radius` — strong, stable predictor
2. Deep dive on `symmetry error` — weak, noisy predictor
3. No-target distributional analysis on `mean radius`
4. Full panel across all 30 features

Expected output: `mean radius` has a lower complexity score than `symmetry error`.

---

## Design notes

**Pool size varies, resample fraction is fixed.** Varying the fraction causes resamples to overlap heavily at high fractions, understating true instability. This is the core architectural choice.

**WOE bins are recomputed per resample.** Fixed bins anchor WOE to the full-sample distribution and suppress variance artificially. Per-resample bins are the honest measure.

**Bandwidth is fixed from the full dataset.** Computing a fresh bandwidth per resample would conflate KDE parameter instability with feature instability.

**The floor, not the curve, is the signal.** A steep curve with a near-zero floor means the feature is fine — it just needs volume. A shallow curve with a high positive floor means more data won't help.