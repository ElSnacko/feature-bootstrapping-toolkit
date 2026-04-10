# bootstrap_stability

Feature stability analysis for credit risk modeling using bootstrap learning curves.

Instead of computing a single IV or correlation on the full development sample, this library treats **feature stability as a learning curve problem**. It varies pool size (not resample fraction), runs bootstrap resamples at each pool size, and fits `k/n^alpha + floor` to the resulting instability curve.

The **floor** parameter is the key output: it separates structural instability (won't resolve with more data) from volume instability (will resolve with more data). A high floor means the feature cannot stabilize within your observed data — a strong signal it won't behave in deployment.

This is a diagnostic tool, not a pass/fail gate.

---

## What's New in v2.0

### Major Additions

| Feature | Description | Section |
|---------|-------------|---------|
| **Meta-Bootstrap Confidence Intervals** | Cross-sample stability validation with 95% CIs | [Meta-Bootstrap](#meta-bootstrap-for-confidence-intervals) |
| **Flexible Alpha Parameter** | Fit `alpha` from data to validate CLT assumptions | [Flexible Alpha](#flexible-alpha-parameter) |
| **Target-Agnostic/Dependent Separation** | Separate complexity scores for credible positioning | [Complexity Score Categories](#complexity-score-categories) |
| **Synthetic Validation Suite** | Ground truth testing with known instabilities | [Synthetic Validation](#synthetic-validation) |
| **Permutation Baseline** | Null-calibrated significance testing for complexity scores | [Permutation Baseline](#permutation-baseline) |
| **Reliability Scoring** | Documented formula combining stability, importance, coverage, consistency | [Reliability Scorer](#reliability-scorer) |
| **Fixed SHAP Stability** | End-to-end SHAP complexity with proper fallbacks | [SHAP Stability Analysis](#shap-stability-analysis) |

### Key Improvements

1. **Confidence Intervals**: Complexity scores now report uncertainty via meta-bootstrap
2. **Assumption Validation**: `estimate_alpha=True` checks if convergence follows CLT (α ≈ 0.5)
3. **Credible Positioning**: Separate scores for distributional vs. target-dependent metrics
4. **Ground Truth Testing**: Validate detection capabilities with synthetic data
5. **Explicit Reliability Formula**: Documented weighted components for auditability
6. **Permutation Baseline**: Null-calibrated significance testing with parallelized permutation loop

---

## Features

### Core Capabilities

- **Bootstrap Learning Curves**: Fit `k/n^alpha + floor` to instability vs. sample size
- **Dual Stability Perspectives**: Marginal (distributional) and SHAP (model decision) stability
- **Multi-Metric Analysis**: Wasserstein, KS, JS divergence, Spearman, IV, Monotonicity
- **Panel Analysis**: Rank features by complexity score across your entire portfolio
- **WOE Stability Tracking**: Per-bin sign flip rates and WOE variance
- **Censoring Detection**: Flags policy-truncated features that may appear artificially stable

### Advanced Features

- **Meta-Bootstrap**: K-fold, repeated random, and bootstrap split strategies for confidence intervals
- **Flexible Alpha**: Validate convergence rate assumptions from data
- **Target-Agnostic/Dependent Separation**: Clear separation for credible positioning
- **Synthetic Validation**: Four instability types with permutation-calibrated detection
- **Permutation Baseline**: Null-calibrated significance testing for complexity scores
- **Reliability Scoring**: Weighted combination of stability, importance, coverage, consistency
- **Train vs Holdout Drift Detection**: Grade-based drift assessment (A-F scale)

---

## Installation

```bash
pip install numpy scipy pandas matplotlib joblib scikit-learn
```

For SHAP stability analysis:

```bash
pip install lightgbm shap
```

Clone the repo and import directly — no package installation required:

```bash
git clone https://github.com/ElSnacko/feature-bootstrapping-toolkit.git
cd feature-bootstrapping-toolkit
python demo.py
```

---

## Quick Start

### Basic Usage

```python
import pandas as pd
from bootstrap_stability import BootstrapStability, plot_results, print_report

df = pd.read_csv("your_data.csv")

bs = BootstrapStability(n_resamples=20, random_state=42)
results = bs.fit(df, feature_col="debt_to_income", target_col="default_flag")

print_report(results)
fig = plot_results(results, save_path="dti_stability.png")
```

### With Confidence Intervals (Meta-Bootstrap)

```python
from bootstrap_stability import MetaBootstrap, SplitStrategy

meta = MetaBootstrap(
    n_splits=10,
    strategy=SplitStrategy.KFOLD,
    random_state=42
)

# Get complexity score with 95% confidence interval
meta_results = meta.fit(df, feature_col="debt_to_income", target_col="default_flag")

print(f"Complexity: {meta_results.mean_complexity:.4f}")
print(f"95% CI: [{meta_results.ci_lower:.4f}, {meta_results.ci_upper:.4f}]")
```

### With Flexible Alpha (Validate Convergence)

```python
# Check if convergence follows CLT (alpha ≈ 0.5)
bs = BootstrapStability(
    n_resamples=20,
    estimate_alpha=True,  # Fit alpha from data
    random_state=42
)
results = bs.fit(df, feature_col="income", target_col="default_flag")

# Access fitted alpha per metric
for metric, fit in results["learning_curves"].items():
    if fit and "alpha" in fit:
        print(f"{metric}: alpha={fit['alpha']:.3f} (CLT expects ~0.5)")
```

### Accessing Target-Agnostic vs Target-Dependent Scores

```python
from bootstrap_stability import get_complexity_score

# Get overall score (backwards compatible)
overall = get_complexity_score(results, category="overall")

# Get target-agnostic score (distributional only)
agnostic = get_complexity_score(results, category="target_agnostic")

# Get target-dependent score (relationship with target)
dependent = get_complexity_score(results, category="target_dependent")

print(f"Overall: {overall:.4f}, Agnostic: {agnostic:.4f}, Dependent: {dependent:.4f}")
```

### Reliability Scoring

```python
from bootstrap_stability import ReliabilityScorer, ReliabilityConfig

# Configure weights (must sum to 1.0)
config = ReliabilityConfig(
    stability_weight=0.40,    # Complexity score contribution
    importance_weight=0.30,   # Feature importance (e.g., SHAP rank)
    coverage_weight=0.15,     # Non-NaN metric ratio
    consistency_weight=0.15,  # Cross-seed standard deviation
)

scorer = ReliabilityScorer(config)
reliability = scorer.compute(
    complexity_score=0.05,
    importance_rank=3,
    coverage_ratio=0.95,
    cross_seed_std=0.02
)

print(f"Reliability Score: {reliability.overall_score:.3f}")
print(f"Grade: {reliability.grade}")  # A, B, C, D, F
```

### Synthetic Validation

```python
from bootstrap_stability import SyntheticValidation, InstabilityType

validator = SyntheticValidation(random_state=42)

# Generate data with known instabilities
X, y, metadata = validator.generate_test_data(
    n_samples=2000,
    n_features=10,
    instability_type=InstabilityType.HETEROSCEDASTIC,
    n_corrupted=3,
)

# Permutation-calibrated detection (default)
result = validator.run_test(X, y, metadata, threshold=0.05)

print(f"Detection Rate: {result.detection_rate:.1%}")
print(f"False Positive Rate: {result.false_positive_rate:.1%}")
print(f"Detection Method: {result.detection_method}")
```

---

## How it works

### The learning curve approach

Traditional stability checks run a metric once on the full sample. That misses the question that actually matters in production: **will this feature behave the same way when the model is trained on a different subset?**

This library answers it by constructing an instability curve:

1. Generate a sequence of pool sizes from `min_pool` up to `n` (linear spacing at small n, log spacing at large n where the curve changes fastest)
2. At each pool size, draw multiple bootstrap resamples (with replacement, fixed at 80% of pool size)
3. Compute distributional and target-dependent metrics on each resample vs the full-sample reference
4. Fit `k/n^alpha + floor` to the resulting means across pool sizes

The two parameters tell you different things:
- **k** — how fast instability decays with more data (volume instability). Large k means you just need more observations.
- **floor** — the irreducible instability that remains even at large n (structural instability). A positive floor means something in the feature itself is unstable, not just the sample size.
- **alpha** — the convergence rate exponent. CLT predicts α ≈ 0.5; significant deviations indicate non-standard convergence.

### Complexity score

A single summary score: weighted average of floor parameters across all metrics with valid fits. Lower is better. Negative scores indicate the feature's instability converges cleanly to zero.

Features with high scores warrant investigation before using them in production.

### Complexity Score Categories

The `complexity_scores` dict provides separated scores for credible positioning:

| Category | Metrics Included | Use Case |
|----------|------------------|----------|
| `overall` | All metrics | Backwards compatible summary |
| `target_agnostic` | Wasserstein, KS, JS divergence | Distributional stability only |
| `target_dependent` | Spearman, IV, Monotonicity | Relationship with target |

Access via `get_complexity_score(results, category)` or `results["complexity_scores"][category]`.

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

## API Reference

### Core Classes

#### `BootstrapStability`

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
    estimate_alpha=False,   # NEW: Fit alpha from data
    alpha_bounds=(0.1, 1.0), # NEW: Bounds for alpha estimation
)
```

#### `.fit(df, feature_col, target_col=None) -> dict`

Analyze a single feature. Returns a results dict with learning curves, fitted parameters, WOE profiles, percentile stability, and metadata.

```python
results = bs.fit(df, "ltv_ratio", target_col="charged_off")
```

#### `.fit_panel(df, target_col=None, feature_cols=None) -> dict`

Analyze multiple features. Defaults to all numeric columns. Returns per-feature results and a summary DataFrame sorted by complexity score.

```python
panel = bs.fit_panel(df, target_col="default_flag")
print(panel["summary"][["feature", "complexity_score", "censoring_flag"]])
```

### Meta-Bootstrap

#### `MetaBootstrap`

```python
from bootstrap_stability import MetaBootstrap, SplitStrategy

MetaBootstrap(
    n_splits=10,                    # Number of data splits
    strategy=SplitStrategy.KFOLD,   # KFOLD, REPEATED_RANDOM, or BOOTSTRAP
    base_analyzer=None,             # Optional custom BootstrapStability instance
    n_jobs=-1,                      # Parallel computation
    random_state=42,
)
```

#### `.fit(df, feature_col, target_col=None) -> MetaBootstrapResult`

Returns a `MetaBootstrapResult` with:
- `mean_complexity`: Mean across splits
- `std_complexity`: Standard deviation
- `ci_lower`, `ci_upper`: 95% confidence interval bounds
- `complexity_scores_by_category`: Per-category statistics

```python
meta = MetaBootstrap(n_splits=10, strategy=SplitStrategy.KFOLD)
result = meta.fit(df, "income", target_col="default_flag")

print(f"Complexity: {result.mean_complexity:.4f} ± {result.std_complexity:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")
```

### Reliability Scorer

#### `ReliabilityScorer`

```python
from bootstrap_stability import ReliabilityScorer, ReliabilityConfig

config = ReliabilityConfig(
    stability_weight=0.40,      # Weight for complexity score (inverted)
    importance_weight=0.30,     # Weight for feature importance
    coverage_weight=0.15,       # Weight for metric coverage
    consistency_weight=0.15,    # Weight for cross-seed consistency
    normalization_method="minmax",  # "minmax", "rank", or "zscore"
)

scorer = ReliabilityScorer(config)
result = scorer.compute(
    complexity_score=0.05,
    importance_rank=3,
    coverage_ratio=0.95,
    cross_seed_std=0.02
)
```

**Reliability Formula:**

```
reliability = w_stability * (1 - normalized_complexity)
            + w_importance * normalized_importance
            + w_coverage * coverage_ratio
            + w_consistency * (1 - normalized_std)
```

Default weights: stability (40%), importance (30%), coverage (15%), consistency (15%)

### Synthetic Validation

#### `SyntheticValidation`

```python
from bootstrap_stability import SyntheticValidation, InstabilityType

validator = SyntheticValidation(random_state=42)

# Generate data, then test with permutation-calibrated detection
X, y, metadata = validator.generate_test_data(
    n_samples=2000, n_features=10,
    instability_type=InstabilityType.DISTRIBUTION_SHIFT,
    n_corrupted=3,
)
result = validator.run_test(
    X, y, metadata,
    threshold=0.05,          # Significance level (alpha) for permutation test
    use_permutation=True,    # Default: permutation-calibrated detection
    n_permutations=30,       # Null distribution size
    n_jobs=-1,               # Parallel permutation runs
)
```

**Instability Types:**

Each type injects an influential minority (20% of samples) with target-aligned extreme values that create **structural instability** — irreducible metric variance that persists at any sample size.

| Type | Mechanism | Structural Signal |
|------|-----------|-------------------|
| `HETEROSCEDASTIC` | Target-aligned extreme values | Spearman/IV swing with influential point inclusion |
| `DISTRIBUTION_SHIFT` | Larger magnitude influential minority | Heavier-tailed structural instability |
| `INTERACTION` | Partner-feature modulated magnitude | Instability clusters in partner-dependent regions |
| `MISSING_NOT_AT_RANDOM` | Influential minority + target-dependent missingness | Combined leverage point and data loss instability |

**Detection Methods:**

| Method | `use_permutation` | How it works |
|--------|-------------------|--------------|
| Permutation-calibrated (default) | `True` | Per-feature null via `PermutationBaseline`; flags features where p < threshold |
| Raw threshold (legacy) | `False` | Flags features where complexity score >= threshold |

### Permutation Baseline

#### `PermutationBaseline`

Builds a null distribution of complexity scores by shuffling the feature-target relationship. Determines whether a feature's observed instability is significantly above what noise alone produces.

```python
from bootstrap_stability import PermutationBaseline

perm = PermutationBaseline(
    n_permutations=30,       # Number of null permutations
    alpha=0.05,              # Significance level
    random_state=42,
    n_jobs=-1,               # Parallel permutation runs (loky processes)
)

result = perm.fit(df, feature_col="income", target_col="default_flag")

print(f"Observed: {result['observed']:.4f}")
print(f"Null mean: {result['null_mean']:.4f}")
print(f"p-value: {result['p_value']:.3f}")
print(f"Significant: {result['significant']}")

# Panel mode: test all features
panel = perm.fit_panel(df, target_col="default_flag")
sig_features = panel["summary"][panel["summary"]["significant"]]
print(f"{len(sig_features)} features significantly above null")
```

Permutations run in parallel using loky processes. Each permutation operates on a slim 2-column dataframe (feature + target only) to minimize memory overhead.

### Output Functions

#### `print_report(results)`

Prints a structured terminal report: metadata, complexity score, metric table with floors and R², extrapolations, WOE bin table with stability status.

#### `plot_results(results, save_path=None, figsize=(15, 11), dpi=150) -> Figure`

Four-panel figure with learning curves, floor decomposition, WOE profile stability, and percentile stability.

#### `plot_panel(panel_results, save_path=None, top_n=30) -> Figure`

Horizontal bar chart of complexity scores across all features.

#### `to_csv(results, save_path) -> pd.DataFrame`

One row per pool size with metadata as commented header lines.

#### `panel_to_csv(panel_results, save_path) -> pd.DataFrame`

Writes the panel summary DataFrame directly.

### Utility Functions

#### `get_complexity_score(results, category) -> float`

Extract complexity score by category from results.

```python
from bootstrap_stability import get_complexity_score

overall = get_complexity_score(results, "overall")
agnostic = get_complexity_score(results, "target_agnostic")
dependent = get_complexity_score(results, "target_dependent")
```

#### `get_metric_category(metric_name) -> str`

Returns `"target_agnostic"` or `"target_dependent"` for a given metric name.

---

## SHAP Stability Analysis

### The Problem: Marginal Space Blind Spot

The default stability metrics measure stability in **marginal (univariate) space**—whether feature distributions look the same across bootstrap resamples. But models operate in a **nonlinear, interactive feature space** where:

- A feature with stable marginal distribution can have unstable contributions to predictions
- A feature with unstable marginal distribution can have stable model contributions
- Marginal stability ≠ Model decision stability

**Key finding:** In comparison analysis, 43.5% of features showed disagreement between marginal and SHAP stability:
- 34.8% false alarms (marginal over-estimates risk)
- 8.7% missed risks (marginal under-estimates risk)

### The Solution: SHAP-Based Stability

SHAP (SHapley Additive exPlanations) values measure how much each feature contributes to individual predictions. SHAP stability measures whether these contributions stabilize—directly measuring what the model "cares about."

**When to use SHAP stability:**
- Before production deployment to validate feature behavior
- When features have complex interactions or non-linear relationships
- For train vs holdout drift detection
- When marginal stability gives unexpected results

### Quick Example

```python
from bootstrap_stability import SHAPStability, TrainHoldoutStability, print_holdout_report

# Define model factory
def lgbm_factory():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(n_estimators=100, max_depth=6, random_state=42, verbose=-1)

# SHAP-based stability with learning curves (same k/n^alpha + floor model)
shap_stability = SHAPStability(model_factory=lgbm_factory)
results = shap_stability.fit(X_train, y_train, pool_sizes, n_resamples=25)

# Train vs holdout comparison for production monitoring
holdout_checker = TrainHoldoutStability(model_factory=lgbm_factory)
drift_results = holdout_checker.compare(X_train, y_train, X_holdout, y_holdout)
print_holdout_report(drift_results)
```

### SHAP Stability Metrics

| Metric | Description | Interpretation |
|--------|-------------|----------------|
| Rank Stability | Kendall's W coefficient for feature importance rankings | `> 0.9` = stable importance |
| Wasserstein | Distribution shift in SHAP space | `< 0.02` = low drift |
| Direction Consistency | Sign stability across resamples | `> 0.95` = consistent direction |
| Magnitude CV | Coefficient of variation of \|SHAP\| | `< 0.10` = stable magnitude |
| Top-k Overlap | Jaccard overlap of top-k features | `> 0.7` = stable top features |

### Train vs Holdout Drift Detection

The `TrainHoldoutStability` class compares SHAP patterns between training and holdout sets:

```python
from bootstrap_stability import TrainHoldoutStability, print_holdout_report

holdout_checker = TrainHoldoutStability(model_factory=lgbm_factory)
drift_results = holdout_checker.compare(X_train, y_train, X_holdout, y_holdout)

# Access drift metrics
print(f"Rank correlation: {drift_results['drift_metrics']['rank_correlation']:.3f}")
print(f"Direction flip rate: {drift_results['drift_metrics']['direction_flip_rate']:.1%}")
print(f"Overall drift grade: {drift_results['drift_grade']}")  # A, B, C, D, F
```

**Drift grades:**

| Grade | Score Range | Interpretation |
|-------|-------------|----------------|
| A | 0.00 - 0.10 | Minimal drift—feature behavior stable |
| B | 0.10 - 0.25 | Moderate drift—monitor closely |
| C | 0.25 - 0.40 | Significant drift—investigate |
| D | 0.40 - 0.60 | Severe drift—consider retraining |
| F | > 0.60 | Critical drift—model may be stale |

---

## Precondition Checks

The library runs two checks before analysis and surfaces the results in `results["meta"]`:

### Imbalance Check

```python
bs = BootstrapStability(
    imbalance_threshold=0.05,  # raises ImbalanceError if minority class < 5%
    allow_imbalance=True,      # downgrade to warning instead
)
```

When imbalance is detected, the minimum event threshold is automatically scaled up (1.5x moderate, 2x severe, 3x critical) to ensure reliable WOE/IV estimates.

### Truncation / Censoring Check

Detects policy-censored features: hard boundary walls, and density spikes at the outer edges of the distribution (which indicate the feature was clipped by upstream rules). A feature that looks stable may only be stable because it was truncated. The flag is surfaced in the report and shown in red in panel charts.

---

## Examples

### Example 1 — Single Feature Deep Dive

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

### Example 2 — No-Target Mode (Distributional Only)

Useful when you want to check whether a feature's distribution is stable across time periods or population segments, independent of any outcome.

```python
# Does this feature look the same in holdout as in development?
holdout_results = bs.fit(holdout_df, "bureau_score")
dev_results = bs.fit(dev_df, "bureau_score")

print(f"Dev floor:  {dev_results['per_metric_floors']['wasserstein']:.4f}")
print(f"holdout floor:  {holdout_results['per_metric_floors']['wasserstein']:.4f}")
```

### Example 3 — Panel Analysis and Feature Selection

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

### Example 4 — Handling Imbalanced Targets

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

### Example 5 — Accessing Raw Results

```python
results = bs.fit(df, "income", target_col="default_flag")

# Fitted curve parameters
fit = results["learning_curves"]["wasserstein"]["fit"]
print(f"k={fit['k']:.4f}, floor={fit['floor']:.4f}, R²={fit['r2']:.3f}")

# Alpha parameter (when estimate_alpha=True)
if "alpha" in fit:
    print(f"alpha={fit['alpha']:.3f}")

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

### Example 6 — Meta-Bootstrap for Confidence Intervals

```python
from bootstrap_stability import MetaBootstrap, SplitStrategy

# Use k-fold splitting for confidence intervals
meta = MetaBootstrap(
    n_splits=10,
    strategy=SplitStrategy.KFOLD,
    random_state=42
)

result = meta.fit(df, "income", target_col="default_flag")

print(f"Mean Complexity: {result.mean_complexity:.4f}")
print(f"Std Dev: {result.std_complexity:.4f}")
print(f"95% CI: [{result.ci_lower:.4f}, {result.ci_upper:.4f}]")

# Check if score is stable across splits
if result.std_complexity > 0.05:
    print("WARNING: Complexity score varies significantly across splits")
```

### Example 7 — Validating Convergence Assumptions

```python
# Enable alpha estimation to check CLT assumption
bs = BootstrapStability(
    n_resamples=30,
    estimate_alpha=True,
    alpha_bounds=(0.3, 0.8),  # Reasonable bounds
    random_state=42
)

results = bs.fit(df, "feature_x", target_col="target")

# Check alpha values
for metric, curve in results["learning_curves"].items():
    fit = curve.get("fit", {})
    alpha = fit.get("alpha")
    if alpha:
        status = "✓ CLT valid" if 0.4 <= alpha <= 0.6 else "⚠ Non-standard convergence"
        print(f"{metric}: α={alpha:.3f} {status}")
```

### Example 8 — Full Reliability Assessment

```python
from bootstrap_stability import (
    BootstrapStability, MetaBootstrap,
    ReliabilityScorer, ReliabilityConfig,
    get_complexity_score
)

# Step 1: Get complexity score with confidence interval
meta = MetaBootstrap(n_splits=5, random_state=42)
meta_result = meta.fit(df, "feature_x", target_col="target")

# Step 2: Compute reliability score
config = ReliabilityConfig(
    stability_weight=0.40,
    importance_weight=0.30,
    coverage_weight=0.15,
    consistency_weight=0.15,
)

scorer = ReliabilityScorer(config)
reliability = scorer.compute(
    complexity_score=meta_result.mean_complexity,
    importance_rank=5,  # From your feature importance analysis
    coverage_ratio=0.92,  # Non-NaN metric ratio
    cross_seed_std=meta_result.std_complexity
)

print(f"Reliability Score: {reliability.overall_score:.3f}")
print(f"Reliability Grade: {reliability.grade}")
print(f"Component Breakdown:")
print(f"  Stability: {reliability.stability_component:.3f}")
print(f"  Importance: {reliability.importance_component:.3f}")
print(f"  Coverage: {reliability.coverage_component:.3f}")
print(f"  Consistency: {reliability.consistency_component:.3f}")
```

---

## Interpreting Results

### Complexity Score

| Score | Interpretation |
|---|---|
| Negative | Feature stabilizes cleanly — structural floor is effectively zero |
| ~0 | On the boundary — converges to stability but has no margin |
| Positive, small (< 0.05) | Mild structural instability — worth monitoring |
| Positive, large (> 0.10) | Significant structural instability — investigate before production use |

### Anomalous Fits

An anomalous fit (low R² or non-monotone fitted curve) means the `k/n^alpha` model doesn't describe the instability pattern well. For Spearman/IV/Monotonicity on small datasets this is expected — those metrics converge quickly and there's no meaningful decay to fit. Anomalous fits are excluded from the complexity score.

### WOE Bin Status

| Status | Condition |
|---|---|
| stable | SD < 0.15 and sign flip rate < 10% |
| noisy | Between stable and unstable |
| unstable | Sign flip rate > 30% — bin polarity reverses frequently across resamples |

### Censoring Warning

A censored feature may appear artificially stable because policy truncation removes the tails where distributional drift would show up. Treat censoring-flagged features with caution even if their complexity score is low.

### Alpha Parameter

| Alpha Value | Interpretation |
|-------------|----------------|
| ≈ 0.5 | Standard CLT convergence (expected) |
| < 0.4 | Faster than expected convergence |
| > 0.6 | Slower convergence; may indicate heavy tails or dependencies |

---

## Running the Demo

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

## Documentation

### Detailed Documentation Files

| Document | Description |
|----------|-------------|
| [`docs/shap_stability_design.md`](docs/shap_stability_design.md) | SHAP stability metrics, API reference, integration patterns |
| [`docs/reliability_score.md`](docs/reliability_score.md) | Reliability formula, component definitions, configuration |
| [`docs/architecture_improvements.md`](docs/architecture_improvements.md) | System architecture, design decisions, extension points |

---

## Design Notes

**Pool size varies, resample fraction is fixed.** Varying the fraction causes resamples to overlap heavily at high fractions, understating true instability. This is the core architectural choice.

**WOE bins are recomputed per resample.** Fixed bins anchor WOE to the full-sample distribution and suppress variance artificially. Per-resample bins are the honest measure.

**Bandwidth is fixed from the full dataset.** Computing a fresh bandwidth per resample would conflate KDE parameter instability with feature instability.

**The floor, not the curve, is the signal.** A steep curve with a near-zero floor means the feature is fine — it just needs volume. A shallow curve with a high positive floor means more data won't help.

**SHAP stability uses the same learning curve model.** The `k/n^alpha + floor` approach applies to SHAP metrics just as it does to marginal metrics, with floor representing irreducible decision-space instability.

**Alpha is estimable from data.** Setting `estimate_alpha=True` allows the convergence rate to be fit from the data, validating whether the CLT assumption (α ≈ 0.5) holds.

**Complexity scores are separable.** The `complexity_scores` dict with 'overall', 'target_agnostic', and 'target_dependent' keys enables credible positioning based on what aspects of stability matter most for your use case.

**Parallelism is single-level.** MetaBootstrap and PermutationBaseline run splits/permutations sequentially at the outer level with parallel pool computation at the inner level. This avoids memory multiplication from nested parallelism that causes OOM kills on constrained systems.

**Permutation null calibrates "unstable".** Without a null distribution, a complexity score of 0.003 vs 0.027 is hard to interpret. The permutation baseline shuffles the target to produce a null, then tests whether the observed score is significantly above it.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
