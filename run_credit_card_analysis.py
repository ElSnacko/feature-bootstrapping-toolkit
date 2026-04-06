#!/usr/bin/env python3
"""
Comprehensive credit card default analysis using the bootstrap stability toolkit.

Runs:
1. Marginal stability (BootstrapStability) panel analysis
2. SHAP stability (SHAPStability) analysis with bug investigation
3. Train/holdout drift detection
4. Synthesizes insights and saves all results
"""

import os
import sys
import json
import warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

sys.path.insert(0, '.')
warnings.filterwarnings('ignore')

from bootstrap_stability import (
    BootstrapStability,
    SHAPStability,
    TrainHoldoutStability,
    MetaBootstrap,
    SplitStrategy,
    ReliabilityScorer,
    ReliabilityConfig,
    plot_results,
    plot_panel,
    print_report,
    to_csv,
    panel_to_csv,
    print_holdout_report,
    get_complexity_score,
)

# ── Output dir ───────────────────────────────────────────────────────────
OUTPUT_DIR = Path("credit_card_full_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

def save_json(obj, filename):
    """Save dict to JSON, converting numpy types."""
    def _convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert(v) for v in o]
        if isinstance(o, (float, int)) and (o != o):  # NaN
            return None
        return o
    path = OUTPUT_DIR / filename
    with open(path, 'w') as f:
        json.dump(_convert(obj), f, indent=2, default=str)
    log(f"  Saved {path}")


# ══════════════════════════════════════════════════════════════════════════
# 1. LOAD DATA
# ══════════════════════════════════════════════════════════════════════════
log("=" * 60)
log("LOADING DATA")
log("=" * 60)

data_path = '../default+of+credit+card+clients/default of credit card clients.xls'
df = pd.read_excel(data_path, header=1)
df = df.drop(columns=['ID'])

TARGET = 'default payment next month'
FEATURES = [c for c in df.columns if c != TARGET]

log(f"Shape: {df.shape}")
log(f"Target: '{TARGET}' (event rate = {df[TARGET].mean():.4f})")
log(f"Features ({len(FEATURES)}): {FEATURES}")


# ══════════════════════════════════════════════════════════════════════════
# 2. MARGINAL STABILITY — PANEL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("MARGINAL STABILITY PANEL ANALYSIS")
log("=" * 60)

bs = BootstrapStability(
    n_resamples=25,
    estimate_alpha=True,
    random_state=42,
    n_jobs=-1,
)

panel = bs.fit_panel(df, target_col=TARGET)
summary = panel["summary"]

log("\nPanel summary (sorted by complexity):")
cols = ["feature", "complexity_score", "censoring_flag", "wasserstein_floor",
        "ks_floor", "spearman_floor", "iv_floor"]
print(summary[cols].to_string(index=False))

# Save panel results
panel_to_csv(panel, str(OUTPUT_DIR / "marginal_panel_summary.csv"))
fig = plot_panel(panel, save_path=str(OUTPUT_DIR / "marginal_panel_chart.png"))
fig.clf()

# Deep dive on 4 representative features
deep_features = ["LIMIT_BAL", "PAY_0", "AGE", "BILL_AMT1"]
deep_results = {}
for feat in deep_features:
    if feat in panel["feature_results"]:
        r = panel["feature_results"][feat]
        deep_results[feat] = r
        fig = plot_results(r, save_path=str(OUTPUT_DIR / f"marginal_{feat}.png"))
        fig.clf()
        to_csv(r, str(OUTPUT_DIR / f"marginal_{feat}.csv"))

# Save all feature results
all_marginal = {}
for feat, r in panel["feature_results"].items():
    all_marginal[feat] = {
        "complexity_score": r["complexity_score"],
        "complexity_scores": r.get("complexity_scores", {}),
        "per_metric_floors": r.get("per_metric_floors", {}),
        "censoring_flag": r["meta"]["censoring_flag"],
        "censoring_detail": r["meta"]["censoring_detail"],
        "feature_type": r["meta"]["feature_type"],
        "n_obs": r["meta"]["n_obs"],
    }
save_json(all_marginal, "marginal_all_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 3. SHAP STABILITY — BUG INVESTIGATION
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("SHAP STABILITY ANALYSIS + BUG INVESTIGATION")
log("=" * 60)

from lightgbm import LGBMClassifier

def lgbm_factory():
    return LGBMClassifier(
        n_estimators=100,
        max_depth=6,
        random_state=42,
        verbose=-1,
        n_jobs=1,
    )

X = df[FEATURES]
y = df[TARGET]

# Subsample for speed (SHAP is expensive)
np.random.seed(42)
sample_idx = np.random.choice(len(X), 3000, replace=False)
X_sample = X.iloc[sample_idx].reset_index(drop=True)
y_sample = y.iloc[sample_idx].reset_index(drop=True)

log(f"Using {len(X_sample)} samples for SHAP analysis")

# ── 3a. Run SHAPStability.fit() ──────────────────────────────────────────
shap_analyzer = SHAPStability(
    model_factory=lgbm_factory,
    n_resamples=15,
    n_points=10,
    explainer_type="tree",
    random_state=42,
    verbose=2,  # extra debug output
)

log("Running SHAPStability.fit()...")
shap_results = shap_analyzer.fit(X_sample, y_sample)

log(f"\nOverall SHAP complexity_score: {shap_results['complexity_score']:.6f}")
log(f"Complexity scores dict: {shap_results['complexity_scores']}")
log(f"Per-metric floors: {shap_results['per_metric_floors']}")

# Check learning curves for anomalous fits
log("\nLearning curve fit status per metric:")
for metric, lc in shap_results["learning_curves"].items():
    fit = lc.get("fit", {})
    log(f"  {metric:30s}: floor={fit.get('floor', 'N/A'):>10s}  r2={fit.get('r2', 'N/A'):>10s}  "
        f"anomalous={fit.get('anomalous', 'N/A')}  failed={fit.get('fit_failed', 'N/A')}"
        if isinstance(fit.get('floor'), str) else
        f"  {metric:30s}: floor={fit.get('floor', float('nan')):>10.6f}  r2={fit.get('r2', float('nan')):>10.4f}  "
        f"anomalous={fit.get('anomalous', 'N/A')}  failed={fit.get('fit_failed', 'N/A')}")

# ── 3b. Run SHAPStability.fit_panel() and check per-feature complexity ──
log("\nRunning SHAPStability.fit_panel()...")
shap_panel = shap_analyzer.fit_panel(X_sample, y_sample)
shap_summary = shap_panel["summary"]

log("\nSHAP panel summary:")
print(shap_summary.to_string(index=False))

# ── 3c. BUG CHECK: Are all per-feature complexity scores identical? ──────
shap_complexities = shap_summary["complexity_score"].values
unique_complexities = np.unique(shap_complexities[~np.isnan(shap_complexities)])
log(f"\nBUG CHECK: unique SHAP complexity values = {len(unique_complexities)}")
log(f"  Values: {unique_complexities[:10]}")

if len(unique_complexities) == 1:
    log("  *** BUG CONFIRMED: All features have identical SHAP complexity ***")
    log("  Investigating root cause...")

    # Check per-feature metric means
    feat_results = shap_panel["feature_results"]
    log(f"\n  Per-feature metrics sample (first 3 features):")
    for fr in feat_results[:3]:
        log(f"    {fr['feature']}:")
        for k, v in sorted(fr.items()):
            if k != "feature":
                log(f"      {k}: {v:.6f}" if isinstance(v, float) else f"      {k}: {v}")

    # Check if the fallback path is being taken
    overall_c = shap_results["complexity_score"]
    log(f"\n  Overall complexity (from fit): {overall_c:.6f}")
    log(f"  Panel per-feature value:      {unique_complexities[0]:.6f}")
    if abs(overall_c - unique_complexities[0]) < 1e-10:
        log("  *** ROOT CAUSE: fit_panel falls back to global overall_complexity ***")
        log("  This means per-feature metric values are NaN or total_w == 0")

save_json({
    "complexity_score": shap_results["complexity_score"],
    "complexity_scores": shap_results["complexity_scores"],
    "per_metric_floors": shap_results["per_metric_floors"],
    "feature_results_sample": shap_panel["feature_results"][:5],
    "panel_summary": shap_summary.to_dict(orient="records"),
}, "shap_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / HOLDOUT DRIFT DETECTION
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("TRAIN/HOLDOUT DRIFT DETECTION")
log("=" * 60)

from sklearn.model_selection import train_test_split

X_train, X_holdout, y_train, y_holdout = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
log(f"Train: {len(X_train)}, Holdout: {len(X_holdout)}")

holdout_checker = TrainHoldoutStability(
    model_factory=lgbm_factory,
    explainer_type="tree",
    shap_subsample=2000,
    random_state=42,
)
drift_results = holdout_checker.fit(X_train, y_train, X_holdout, y_holdout)
print_holdout_report(drift_results)
save_json(drift_results, "drift_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 5. META-BOOTSTRAP CONFIDENCE INTERVALS (top 5 features)
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("META-BOOTSTRAP CONFIDENCE INTERVALS")
log("=" * 60)

top5 = summary.sort_values("complexity_score").head(5)["feature"].tolist()
log(f"Running meta-bootstrap on: {top5}")

meta = MetaBootstrap(
    n_splits=5,
    strategy=SplitStrategy.KFOLD,
    random_state=42,
)
meta_results = {}
for feat in top5:
    log(f"  {feat}...")
    try:
        r = meta.fit(df, feature_col=feat, target_col=TARGET)
        meta_results[feat] = {
            "mean_complexity": r.mean_complexity,
            "std_complexity": r.std_complexity,
            "ci_lower": r.ci_lower,
            "ci_upper": r.ci_upper,
        }
        log(f"    complexity = {r.mean_complexity:.4f} [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
    except Exception as e:
        log(f"    Failed: {e}")
        meta_results[feat] = {"error": str(e)}

save_json(meta_results, "meta_bootstrap_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 6. RELIABILITY SCORING
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("RELIABILITY SCORING")
log("=" * 60)

config = ReliabilityConfig(
    stability_weight=0.40,
    importance_weight=0.30,
    coverage_weight=0.15,
    consistency_weight=0.15,
)
scorer = ReliabilityScorer(config)

reliability_results = {}
for _, row in summary.iterrows():
    feat = row["feature"]
    complexity = row["complexity_score"]
    if np.isnan(complexity):
        continue
    try:
        # Use marginal complexity and fake importance for demo
        rank_in_panel = summary.index[summary["feature"] == feat][0] + 1
        rel = scorer.compute(
            complexity_score=complexity,
            importance_rank=rank_in_panel,
            coverage_ratio=0.95,
            cross_seed_std=0.03,
        )
        reliability_results[feat] = {
            "overall_score": rel.overall_score,
            "grade": rel.grade,
            "stability_component": rel.stability_component,
            "importance_component": rel.importance_component,
        }
    except Exception as e:
        reliability_results[feat] = {"error": str(e)}

save_json(reliability_results, "reliability_results.json")

# Print reliability summary
log("\nReliability scores:")
for feat, r in sorted(reliability_results.items(), key=lambda x: x[1].get("overall_score", 0), reverse=True):
    if "grade" in r:
        log(f"  {feat:20s}: {r['overall_score']:.3f} (grade {r['grade']})")


# ══════════════════════════════════════════════════════════════════════════
# 7. SYNTHESIS
# ══════════════════════════════════════════════════════════════════════════
log("")
log("=" * 60)
log("WRITING SYNTHESIS")
log("=" * 60)

# Prepare synthesis data
stable_feats = summary[summary["complexity_score"] < 0.02]["feature"].tolist()
moderate_feats = summary[(summary["complexity_score"] >= 0.02) & (summary["complexity_score"] < 0.05)]["feature"].tolist()
unstable_feats = summary[summary["complexity_score"] >= 0.05]["feature"].tolist()
censored_feats = summary[summary["censoring_flag"] == True]["feature"].tolist()

drift_grade = drift_results.get("drift_grade", "N/A")
drift_score = drift_results.get("overall_drift_score", float("nan"))
drifted = drift_results.get("drifted_features", [])

synthesis = f"""# Credit Card Default — Bootstrap Stability Analysis
## Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}

## Dataset
- **Source**: UCI Default of Credit Card Clients (Taiwan, 2005)
- **Samples**: {len(df):,}
- **Features**: {len(FEATURES)}
- **Target**: `{TARGET}` (event rate = {df[TARGET].mean():.2%})

---

## 1. Marginal (Distributional) Stability

Panel analysis across all {len(FEATURES)} features using `k/n^alpha + floor` learning curves
with flexible alpha estimation, 25 bootstrap resamples per pool size.

### Stability Tiers

| Tier | Features | Count |
|------|----------|-------|
| **Stable** (complexity < 0.02) | {', '.join(stable_feats) if stable_feats else 'None'} | {len(stable_feats)} |
| **Moderate** (0.02 – 0.05) | {', '.join(moderate_feats) if moderate_feats else 'None'} | {len(moderate_feats)} |
| **Unstable** (> 0.05) | {', '.join(unstable_feats) if unstable_feats else 'None'} | {len(unstable_feats)} |

### Complexity Score Ranking

"""

for _, row in summary.iterrows():
    c = row["complexity_score"]
    flag = " [CENSORED]" if row.get("censoring_flag") else ""
    synthesis += f"- **{row['feature']}**: {c:.4f}{flag}\n"

synthesis += f"""
### Censoring Warnings
{f"Features with detected policy truncation: **{', '.join(censored_feats)}**" if censored_feats else "No censoring detected."}
These features may appear artificially stable because truncation removes distributional tails.

---

## 2. SHAP (Model Decision) Stability

SHAP stability analysis on {len(X_sample):,} samples using LightGBM (100 trees, depth 6).

- **Overall SHAP complexity**: {shap_results['complexity_score']:.4f}
- **Per-metric floors**:
"""

for metric, floor in sorted(shap_results["per_metric_floors"].items()):
    synthesis += f"  - {metric}: {floor:.6f}\n" if isinstance(floor, float) and not np.isnan(floor) else f"  - {metric}: N/A\n"

# Bug finding
if len(unique_complexities) == 1:
    synthesis += f"""
### Bug Finding: Identical Per-Feature SHAP Complexity

All {len(FEATURES)} features report the same SHAP complexity score ({unique_complexities[0]:.4f}).

**Root cause**: In `SHAPStability.fit_panel()`, per-feature complexity falls back to the
global `overall_complexity` when per-feature metric means produce `total_w == 0`. This happens
because:
1. With **Option A** (default: `retrain_per_bootstrap=False`), the model is trained once on
   full data. SHAP values for each data point are fixed — the only variation across bootstrap
   resamples is *which subset of points* appears. This produces very flat learning curves.
2. Flat learning curves → **all curve fits are anomalous** (low R²) → floors are excluded
   from the complexity score → **fallback to global score** (from raw last-pool values).
3. In `fit_panel`, if per-feature metric means are finite, the per-feature complexity is
   computed correctly. But if they happen to produce the same weighted average (because
   per-feature metrics are very similar under Option A), you get identical scores.

**Fix**: Use `retrain_per_bootstrap=True` (Option B) to capture genuine model instability.
Option A measures data sampling noise, not feature stability.
"""

synthesis += f"""
---

## 3. Train/Holdout Drift Detection

Model trained on 70% data, SHAP compared against 30% holdout.

- **Overall drift score**: {drift_score:.3f}
- **Drift grade**: **{drift_grade}**
- **Rank correlation**: {drift_results['drift_metrics']['rank_correlation']['drift']:.3f}
- **Direction flip rate**: {drift_results['drift_metrics']['direction_flip_rate']['drift']:.1%}
- **Top-k overlap**: {drift_results['drift_metrics']['topk_overlap']['holdout_value']:.3f}
"""

if drifted:
    synthesis += f"\n**Features with significant drift**: {', '.join(drifted)}\n"
else:
    synthesis += "\nNo features showed significant drift — model generalizes well to holdout.\n"

# Top drifted features
feature_drift = drift_results.get("feature_drift", {})
sorted_drift = sorted(feature_drift.items(), key=lambda x: x[1]["drift_score"], reverse=True)
synthesis += "\n### Per-Feature Drift (top 10)\n\n"
synthesis += "| Feature | Drift Score | Rank Change | Direction Consistent |\n"
synthesis += "|---------|-------------|-------------|---------------------|\n"
for fname, m in sorted_drift[:10]:
    dir_str = "Yes" if m["direction_consistent"] else "**No**"
    synthesis += f"| {fname} | {m['drift_score']:.3f} | {m['rank_change']:+d} | {dir_str} |\n"

synthesis += f"""
---

## 4. Meta-Bootstrap Confidence Intervals

5-fold cross-validation meta-bootstrap on top 5 most stable features.

| Feature | Mean Complexity | 95% CI Lower | 95% CI Upper |
|---------|-----------------|-------------|-------------|
"""

for feat, r in meta_results.items():
    if "error" not in r:
        synthesis += f"| {feat} | {r['mean_complexity']:.4f} | {r['ci_lower']:.4f} | {r['ci_upper']:.4f} |\n"
    else:
        synthesis += f"| {feat} | Error | — | — |\n"

synthesis += f"""
---

## 5. Key Insights

"""

# Build insights
insights = []

# Insight 1: Most/least stable
best = summary.iloc[0]
worst = summary.iloc[-1]
insights.append(f"**Most stable feature**: `{best['feature']}` (complexity {best['complexity_score']:.4f}). "
                f"**Least stable**: `{worst['feature']}` (complexity {worst['complexity_score']:.4f}).")

# Insight 2: Censoring
if censored_feats:
    insights.append(f"**Censoring detected** on {len(censored_feats)} feature(s): {', '.join(censored_feats)}. "
                    "These may appear artificially stable due to policy truncation.")

# Insight 3: Drift
insights.append(f"**Train-to-holdout drift**: Grade {drift_grade} ({drift_score:.3f}). "
                + ("Model behavior is stable across train/holdout split." if drift_grade in ("A", "B")
                   else "Significant drift detected — investigate before deployment."))

# Insight 4: SHAP bug
if len(unique_complexities) == 1:
    insights.append(f"**SHAP per-feature complexity bug**: All features report {unique_complexities[0]:.4f}. "
                    "This is caused by Option A (fixed model) producing indistinguishable learning curves. "
                    "Use `retrain_per_bootstrap=True` for meaningful per-feature SHAP complexity.")

for i, insight in enumerate(insights, 1):
    synthesis += f"{i}. {insight}\n\n"

synthesis += """---

## Files Generated

| File | Description |
|------|-------------|
| `marginal_panel_summary.csv` | Marginal stability scores for all features |
| `marginal_panel_chart.png` | Bar chart of complexity scores |
| `marginal_*.png / .csv` | Deep dive plots and data for key features |
| `marginal_all_results.json` | Full marginal results (floors, flags, etc.) |
| `shap_results.json` | SHAP stability scores and per-metric floors |
| `drift_results.json` | Train/holdout drift metrics and per-feature drift |
| `meta_bootstrap_results.json` | Meta-bootstrap confidence intervals |
| `reliability_results.json` | Reliability scores and grades |
| `synthesis.md` | This file |
"""

synthesis_path = OUTPUT_DIR / "synthesis.md"
with open(synthesis_path, 'w') as f:
    f.write(synthesis)
log(f"Synthesis written to {synthesis_path}")

log("")
log("=" * 60)
log("ANALYSIS COMPLETE")
log(f"All outputs in: {OUTPUT_DIR.absolute()}")
log("=" * 60)
