#!/usr/bin/env python3
"""
Full bootstrap stability analysis on UCI Credit Card Default dataset.
No model retraining (Option A for SHAP).
"""
import os, sys, json, warnings
from pathlib import Path
from datetime import datetime

import matplotlib
matplotlib.use('Agg')
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
warnings.filterwarnings('ignore')

from bootstrap_stability import (
    BootstrapStability, SHAPStability, TrainHoldoutStability,
    ReliabilityScorer, ReliabilityConfig,
    plot_results, plot_panel, print_report, to_csv, panel_to_csv,
    print_holdout_report, get_complexity_score,
)

OUTPUT_DIR = Path("credit_card_full_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}")

def save_json(obj, filename):
    def _convert(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer, np.int64)): return int(o)
        if isinstance(o, (np.floating, np.float64)): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: _convert(v) for k, v in o.items()}
        if isinstance(o, list): return [_convert(v) for v in o]
        if isinstance(o, (float, int)) and (o != o): return None
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

data_path = os.path.join(os.path.dirname(__file__), '..', '..', 'default+of+credit+card+clients', 'default of credit card clients.xls')
df = pd.read_excel(data_path, header=1)
df = df.drop(columns=['ID'])

TARGET = 'default payment next month'
FEATURES = [c for c in df.columns if c != TARGET]
X = df[FEATURES]
y = df[TARGET]

log(f"Shape: {df.shape}, Target event rate: {y.mean():.4f}")


# ══════════════════════════════════════════════════════════════════════════
# 2. MARGINAL STABILITY — PANEL
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("MARGINAL STABILITY PANEL (fixed alpha=0.5)")

bs = BootstrapStability(
    n_resamples=25,
    estimate_alpha=False,
    fixed_alpha=0.5,
    support_categorical=True,
    random_state=42,
    n_jobs=-1,
)
panel = bs.fit_panel(df, target_col=TARGET)
summary = panel["summary"]

log("\nPanel summary:")
cols = ["feature", "complexity_score", "censoring_flag", "wasserstein_floor",
        "ks_floor", "spearman_floor", "iv_floor"]
available = [c for c in cols if c in summary.columns]
print(summary[available].to_string(index=False))

# Save
panel_to_csv(panel, str(OUTPUT_DIR / "marginal_panel_summary.csv"))
fig = plot_panel(panel, save_path=str(OUTPUT_DIR / "marginal_panel_chart.png"))
fig.clf()

# Deep dives
for feat in ["LIMIT_BAL", "PAY_0", "AGE", "BILL_AMT1"]:
    if feat in panel["feature_results"]:
        r = panel["feature_results"][feat]
        fig = plot_results(r, save_path=str(OUTPUT_DIR / f"marginal_{feat}.png"))
        fig.clf()
        to_csv(r, str(OUTPUT_DIR / f"marginal_{feat}.csv"))

# Save all marginal results
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
# 3. SHAP STABILITY (Option A — no retraining)
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("SHAP STABILITY (Option A, retrain_per_bootstrap=False)")

from lightgbm import LGBMClassifier

def lgbm_factory():
    return LGBMClassifier(
        n_estimators=100, max_depth=6,
        random_state=42, verbose=-1, n_jobs=1,
    )

# Subsample for SHAP speed
np.random.seed(42)
idx = np.random.choice(len(X), 3000, replace=False)
X_shap = X.iloc[idx].reset_index(drop=True)
y_shap = y.iloc[idx].reset_index(drop=True)

log(f"SHAP subsample: {len(X_shap)} rows")

shap_analyzer = SHAPStability(
    model_factory=lgbm_factory,
    n_resamples=15,
    n_points=10,
    explainer_type="tree",
    retrain_per_bootstrap=False,
    random_state=42,
    verbose=2,
)

log("Running SHAPStability.fit()...")
shap_results = shap_analyzer.fit(X_shap, y_shap)

log(f"\nOverall SHAP complexity: {shap_results['complexity_score']}")

log("\nLearning curve fits:")
for metric, lc in shap_results["learning_curves"].items():
    fit = lc.get("fit", {})
    floor = fit.get("floor", float("nan"))
    r2 = fit.get("r2", float("nan"))
    anom = fit.get("anomalous", "N/A")
    log(f"  {metric:30s}  floor={floor:>10.6f}  r2={r2:>8.4f}  anomalous={anom}")

# fit_panel for per-feature complexity
log("\nRunning SHAPStability.fit_panel()...")
shap_analyzer2 = SHAPStability(
    model_factory=lgbm_factory,
    n_resamples=15, n_points=10,
    explainer_type="tree",
    retrain_per_bootstrap=False,
    random_state=42, verbose=1,
)
shap_panel = shap_analyzer2.fit_panel(X_shap, y_shap)
shap_summary = shap_panel["summary"]

log("\nSHAP panel summary:")
print(shap_summary.to_string(index=False))

shap_complexities = shap_summary["complexity_score"].dropna().values
unique_c = np.unique(np.round(shap_complexities, 6))
log(f"\nUnique per-feature SHAP complexity values: {len(unique_c)}")

save_json({
    "complexity_score": shap_results["complexity_score"],
    "complexity_scores": shap_results.get("complexity_scores", {}),
    "per_metric_floors": shap_results.get("per_metric_floors", {}),
    "learning_curve_fits": {
        m: lc.get("fit", {}) for m, lc in shap_results["learning_curves"].items()
    },
    "panel_summary": shap_summary.to_dict(orient="records"),
    "feature_results_sample": shap_panel["feature_results"][:5],
}, "shap_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 4. TRAIN / HOLDOUT DRIFT
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("TRAIN/HOLDOUT DRIFT DETECTION")

from sklearn.model_selection import train_test_split

X_train, X_hold, y_train, y_hold = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
log(f"Train: {len(X_train)}, Holdout: {len(X_hold)}")

holdout_checker = TrainHoldoutStability(
    model_factory=lgbm_factory,
    explainer_type="tree",
    shap_subsample=2000,
    random_state=42,
)
drift_results = holdout_checker.fit(X_train, y_train, X_hold, y_hold)
print_holdout_report(drift_results)
save_json(drift_results, "drift_results.json")


# ══════════════════════════════════════════════════════════════════════════
# 5. RELIABILITY SCORING
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log("RELIABILITY SCORING")

scorer = ReliabilityScorer(ReliabilityConfig())

# Get SHAP importances from the panel for importance_score
shap_importance = {}
for fr in shap_panel["feature_results"]:
    fname = fr["feature"]
    # Use mean absolute wasserstein as proxy for importance
    shap_importance[fname] = fr.get("wasserstein_mean", 0.0)

# Normalize importance to [0, 1]
max_imp = max(shap_importance.values()) if shap_importance else 1.0
if max_imp == 0: max_imp = 1.0

reliability_results = {}
for _, row in summary.iterrows():
    feat = row["feature"]
    complexity = row["complexity_score"]
    if np.isnan(complexity):
        continue
    imp = shap_importance.get(feat, 0.0) / max_imp
    try:
        rel = scorer.compute(
            feature_name=feat,
            complexity_score=complexity,
            importance_score=imp,
            coverage_ratio=0.95,
            cross_seed_std=0.03,
        )
        reliability_results[feat] = {
            "reliability_score": rel.reliability_score,
            "stability_component": rel.stability_component,
            "importance_component": rel.importance_component,
            "coverage_component": rel.coverage_component,
            "consistency_component": rel.consistency_component,
        }
    except Exception as e:
        reliability_results[feat] = {"error": str(e)}

save_json(reliability_results, "reliability_results.json")

log("\nReliability scores:")
for feat, r in sorted(reliability_results.items(),
                       key=lambda x: x[1].get("reliability_score", 0), reverse=True):
    if "reliability_score" in r:
        log(f"  {feat:20s}: {r['reliability_score']:.3f}")


# ══════════════════════════════════════════════════════════════════════════
# DONE
# ══════════════════════════════════════════════════════════════════════════
log("\n" + "=" * 60)
log(f"ANALYSIS COMPLETE — outputs in {OUTPUT_DIR.absolute()}")
log("=" * 60)
