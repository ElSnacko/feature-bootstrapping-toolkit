#!/usr/bin/env python3
"""
Lending Club Experiment: Feature Stability as a Learning Curve Problem
======================================================================

Same pipeline as run_article_experiment.py but on Lending Club accepted
loans (2007-2018) with fintech-style behavioural features.

Stages mirror the article experiment:
  1. Data load + feature engineering + EDA
  2. Marginal (distributional) stability panel
  3. SHAP (model-decision) stability panel
  4. Permutation baseline (target-dependent null)
  5. Meta-bootstrap confidence intervals on key features
  6. Marginal-vs-SHAP validation
  7. Synthetic ground-truth detection rates
  8. Train/holdout drift
  9. Consolidated outputs

Usage:
    python run_lending_club_experiment.py [--data-path PATH] [--sample-n 30000]
"""

import os, sys, json, time, warnings, argparse, atexit, signal
from pathlib import Path
from datetime import datetime
from collections import OrderedDict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from lending_club_features import load_lending_club, FEATURE_GROUPS

from bootstrap_stability import (
    BootstrapStability,
    SHAPStability,
    TrainHoldoutStability,
    ReliabilityScorer, ReliabilityConfig,
    PermutationBaseline,
    MetaBootstrap, SplitStrategy,
    MarginalVsSHAPValidator, plot_marginal_vs_shap,
    SyntheticValidation, InstabilityType, print_synthetic_report,
    plot_results, plot_panel, print_report, to_csv, panel_to_csv,
    print_holdout_report, get_complexity_score,
)


# ── process cleanup ───────────────────────────────────────────────────
def _cleanup_workers():
    pid = os.getpid()
    for entry in os.listdir("/proc"):
        if not entry.isdigit():
            continue
        try:
            with open(f"/proc/{entry}/stat") as f:
                ppid = int(f.read().split()[3])
                if ppid == pid:
                    os.kill(int(entry), signal.SIGKILL)
        except (FileNotFoundError, PermissionError, ProcessLookupError,
                IndexError, ValueError):
            pass

atexit.register(_cleanup_workers)
signal.signal(signal.SIGTERM, lambda s, f: (_cleanup_workers(), sys.exit(128 + s)))
signal.signal(signal.SIGINT,  lambda s, f: (_cleanup_workers(), sys.exit(128 + s)))


# ── CONFIG ────────────────────────────────────────────────────────────
OUTPUT_DIR = Path("lending_club_outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv")
SAMPLE_N = 30_000   # subsample for tractable runtime

# Marginal
MARGINAL_RESAMPLES = 25
MARGINAL_SEED = 42

# SHAP
SHAP_SUBSAMPLE = 3000
SHAP_RESAMPLES = 15
SHAP_POINTS = 10

# Permutation baseline
PERM_N = 30
PERM_RESAMPLES = 10

# Meta-bootstrap
META_SPLITS = 10

# Deep-dive features (one per behavioural theme)
DEEP_DIVE = [
    "inq_acceleration",         # credit-seeking velocity
    "bc_liquidity_buffer",      # utilisation dynamics
    "delinq_intensity",         # payment discipline
    "seasoning_gap",            # account lifecycle
    "debt_ex_mort_to_income",   # capacity stress
    "loan_to_income",           # loan-level risk
]

# Meta-bootstrap feature set
META_FEATURES = [
    "inq_acceleration",
    "revol_util",
    "pct_tl_nvr_dlq",
    "debt_ex_mort_to_income",
    "fico_mid",
]

# LightGBM factory
def lgbm_factory():
    from lightgbm import LGBMClassifier
    return LGBMClassifier(
        n_estimators=100, max_depth=6,
        random_state=42, verbose=-1, n_jobs=1,
    )


# ── HELPERS ───────────────────────────────────────────────────────────
_t0 = time.time()

def log(msg=""):
    elapsed = time.time() - _t0
    print(f"[{elapsed:7.1f}s] {msg}")

def save_json(obj, filename):
    def _c(o):
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.bool_): return bool(o)
        if isinstance(o, dict): return {k: _c(v) for k, v in o.items()}
        if isinstance(o, list): return [_c(v) for v in o]
        if isinstance(o, float) and np.isnan(o): return None
        return o
    path = OUTPUT_DIR / filename
    with open(path, "w") as f:
        json.dump(_c(obj), f, indent=2, default=str)
    log(f"  saved  {path}")


# ═══════════════════════════════════════════════════════════════════════
# STAGE 1 — DATA + FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════
def load_data(data_path, sample_n):
    log("STAGE 1 — LOADING DATA + FEATURE ENGINEERING")
    features, target, feature_groups = load_lending_club(
        data_path, sample_n=sample_n, random_state=42,
    )

    FEATURES = list(features.columns)
    TARGET = "default"

    # Combine for compatibility with marginal pipeline
    df = features.copy()
    df[TARGET] = target.values

    log(f"  shape={df.shape}  event_rate={target.mean():.4f}  features={len(FEATURES)}")

    # EDA snapshot
    eda = {
        "n_rows": len(df),
        "n_features": len(FEATURES),
        "event_rate": float(target.mean()),
        "feature_groups": {k: v for k, v in feature_groups.items()},
        "feature_types": {},
    }
    for f in FEATURES:
        nuniq = df[f].nunique()
        if nuniq == 2:
            eda["feature_types"][f] = "binary"
        elif nuniq <= 10:
            eda["feature_types"][f] = "categorical"
        else:
            eda["feature_types"][f] = "continuous"
    save_json(eda, "eda_summary.json")

    return df, TARGET, FEATURES, feature_groups


# ═══════════════════════════════════════════════════════════════════════
# STAGE 2 — MARGINAL STABILITY
# ═══════════════════════════════════════════════════════════════════════
def run_marginal(df, TARGET):
    log("\nSTAGE 2 — MARGINAL STABILITY PANEL")
    bs = BootstrapStability(
        n_resamples=MARGINAL_RESAMPLES,
        estimate_alpha=False, fixed_alpha=0.5,
        support_categorical=True,
        random_state=MARGINAL_SEED, n_jobs=-1,
    )
    panel = bs.fit_panel(df, target_col=TARGET)
    summary = panel["summary"]

    panel_to_csv(panel, str(OUTPUT_DIR / "marginal_panel.csv"))
    fig = plot_panel(panel, save_path=str(OUTPUT_DIR / "fig_marginal_panel.png"))
    plt.close(fig)

    for feat in DEEP_DIVE:
        if feat in panel["feature_results"]:
            r = panel["feature_results"][feat]
            fig = plot_results(r, save_path=str(OUTPUT_DIR / f"fig_marginal_{feat}.png"))
            plt.close(fig)
            to_csv(r, str(OUTPUT_DIR / f"marginal_{feat}.csv"))

    marginal_json = {}
    for feat, r in panel["feature_results"].items():
        marginal_json[feat] = {
            "complexity_score": r["complexity_score"],
            "complexity_scores": r.get("complexity_scores", {}),
            "per_metric_floors": r.get("per_metric_floors", {}),
            "censoring_flag": r["meta"]["censoring_flag"],
            "censoring_severity": r["meta"].get("censoring_severity", 0.0),
            "censoring_detail": r["meta"]["censoring_detail"],
            "feature_type": r["meta"]["feature_type"],
        }
    save_json(marginal_json, "marginal_results.json")

    log(f"  {len(summary)} features analysed")
    return panel


# ═══════════════════════════════════════════════════════════════════════
# STAGE 3 — SHAP STABILITY
# ═══════════════════════════════════════════════════════════════════════
def run_shap(df, TARGET, FEATURES, retrain_per_bootstrap=False):
    log("\nSTAGE 3 — SHAP STABILITY")
    X = df[FEATURES]
    y = df[TARGET]

    np.random.seed(42)
    idx = np.random.choice(len(X), min(SHAP_SUBSAMPLE, len(X)), replace=False)
    X_sub = X.iloc[idx].reset_index(drop=True)
    y_sub = y.iloc[idx].reset_index(drop=True)

    shap_analyzer = SHAPStability(
        model_factory=lgbm_factory,
        n_resamples=SHAP_RESAMPLES, n_points=SHAP_POINTS,
        explainer_type="tree",
        retrain_per_bootstrap=retrain_per_bootstrap,
        random_state=42, verbose=1,
    )

    log("  fitting overall SHAP learning curves ...")
    shap_overall = shap_analyzer.fit(X_sub, y_sub)

    log("  fitting per-feature SHAP panel ...")
    shap_panel = shap_analyzer.fit_panel(X_sub, y_sub)

    save_json({
        "complexity_score": shap_overall["complexity_score"],
        "complexity_scores": shap_overall.get("complexity_scores", {}),
        "per_metric_floors": shap_overall.get("per_metric_floors", {}),
        "learning_curve_fits": {
            m: lc.get("fit", {}) for m, lc in shap_overall["learning_curves"].items()
        },
        "panel_summary": shap_panel["summary"].to_dict(orient="records"),
    }, "shap_results.json")

    log(f"  overall SHAP complexity = {shap_overall['complexity_score']:.4f}")
    return shap_panel, X, y


# ═══════════════════════════════════════════════════════════════════════
# STAGE 4 — PERMUTATION BASELINE
# ═══════════════════════════════════════════════════════════════════════
def run_permutation(df, TARGET, FEATURES):
    log("\nSTAGE 4 — PERMUTATION BASELINE (target_dependent)")
    perm = PermutationBaseline(
        n_permutations=PERM_N,
        analyzer_kwargs=dict(
            n_resamples=PERM_RESAMPLES,
            support_categorical=True,
            estimate_alpha=False, fixed_alpha=0.5,
        ),
        random_state=42, verbose=0,
    )
    perm_results = perm.fit_panel(
        df, target_col=TARGET,
        feature_cols=FEATURES,
        category="target_dependent",
    )

    save_json({
        "summary": perm_results["summary"].to_dict(orient="records"),
        "per_feature": [
            {
                "feature": r["feature"],
                "observed": r["observed"],
                "null_mean": r["null_mean"],
                "null_std": r["null_std"],
                "null_scores": r.get("null_scores", []),
                "p_value": r["p_value"],
                "z_score": r["z_score"],
                "significant": r["significant"],
            }
            for r in perm_results["results"]
        ],
    }, "permutation_results.json")

    sig_count = sum(1 for r in perm_results["results"] if r["significant"])
    log(f"  {sig_count}/{len(FEATURES)} features significantly above null (p<0.05)")
    return perm_results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 5 — META-BOOTSTRAP CIs
# ═══════════════════════════════════════════════════════════════════════
def run_meta_bootstrap(df, TARGET):
    log("\nSTAGE 5 — META-BOOTSTRAP CIs")
    meta = MetaBootstrap(
        n_splits=META_SPLITS,
        strategy=SplitStrategy.KFOLD,
        random_state=42,
        n_jobs=1,
    )

    ci_results = {}
    for feat in META_FEATURES:
        log(f"  {feat} ...")
        try:
            results = meta.fit(
                df, feature_col=feat, target_col=TARGET,
                n_resamples=15, support_categorical=True,
                random_state=42, n_jobs=-1,
            )
            r = results[feat]
            ci_results[feat] = {
                "mean": r.mean_complexity,
                "std": r.std_complexity,
                "ci_lower": r.ci_lower,
                "ci_upper": r.ci_upper,
            }
            log(f"    complexity = {r.mean_complexity:.4f}  "
                f"95% CI = [{r.ci_lower:.4f}, {r.ci_upper:.4f}]")
        except Exception as e:
            log(f"    FAILED: {e}")
            ci_results[feat] = {"error": str(e)}

    save_json(ci_results, "meta_bootstrap_cis.json")
    return ci_results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 6 — MARGINAL vs SHAP VALIDATION
# ═══════════════════════════════════════════════════════════════════════
def run_validation(marginal_panel, shap_panel, perm_results):
    log("\nSTAGE 6 — MARGINAL vs SHAP VALIDATION")
    validator = MarginalVsSHAPValidator()
    comparison = validator.compare(
        marginal_panel, shap_panel,
        permutation_results=perm_results,
    )

    comp_df = comparison["comparison"]
    comp_df.to_csv(OUTPUT_DIR / "validation_comparison.csv", index=False)

    save_json({
        "rank_correlation": comparison["rank_correlation"],
        "rank_pvalue": comparison["rank_pvalue"],
        "marginal_threshold": comparison["marginal_threshold"],
        "shap_threshold": comparison["shap_threshold"],
        "threshold_source": comparison["threshold_source"],
        "quadrant_counts": comparison["quadrant_counts"],
        "per_feature": comp_df.to_dict(orient="records"),
    }, "validation_results.json")

    fig = plot_marginal_vs_shap(comparison,
                                save_path=str(OUTPUT_DIR / "fig_marginal_vs_shap.png"))
    plt.close(fig)

    log(f"  rho = {comparison['rank_correlation']:.3f}  "
        f"threshold_source = {comparison['threshold_source']}")
    for q, n in comparison["quadrant_counts"].items():
        log(f"    {q}: {n}")

    return comparison


# ═══════════════════════════════════════════════════════════════════════
# STAGE 7 — SYNTHETIC GROUND TRUTH
# ═══════════════════════════════════════════════════════════════════════
def run_synthetic():
    log("\nSTAGE 7 — SYNTHETIC VALIDATION")
    validator = SyntheticValidation(random_state=42)

    test_configs = [
        ("heteroscedastic", InstabilityType.HETEROSCEDASTIC),
        ("distribution_shift", InstabilityType.DISTRIBUTION_SHIFT),
        ("interaction", InstabilityType.INTERACTION),
        ("missing_not_at_random", InstabilityType.MISSING_NOT_AT_RANDOM),
    ]

    synthetic_results = {}
    for name, itype in test_configs:
        log(f"  {name} ...")
        try:
            X_syn, y_syn, meta = validator.generate_test_data(
                n_samples=2000,
                n_features=10,
                instability_type=itype,
                n_corrupted=3,
            )
            r = validator.run_test(X_syn, y_syn, meta, threshold=0.05)
            synthetic_results[name] = {
                "detection_rate": r.detection_rate,
                "false_positive_rate": r.false_positive_rate,
                "f1_score": r.f1_score,
                "precision": r.precision,
                "recall": r.recall,
            }
            log(f"    detection={r.detection_rate:.0%}  "
                f"FPR={r.false_positive_rate:.0%}  F1={r.f1_score:.3f}")
        except Exception as e:
            log(f"    FAILED: {e}")
            synthetic_results[name] = {"error": str(e)}

    save_json(synthetic_results, "synthetic_results.json")
    return synthetic_results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 8 — TRAIN/HOLDOUT DRIFT
# ═══════════════════════════════════════════════════════════════════════
def run_drift(X, y):
    log("\nSTAGE 8 — TRAIN/HOLDOUT DRIFT")
    from sklearn.model_selection import train_test_split

    X_tr, X_ho, y_tr, y_ho = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y,
    )

    checker = TrainHoldoutStability(
        model_factory=lgbm_factory,
        explainer_type="tree",
        shap_subsample=2000,
        random_state=42,
    )
    drift = checker.fit(X_tr, y_tr, X_ho, y_ho)
    save_json(drift, "drift_results.json")

    grade = drift.get("drift_grade", "?")
    score = drift.get("overall_drift_score", float("nan"))
    log(f"  grade={grade}  score={score:.4f}")
    return drift


# ═══════════════════════════════════════════════════════════════════════
# STAGE 9 — SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════
def write_synthesis(
    eda, marginal_panel, shap_panel, perm_results,
    ci_results, comparison, synthetic_results, drift, feature_groups,
):
    log("\nSTAGE 9 — WRITING SYNTHESIS")

    m_summary = marginal_panel["summary"]
    s_summary = shap_panel["summary"]
    comp_df = comparison["comparison"]

    lines = []
    def w(s=""): lines.append(s)

    w("# Lending Club Feature Stability Analysis")
    w()
    w(f"**Generated**: {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}")
    w(f"**Dataset**: Lending Club Accepted Loans (2007-2018)")
    w(f"**Samples**: {eda['n_rows']:,} | **Features**: {eda['n_features']} | "
      f"**Event rate**: {eda['event_rate']:.2%}")
    w()

    # Feature groups summary
    w("## Feature Groups (Fintech Behavioural Themes)")
    w()
    for grp, cols in feature_groups.items():
        w(f"- **{grp}** ({len(cols)} features): {', '.join(cols)}")
    w()

    # Marginal results
    w("## Marginal (Distributional) Stability")
    w()
    w("| Feature | Complexity | Censoring Sev. | Spearman Floor | IV Floor |")
    w("|---------|-----------|---------------|---------------|---------|")
    for _, row in m_summary.iterrows():
        cs = row.get("censoring_severity", 0.0)
        w(f"| {row['feature']} | {row['complexity_score']:.4f} | "
          f"{cs:.3f} | {row.get('spearman_floor', float('nan')):.4f} | "
          f"{row.get('iv_floor', float('nan')):.4f} |")
    w()

    # Permutation baseline
    w("## Permutation Baseline (Target-Dependent)")
    w()
    if "summary" in perm_results:
        perm_df = perm_results["summary"]
        sig = perm_df[perm_df["significant"] == True]
        nonsig = perm_df[perm_df["significant"] == False]
        w(f"**{len(sig)}/{len(perm_df)} features** significantly above null (p < 0.05).")
        w()
        if len(sig) > 0:
            w("Significant: " + ", ".join(sig["feature"].tolist()))
        if len(nonsig) > 0:
            w()
            w("Noise-level: " + ", ".join(nonsig["feature"].tolist()))
    w()

    # SHAP stability
    w("## SHAP (Model-Decision) Stability")
    w()
    w("| Feature | SHAP Complexity | Direction Consistency | Rank Stability |")
    w("|---------|----------------|----------------------|----------------|")
    for _, row in s_summary.iterrows():
        w(f"| {row['feature']} | {row['complexity_score']:.3f} | "
          f"{row.get('direction_consistency', float('nan')):.3f} | "
          f"{row.get('rank_stability', float('nan')):.3f} |")
    w()

    # Meta-bootstrap CIs
    w("## Confidence Intervals (Meta-Bootstrap)")
    w()
    w("| Feature | Mean Complexity | 95% CI |")
    w("|---------|----------------|--------|")
    for feat in META_FEATURES:
        ci = ci_results.get(feat, {})
        if "error" in ci:
            w(f"| {feat} | ERROR | {ci['error']} |")
        else:
            w(f"| {feat} | {ci['mean']:.4f} | "
              f"[{ci['ci_lower']:.4f}, {ci['ci_upper']:.4f}] |")
    w()

    # Marginal vs SHAP
    w("## Marginal vs SHAP Divergence")
    w()
    rho = comparison["rank_correlation"]
    src = comparison["threshold_source"]
    w(f"Spearman ρ = **{rho:.3f}** (threshold source: {src})")
    w()
    w("| Quadrant | Count | Features |")
    w("|----------|-------|----------|")
    for q in ["concordant_stable", "concordant_unstable", "false_alarm", "missed_risk"]:
        feats = comp_df[comp_df["quadrant"] == q]["feature"].tolist()
        w(f"| {q} | {len(feats)} | {', '.join(feats)} |")
    w()

    # Synthetic
    w("## Synthetic Ground Truth")
    w()
    w("| Instability Type | Detection Rate | FPR | F1 |")
    w("|-----------------|---------------|-----|-----|")
    for name, r in synthetic_results.items():
        if "error" in r:
            w(f"| {name} | ERROR | | {r['error']} |")
        else:
            w(f"| {name} | {r['detection_rate']:.0%} | "
              f"{r['false_positive_rate']:.0%} | {r['f1_score']:.3f} |")
    w()

    # Drift
    w("## Train/Holdout Drift")
    w()
    grade = drift.get("drift_grade", "?")
    score = drift.get("overall_drift_score", float("nan"))
    w(f"Grade: **{grade}** (score = {score:.4f})")
    w()

    # Files
    w("## Files Generated")
    w()
    for p in sorted(OUTPUT_DIR.iterdir()):
        w(f"  {p.name}")
    w()

    text = "\n".join(lines)
    path = OUTPUT_DIR / "lending_club_synthesis.md"
    path.write_text(text)
    log(f"  saved  {path}")


# ═══════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", default=DATA_PATH)
    parser.add_argument("--sample-n", type=int, default=SAMPLE_N,
                        help="Subsample size (default 30000). Use 0 for all data.")
    parser.add_argument("--start-stage", type=int, default=1, choices=range(1, 10))
    parser.add_argument("--shap-retrain", action="store_true", default=False)
    args = parser.parse_args()

    sample_n = args.sample_n if args.sample_n > 0 else None
    start = args.start_stage

    log("=" * 70)
    log("LENDING CLUB EXPERIMENT — Feature Stability (Behavioural Metrics)")
    if start > 1:
        log(f"  RESUMING FROM STAGE {start}")
    log("=" * 70)

    # Stage 1
    df, TARGET, FEATURES, feature_groups = load_data(args.data_path, sample_n)
    eda = json.loads((OUTPUT_DIR / "eda_summary.json").read_text())

    # Stage 2
    if start <= 2:
        marginal_panel = run_marginal(df, TARGET)
    else:
        log("\n  [re-running stage 2 for live objects]")
        marginal_panel = run_marginal(df, TARGET)

    # Stage 3
    if start <= 3:
        shap_panel, X, y = run_shap(df, TARGET, FEATURES, args.shap_retrain)
    else:
        log("\n  [re-running stage 3 for live objects]")
        shap_panel, X, y = run_shap(df, TARGET, FEATURES, args.shap_retrain)

    # Stage 4
    if start <= 4:
        perm_results = run_permutation(df, TARGET, FEATURES)
    else:
        perm_json = json.loads((OUTPUT_DIR / "permutation_results.json").read_text())
        perm_results = {
            "summary": pd.DataFrame(perm_json["summary"]),
            "results": perm_json["per_feature"],
        }
        has_null_scores = any(r.get("null_scores") for r in perm_results["results"])
        if not has_null_scores and start <= 6:
            log("\n  [re-running stage 4 — null_scores missing]")
            perm_results = run_permutation(df, TARGET, FEATURES)

    # Stage 5
    if start <= 5:
        ci_results = run_meta_bootstrap(df, TARGET)
    else:
        ci_results = json.loads((OUTPUT_DIR / "meta_bootstrap_cis.json").read_text())

    # Stage 6
    if start <= 6:
        comparison = run_validation(marginal_panel, shap_panel, perm_results)
    else:
        comp_json = json.loads((OUTPUT_DIR / "validation_results.json").read_text())
        comparison = comp_json
        comparison["comparison"] = pd.DataFrame(comp_json["per_feature"])

    # Stage 7
    if start <= 7:
        synthetic_results = run_synthetic()
    else:
        synthetic_results = json.loads((OUTPUT_DIR / "synthetic_results.json").read_text())

    # Stage 8
    if start <= 8:
        drift = run_drift(X, y)
    else:
        drift = json.loads((OUTPUT_DIR / "drift_results.json").read_text())

    # Stage 9
    write_synthesis(
        eda, marginal_panel, shap_panel, perm_results,
        ci_results, comparison, synthetic_results, drift, feature_groups,
    )

    log()
    log("=" * 70)
    log(f"DONE — all outputs in {OUTPUT_DIR.absolute()}")
    log("=" * 70)


if __name__ == "__main__":
    main()
