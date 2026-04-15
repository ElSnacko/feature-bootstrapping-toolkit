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

from lending_club_features import load_lending_club, load_lending_club_temporal, FEATURE_GROUPS

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

# Forward CV feature selection (Stage 2b)
FWD_CV_FOLDS = 5
FWD_VIF_THRESHOLD = 5.0      # incremental VIF above which a candidate is collinear
FWD_MIN_AUC_GAIN = 0.003     # minimum ΔAUCroc to admit a feature
FWD_COLLINEAR_AUC_PREMIUM = 0.005  # extra ΔAUC required when incremental VIF > threshold

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
# STAGE 2b — FORWARD CV FEATURE SELECTION
# ═══════════════════════════════════════════════════════════════════════

def _compute_vif(X_sel: pd.DataFrame) -> list:
    """
    Variance Inflation Factor for each column in X_sel.

    VIF_j = 1 / (1 - R²_j), where R²_j comes from regressing column j
    on all remaining columns via OLS.  Values > 10 indicate severe
    multicollinearity; 5-10 is moderate.

    Returns a list of dicts sorted by VIF descending.
    """
    if X_sel.shape[1] < 2:
        return [{"feature": c, "vif": 1.0} for c in X_sel.columns]

    X_arr = X_sel.values.astype(float)
    X_arr = X_arr - X_arr.mean(axis=0)   # centre for numerical stability
    records = []

    for j, col in enumerate(X_sel.columns):
        others = np.delete(X_arr, j, axis=1)
        if np.linalg.matrix_rank(others) < others.shape[1]:
            vif = np.inf
        else:
            coef, _, _, _ = np.linalg.lstsq(others, X_arr[:, j], rcond=None)
            fitted = others @ coef
            ss_res = float(np.sum((X_arr[:, j] - fitted) ** 2))
            ss_tot = float(np.sum((X_arr[:, j] - X_arr[:, j].mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
            vif = 1.0 / (1.0 - r2) if r2 < 1.0 else np.inf

        records.append({
            "feature": col,
            "vif": float(vif) if np.isfinite(vif) else None,
        })

    return sorted(records, key=lambda x: (x["vif"] or 0), reverse=True)


def _plot_forward_selection_curve(step_results: list, output_dir: Path) -> None:
    """
    Four-panel plot of CV metric trajectories across forward selection steps.

    Green vertical bands = feature admitted; red = feature rejected.
    Alternative optimisation metrics shown alongside AUC so analysts can
    evaluate whether a different primary criterion would change the outcome:

    - AUC-ROC   : standard discrimination (primary optimisation target)
    - Gini      : 2*AUC-1, common in credit scoring; same ordering as AUC
    - KS stat   : max CDF separation, widely reported in consumer lending
    - Brier     : proper scoring rule; penalises mis-calibration as well as
                  poor discrimination — a lower-is-better complement to AUC
    """
    steps = [s for s in step_results if s.get("feature") is not None]
    if not steps:
        return

    xs = [s["step"] for s in steps]
    panels = [
        ("cv_auc",   "CV AUC-ROC (↑)",             "#2980b9"),
        ("cv_gini",  "CV Gini / Somers D (↑)",      "#27ae60"),
        ("cv_ks",    "CV KS Statistic (↑)",          "#8e44ad"),
        ("cv_brier", "CV Brier Score (↓ better)",    "#e74c3c"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    for ax, (key, label, color) in zip(axes.flat, panels):
        ys = [s.get(key, np.nan) for s in steps]
        ax.plot(xs, ys, color=color, linewidth=2, zorder=3)

        for s in steps:
            band_color = "#2ecc71" if s.get("action") == "add" else "#e74c3c"
            ax.axvline(s["step"], color=band_color, alpha=0.12, linewidth=1)

        ax.set_xlabel("Forward selection step", fontsize=9)
        ax.set_ylabel(label, fontsize=9)
        ax.set_title(label, fontsize=10)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Forward CV Feature Selection — Metric Trajectories\n"
        "(green bands = admitted, red = rejected)",
        fontsize=12,
    )
    fig.tight_layout()
    path = str(output_dir / "fig_forward_cv_selection.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    log(f"  saved  {path}")


def run_forward_cv_selection(
    df: pd.DataFrame,
    TARGET: str,
    FEATURES: list,
    marginal_panel: dict,
) -> dict:
    """
    Stage 2b — Forward stepwise CV feature selection.

    PURPOSE
    -------
    Removes features that inflate the collinearity of the feature space
    without adding commensurate model-attribution signal.  This improves
    downstream SHAP stability and reduces variance in permutation-importance
    rankings, two known consequences of correlated predictors.

    ALGORITHM
    ---------
    1. Compute the initial VIF of every candidate using the full feature
       matrix.  Sort candidates ascending by VIF so the most orthogonal
       features enter first.  This ordering is independent of marginal
       stability scores, making the selection orthogonal to Stage 2.
    2. Maintain a growing selected set S (initially empty).
    3. For each candidate f (in ascending-VIF order):
         a. Compute the *incremental* VIF of f given S: regress f on all
            features already in S and derive VIF = 1/(1−R²).  This
            captures multivariate collinearity — how much of f's variance
            the joint linear combination of S already explains — rather
            than the maximum of pairwise correlations.
         b. Evaluate CV metrics (AUC, Gini, KS, Brier) for S ∪ {f}.
         c. Compute ΔAUC = AUC(S ∪ {f}) − AUC(S).
         d. Admit f if ΔAUC ≥ required_gain, where:
              required_gain = FWD_MIN_AUC_GAIN
                            + FWD_COLLINEAR_AUC_PREMIUM  (if incremental VIF > threshold)
            A collinear feature must prove unique predictive signal beyond
            what S already captures.
    4. Compute VIF for the final selected set as a post-hoc audit.

    OPTIMISATION METRIC
    -------------------
    Primary: AUC-ROC (threshold-free, handles class imbalance well).

    Alternatives tracked at each step — choose a different primary if:
    • Gini / Somers D  — identical ordering to AUC; preferred in regulatory
                         credit-risk scorecards (Basel, IFRS 9).
    • KS statistic     — max CDF separation; natural cut-point for scoring
                         bands; legacy metric in US consumer lending.
    • Brier score      — proper scoring rule; penalises bad calibration and
                         discrimination jointly; use when predicted probabilities
                         feed downstream pricing or provisioning models.
    • Log-loss / NLL   — another proper rule; differentiable, useful when
                         the model is used inside a larger optimisation loop.
    • Partial AUC      — restrict ROC integration to low FPR region; prefer
                         when false-positive costs dominate (fraud, collections).
    • AUCPR            — precision-recall AUC; strictly better than AUC-ROC
                         when positive-class prevalence is < ~5%.

    Returns
    -------
    dict with keys: selected_features, dropped_features, n_selected,
    n_dropped, n_collinear_dropped, n_no_gain_dropped, final_metrics,
    step_results, vif_report, config.
    """
    from sklearn.model_selection import StratifiedKFold, cross_val_predict
    from sklearn.metrics import roc_auc_score, brier_score_loss, roc_curve

    log("\nSTAGE 2b — FORWARD CV FEATURE SELECTION (primary: AUC-ROC)")

    y = df[TARGET]
    # Fill NaN with column median so LightGBM sees complete rows every fold.
    X_full = df[FEATURES].copy().fillna(df[FEATURES].median())
    m_results = marginal_panel["feature_results"]

    # ── rank candidates by ascending initial VIF ─────────────────────
    # Ordering is intentionally independent of marginal stability scores
    # so that Stage 2b is orthogonal to Stage 2.  Features with low VIF
    # in the full feature matrix are most orthogonal to the rest; they
    # enter first and form the stable core of the selected set.
    log("  Computing initial VIF for candidate ordering ...")
    initial_vif = {
        row["feature"]: (row["vif"] if row["vif"] is not None else np.inf)
        for row in _compute_vif(X_full)
    }
    candidates = sorted(FEATURES, key=lambda f: initial_vif.get(f, np.inf))
    log(f"  {len(candidates)} candidates ordered by ascending VIF "
        f"(low VIF = orthogonal, enters first)")

    cv = StratifiedKFold(n_splits=FWD_CV_FOLDS, shuffle=True, random_state=42)

    def _cv_metrics(feature_set: list) -> dict:
        """Cross-validated AUC, Gini, KS, Brier for a feature set."""
        if not feature_set:
            return {"auc": 0.5, "gini": 0.0, "ks": 0.0, "brier": 1.0}
        X_sel = X_full[feature_set].values
        probas = cross_val_predict(
            lgbm_factory(), X_sel, y.values,
            cv=cv, method="predict_proba", n_jobs=1,
        )[:, 1]
        auc = float(roc_auc_score(y, probas))
        gini = 2.0 * auc - 1.0
        fpr, tpr, _ = roc_curve(y, probas)
        ks = float(np.max(np.abs(tpr - fpr)))
        brier = float(brier_score_loss(y, probas))
        return {"auc": auc, "gini": gini, "ks": ks, "brier": brier}

    def _incremental_vif(feat: str, selected: list) -> float:
        """
        VIF of feat when tentatively added to the current selected set.

        Regresses feat on the joint linear span of selected features and
        returns VIF = 1 / (1 − R²).  Captures multivariate collinearity:
        a high value means selected already explains most of feat's
        variance, so feat contributes little orthogonal signal.

        VIF > 5  → moderate collinearity (requires AUC premium to admit)
        VIF > 10 → severe collinearity
        """
        if not selected:
            return 1.0
        X_others = X_full[selected].values.astype(float)
        x_feat = X_full[feat].values.astype(float)
        # Centre for numerical stability
        X_others = X_others - X_others.mean(axis=0)
        x_feat = x_feat - x_feat.mean()
        ss_tot = float(np.dot(x_feat, x_feat))
        if ss_tot == 0.0:
            return 1.0
        if np.linalg.matrix_rank(X_others) < X_others.shape[1]:
            return np.inf
        coef, _, _, _ = np.linalg.lstsq(X_others, x_feat, rcond=None)
        ss_res = float(np.sum((x_feat - X_others @ coef) ** 2))
        r2 = max(0.0, min(1.0 - ss_res / ss_tot, 1.0 - 1e-9))
        return 1.0 / (1.0 - r2)

    # ── full-set baseline ─────────────────────────────────────────────
    # Evaluate the model on all candidates before any selection.  This
    # anchors the forward process: the final selected-set metrics can be
    # compared directly against the full-feature ceiling.
    log("  Evaluating full feature set (pre-selection baseline) ...")
    full_metrics = _cv_metrics(candidates)
    log(f"  Full set  ({len(candidates)} features): "
        f"AUC={full_metrics['auc']:.4f}  "
        f"Gini={full_metrics['gini']:.4f}  "
        f"KS={full_metrics['ks']:.4f}  "
        f"Brier={full_metrics['brier']:.4f}")

    # ── greedy forward pass ───────────────────────────────────────────
    selected: list = []
    dropped: list = []
    step_results: list = []

    baseline = _cv_metrics([])
    log(f"  Baseline (empty set): AUC={baseline['auc']:.4f}")
    step_results.append({
        "step": 0, "feature": None, "action": None,
        "selected_count": 0, "incremental_vif": None,
        "is_collinear": None, "auc_gain": None,
        **{f"cv_{k}": v for k, v in baseline.items()},
    })
    current = baseline

    for step_idx, feat in enumerate(candidates, start=1):
        inc_vif = _incremental_vif(feat, selected)
        is_collinear = np.isfinite(inc_vif) and inc_vif > FWD_VIF_THRESHOLD

        trial = _cv_metrics(selected + [feat])
        delta = trial["auc"] - current["auc"]
        required = FWD_MIN_AUC_GAIN + (FWD_COLLINEAR_AUC_PREMIUM if is_collinear else 0.0)

        vif_str = f"{inc_vif:.2f}" if np.isfinite(inc_vif) else "inf"

        if delta >= required:
            selected.append(feat)
            current = trial
            action = "add"
            log(f"  [{step_idx:2d}] ADD  {feat:<35s}"
                f"  AUC={trial['auc']:.4f}  Δ={delta:+.4f}  VIF={vif_str}")
        else:
            reason = (
                f"collinear|VIF={vif_str}"
                if is_collinear
                else f"no_gain|Δ={delta:.4f}"
            )
            dropped.append({
                "feature": feat,
                "reason": reason,
                "incremental_vif": float(inc_vif) if np.isfinite(inc_vif) else None,
                "auc_gain": delta,
                "is_collinear": is_collinear,
                "marginal_complexity": float(
                    m_results.get(feat, {}).get("complexity_score", np.nan)
                ),
            })
            action = "drop"
            log(f"  [{step_idx:2d}] DROP {feat:<35s}"
                f"  Δ={delta:+.4f}  VIF={vif_str}  [{reason}]")

        step_results.append({
            "step": step_idx,
            "feature": feat,
            "action": action,
            "selected_count": len(selected),
            "incremental_vif": float(inc_vif) if np.isfinite(inc_vif) else None,
            "is_collinear": bool(is_collinear),
            "auc_gain": float(delta),
            **{f"cv_{k}": v for k, v in trial.items()},
        })

    # ── final metrics & diagnostics ───────────────────────────────────
    final = _cv_metrics(selected)
    n_coll = sum(1 for d in dropped if d["is_collinear"])
    n_gain = sum(1 for d in dropped if not d["is_collinear"])

    log(f"\n  Selected {len(selected)}/{len(candidates)} features")
    log(f"  AUC={final['auc']:.4f}  Gini={final['gini']:.4f}"
        f"  KS={final['ks']:.4f}  Brier={final['brier']:.4f}")
    log(f"  Dropped: {n_coll} collinear, {n_gain} no-gain")

    vif_report = _compute_vif(X_full[selected]) if selected else []

    result = {
        "selected_features": selected,
        "dropped_features": dropped,
        "n_selected": len(selected),
        "n_dropped": len(dropped),
        "n_collinear_dropped": n_coll,
        "n_no_gain_dropped": n_gain,
        "full_set_metrics": full_metrics,
        "final_metrics": final,
        "step_results": step_results,
        "vif_report": vif_report,
        "config": {
            "cv_folds": FWD_CV_FOLDS,
            "vif_threshold": FWD_VIF_THRESHOLD,
            "min_auc_gain": FWD_MIN_AUC_GAIN,
            "collinear_auc_premium": FWD_COLLINEAR_AUC_PREMIUM,
            "candidate_ordering": "ascending_initial_vif",
            "collinearity_check": "incremental_vif",
            "optimisation_metric": "auc",
            "alternative_metrics": {
                "gini": "2*AUC-1; same ordering, preferred in regulatory scorecards",
                "ks":   "max CDF separation; legacy metric in US consumer lending",
                "brier": "proper scoring rule; penalises mis-calibration alongside poor discrimination",
            },
        },
    }

    save_json(result, "forward_cv_selection.json")
    _plot_forward_selection_curve(step_results, OUTPUT_DIR)
    return result


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
# STAGE 8b — TEMPORAL VALIDATION (out-of-time)
# ═══════════════════════════════════════════════════════════════════════
def run_temporal_validation(data_path, marginal_panel):
    """Validate that dev-period complexity scores predict holdout-period degradation."""
    log("\nSTAGE 8b — TEMPORAL VALIDATION (out-of-time)")
    from scipy.stats import spearmanr, mannwhitneyu
    from bootstrap_stability.core import (
        MetricRunner, CategoricalMetricRunner, detect_feature_type,
        DEFAULT_WEIGHTS,
    )

    # ── Step 1: load temporal split ──────────────────────────────────
    dev_feats, dev_target, ho_feats, ho_target, _ = load_lending_club_temporal(
        data_path, dev_end="Dec-2016", holdout_start="Jan-2017",
        sample_n=SAMPLE_N, random_state=42,
    )

    # Align columns (some may have been dropped differently per period)
    common_cols = sorted(set(dev_feats.columns) & set(ho_feats.columns))
    dev_feats = dev_feats[common_cols]
    ho_feats = ho_feats[common_cols]
    log(f"  {len(common_cols)} common features across periods")

    # ── Step 2: compute holdout-vs-dev metrics per feature ───────────
    log("  Computing temporal shift metrics ...")
    temporal_metrics = {}
    for feat in common_cols:
        x_dev = dev_feats[feat].values
        y_dev = dev_target.values
        x_ho = ho_feats[feat].values
        y_ho = ho_target.values

        feat_type = detect_feature_type(x_dev)
        if feat_type == "categorical":
            runner = CategoricalMetricRunner(x_dev, y_dev)
        else:
            runner = MetricRunner(x_dev, y_dev)

        result = runner(x_ho, y_ho)
        temporal_metrics[feat] = result

    # ── Step 3: build comparison dataframe ────────────────────────────
    dist_w = DEFAULT_WEIGHTS["distributional"]
    td_w = DEFAULT_WEIGHTS["target_dependent"]

    rows = []
    m_results = marginal_panel["feature_results"]
    for feat in common_cols:
        if feat not in m_results:
            continue
        mr = m_results[feat]
        floors = mr.get("per_metric_floors", {})
        tm = temporal_metrics[feat]

        # Compute holdout composite using same weighting as complexity score
        total_w = 0.0
        weighted_sum = 0.0
        for metric, w in {**dist_w, **td_w}.items():
            val = tm.get(metric)
            if val is not None and np.isfinite(val):
                weighted_sum += w * abs(val)
                total_w += w
        holdout_composite = weighted_sum / total_w if total_w > 0 else np.nan

        rows.append({
            "feature": feat,
            "complexity_score": mr["complexity_score"],
            "wasserstein_floor": floors.get("wasserstein", np.nan),
            "spearman_floor": floors.get("spearman", np.nan),
            "iv_floor": floors.get("iv", np.nan),
            "ks_floor": floors.get("ks", np.nan),
            "js_floor": floors.get("js", np.nan),
            "holdout_wasserstein": tm.get("wasserstein", np.nan),
            "holdout_ks": tm.get("ks", np.nan),
            "holdout_js": tm.get("js", np.nan),
            "holdout_spearman": tm.get("spearman", np.nan),
            "holdout_iv": tm.get("iv", np.nan),
            "holdout_composite": holdout_composite,
        })

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUT_DIR / "temporal_validation_comparison.csv", index=False)

    # ── Step 4: validation statistics ─────────────────────────────────
    valid = comp_df.dropna(subset=["complexity_score", "holdout_composite"])

    # Primary correlation
    rho_overall, p_overall = spearmanr(valid["complexity_score"], valid["holdout_composite"])
    log(f"  Overall: rho={rho_overall:.3f}  p={p_overall:.4f}")

    # Per-metric correlations
    per_metric_corr = {}
    for metric in ["wasserstein", "ks", "js", "spearman", "iv"]:
        floor_col = f"{metric}_floor"
        ho_col = f"holdout_{metric}"
        sub = valid.dropna(subset=[floor_col, ho_col])
        if len(sub) >= 5:
            rho, p = spearmanr(sub[floor_col], sub[ho_col])
            per_metric_corr[metric] = {"rho": float(rho), "p_value": float(p), "n": len(sub)}
            log(f"    {metric}: rho={rho:.3f}  p={p:.4f}  n={len(sub)}")
        else:
            per_metric_corr[metric] = {"rho": np.nan, "p_value": np.nan, "n": len(sub)}

    # Group test: stable vs unstable
    threshold = valid["complexity_score"].median()
    stable = valid[valid["complexity_score"] <= threshold]
    unstable = valid[valid["complexity_score"] > threshold]

    if len(stable) >= 3 and len(unstable) >= 3:
        U, p_group = mannwhitneyu(
            unstable["holdout_composite"], stable["holdout_composite"],
            alternative="greater",
        )
        group_result = {
            "threshold": float(threshold),
            "stable_mean": float(stable["holdout_composite"].mean()),
            "unstable_mean": float(unstable["holdout_composite"].mean()),
            "n_stable": len(stable),
            "n_unstable": len(unstable),
            "U_statistic": float(U),
            "p_value": float(p_group),
        }
        log(f"  Group test: stable={group_result['stable_mean']:.4f}  "
            f"unstable={group_result['unstable_mean']:.4f}  "
            f"p={p_group:.4f}")
    else:
        group_result = {"error": "too few features in one group"}

    # ── Step 5: figures ───────────────────────────────────────────────
    # Scatter: complexity_score vs holdout_composite
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(valid["complexity_score"], valid["holdout_composite"],
               alpha=0.7, edgecolors="k", linewidths=0.5)
    for _, row in valid.iterrows():
        ax.annotate(row["feature"], (row["complexity_score"], row["holdout_composite"]),
                    fontsize=5, alpha=0.6)
    ax.set_xlabel("Dev-Period Complexity Score (floor)")
    ax.set_ylabel("Holdout-Period Metric Degradation")
    ax.set_title(f"Temporal Validation: rho={rho_overall:.3f}  p={p_overall:.4f}")
    # Trend line
    if len(valid) >= 5:
        z = np.polyfit(valid["complexity_score"], valid["holdout_composite"], 1)
        x_line = np.linspace(valid["complexity_score"].min(), valid["complexity_score"].max(), 50)
        ax.plot(x_line, np.polyval(z, x_line), "r--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "fig_temporal_scatter.png"), dpi=150)
    plt.close(fig)
    log(f"  saved  {OUTPUT_DIR / 'fig_temporal_scatter.png'}")

    # Box plot: stable vs unstable
    fig, ax = plt.subplots(figsize=(6, 5))
    box_data = [stable["holdout_composite"].values, unstable["holdout_composite"].values]
    bp = ax.boxplot(box_data, labels=["Stable", "Unstable"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ecc71")
    bp["boxes"][1].set_facecolor("#e74c3c")
    if "p_value" in group_result:
        ax.set_title(f"Holdout Degradation by Stability Group  (p={group_result['p_value']:.4f})")
    else:
        ax.set_title("Holdout Degradation by Stability Group")
    ax.set_ylabel("Holdout Composite Metric")
    fig.tight_layout()
    fig.savefig(str(OUTPUT_DIR / "fig_temporal_boxplot.png"), dpi=150)
    plt.close(fig)
    log(f"  saved  {OUTPUT_DIR / 'fig_temporal_boxplot.png'}")

    # ── Assemble results ──────────────────────────────────────────────
    temporal_results = {
        "dev_period": "2007-01 to 2016-12",
        "holdout_period": "2017-01 to 2018-12",
        "n_features": len(valid),
        "overall_correlation": {"rho": float(rho_overall), "p_value": float(p_overall)},
        "per_metric_correlations": per_metric_corr,
        "group_test": group_result,
        "per_feature": comp_df.to_dict(orient="records"),
    }
    save_json(temporal_results, "temporal_validation_results.json")
    return temporal_results


# ═══════════════════════════════════════════════════════════════════════
# STAGE 9 — SYNTHESIS
# ═══════════════════════════════════════════════════════════════════════
def write_synthesis(
    eda, marginal_panel, shap_panel, perm_results,
    ci_results, comparison, synthetic_results, drift,
    feature_groups, temporal_results=None, fwd_result=None,
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

    # Forward CV feature selection
    if fwd_result is not None:
        w("## Forward CV Feature Selection (Stage 2b)")
        w()
        cfg = fwd_result.get("config", {})
        fm = fwd_result.get("final_metrics", {})
        w(f"**Optimisation metric**: {cfg.get('optimisation_metric', 'auc').upper()} "
          f"(CV folds={cfg.get('cv_folds', '?')})")
        w()
        w(f"**Candidate ordering**: {cfg.get('candidate_ordering', '?')} "
          f"(orthogonal to marginal stability)")
        w()
        w(f"**Collinearity check**: incremental VIF > {cfg.get('vif_threshold', '?')} "
          f"requires +{cfg.get('collinear_auc_premium', '?'):.3f} ΔAUC premium")
        w()
        w(f"**Result**: {fwd_result['n_selected']} features selected, "
          f"{fwd_result['n_dropped']} dropped "
          f"({fwd_result['n_collinear_dropped']} collinear, "
          f"{fwd_result['n_no_gain_dropped']} no-gain)")
        w()
        full_m = fwd_result.get("full_set_metrics", {})
        w("| Metric | Full set (pre-selection) | Selected set |")
        w("|--------|--------------------------|--------------|")
        for key, label in [("auc", "AUC-ROC"), ("gini", "Gini"),
                           ("ks", "KS stat"), ("brier", "Brier ↓")]:
            w(f"| {label} "
              f"| {full_m.get(key, float('nan')):.4f} "
              f"| {fm.get(key, float('nan')):.4f} |")
        w()
        w("### Selected features")
        w()
        for f in fwd_result.get("selected_features", []):
            w(f"- {f}")
        w()
        dropped = fwd_result.get("dropped_features", [])
        if dropped:
            w("### Dropped features")
            w()
            w("| Feature | Reason | Incremental VIF | ΔAUC | Marginal Complexity |")
            w("|---------|--------|----------------|------|---------------------|")
            for d in dropped:
                iv = d.get("incremental_vif")
                iv_str = f"{iv:.2f}" if iv is not None else "∞"
                w(f"| {d['feature']} | {d['reason']} "
                  f"| {iv_str} "
                  f"| {d['auc_gain']:+.4f} "
                  f"| {d.get('marginal_complexity', float('nan')):.4f} |")
        w()
        vif = fwd_result.get("vif_report", [])
        if vif:
            w("### VIF — selected feature set")
            w()
            w("| Feature | VIF |")
            w("|---------|-----|")
            for row in vif:
                vif_val = f"{row['vif']:.2f}" if row["vif"] is not None else "∞"
                w(f"| {row['feature']} | {vif_val} |")
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
    w("| Feature | SHAP Complexity | Constant SHAP | Direction Consistency | Rank Stability |")
    w("|---------|----------------|---------------|----------------------|----------------|")
    for _, row in s_summary.iterrows():
        const_flag = "yes" if row.get("constant_shap", False) else ""
        w(f"| {row['feature']} | {row['complexity_score']:.3f} | "
          f"{const_flag} | "
          f"{row.get('direction_consistency', float('nan')):.3f} | "
          f"{row.get('rank_stability', float('nan')):.3f} |")
    n_const = s_summary["constant_shap"].sum() if "constant_shap" in s_summary.columns else 0
    if n_const > 0:
        const_list = s_summary.loc[s_summary["constant_shap"] == True, "feature"].tolist()
        w()
        w(f"*{n_const} features have constant SHAP values (model assigns no importance): "
          f"{', '.join(const_list)}. Stability metrics for these are degenerate.*")
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
    w("## Train/Holdout Drift (Random Split)")
    w()
    grade = drift.get("drift_grade", "?")
    score = drift.get("overall_drift_score", float("nan"))
    w(f"Grade: **{grade}** (score = {score:.4f})")
    w()

    # Temporal validation
    if temporal_results is not None:
        w("## Temporal Validation (Out-of-Time)")
        w()
        w(f"**Development**: {temporal_results['dev_period']} | "
          f"**Holdout**: {temporal_results['holdout_period']}")
        w()
        oc = temporal_results["overall_correlation"]
        w(f"Spearman correlation between dev-period complexity score and "
          f"holdout-period degradation: **rho = {oc['rho']:.3f}** (p = {oc['p_value']:.4f})")
        w()
        w("### Per-metric floor-to-degradation correlations")
        w()
        w("| Metric | rho | p-value | n |")
        w("|--------|-----|---------|---|")
        for metric, mc in temporal_results["per_metric_correlations"].items():
            w(f"| {metric} | {mc['rho']:.3f} | {mc['p_value']:.4f} | {mc['n']} |")
        w()
        gt = temporal_results["group_test"]
        if "error" not in gt:
            w("### Stable vs Unstable group comparison")
            w()
            w(f"Threshold (median complexity): {gt['threshold']:.4f}")
            w()
            w(f"| Group | n | Mean Holdout Degradation |")
            w(f"|-------|---|--------------------------|")
            w(f"| Stable | {gt['n_stable']} | {gt['stable_mean']:.4f} |")
            w(f"| Unstable | {gt['n_unstable']} | {gt['unstable_mean']:.4f} |")
            w()
            w(f"Mann-Whitney U (one-sided): p = {gt['p_value']:.4f}")
            w()
            if gt["p_value"] < 0.05:
                w("*Features flagged as structurally unstable in the development period "
                  "show significantly higher metric degradation in the holdout period. "
                  "The toolkit's pre-deployment diagnostic is a validated temporal prediction.*")
            else:
                w("*The group difference is not statistically significant at p < 0.05. "
                  "This may reflect limited feature count or the specific temporal split chosen.*")
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

    # Stage 2b — Forward CV feature selection
    if start <= 2:
        fwd_result = run_forward_cv_selection(df, TARGET, FEATURES, marginal_panel)
    else:
        fcs_path = OUTPUT_DIR / "forward_cv_selection.json"
        if fcs_path.exists():
            fwd_result = json.loads(fcs_path.read_text())
        else:
            log("\n  [stage 2b artefact missing — re-running forward CV selection]")
            fwd_result = run_forward_cv_selection(df, TARGET, FEATURES, marginal_panel)

    # Propagate selected feature set to all downstream stages
    if fwd_result and fwd_result.get("selected_features"):
        FEATURES = fwd_result["selected_features"]
        log(f"\n  Downstream stages use {len(FEATURES)} forward-selected features "
            f"(dropped {fwd_result['n_dropped']} — "
            f"{fwd_result['n_collinear_dropped']} collinear, "
            f"{fwd_result['n_no_gain_dropped']} no-gain)")

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

    # Stage 8b — Temporal validation
    if start <= 8:
        temporal_results = run_temporal_validation(args.data_path, marginal_panel)
    else:
        tv_path = OUTPUT_DIR / "temporal_validation_results.json"
        temporal_results = json.loads(tv_path.read_text()) if tv_path.exists() else None

    # Stage 9
    write_synthesis(
        eda, marginal_panel, shap_panel, perm_results,
        ci_results, comparison, synthetic_results, drift,
        feature_groups, temporal_results, fwd_result,
    )

    log()
    log("=" * 70)
    log(f"DONE — all outputs in {OUTPUT_DIR.absolute()}")
    log("=" * 70)


if __name__ == "__main__":
    main()
