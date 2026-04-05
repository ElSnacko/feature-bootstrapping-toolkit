"""
Comprehensive Feature Analysis: Instability & Impact Methods
=============================================================

This script implements ALL feature instability/impact methods (except drop-column
importance) on the credit card default dataset. It produces a master comparison
table, correlation analysis, reliability scoring, and a key findings summary.

Sections:
  1.  Data Loading & Model Training
  2.  LightGBM Feature Importance (Split + Gain)
  3.  Gain vs Split Divergence
  4.  Feature Tree Coverage
  5.  Feature Depth Statistics
  6.  Permutation Importance
  7.  SHAP Analysis
  8.  SHAP Value Variance & Coefficient of Variation
  9.  SHAP Interaction Values
  10. SHAP Distribution Shape Analysis
  11. SHAP Entropy / Concentration
  12. Cross-Seed SHAP Consistency
  13. Bootstrap Stability Toolkit (Marginal)
  13b. OOT Bootstrap Stability Comparison (Marginal)
  14. Master Comparison Table
  15. Correlation Analysis
  16. Feature Reliability Scoring
  17. Key Findings Summary
  
  NEW SHAP Stability Sections:
  18. SHAP Stability Learning Curves
  19. Train/OOT Drift Analysis
  20. Marginal vs SHAP Comparison Summary

All output files are saved in the feature-bootstrapping-toolkit/ directory.
"""

import sys
import os
import time
import warnings

sys.path.insert(0, ".")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Handle imports gracefully
try:
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    from sklearn.inspection import permutation_importance
except ImportError:
    print("ERROR: scikit-learn is required. Install with: pip install scikit-learn")
    sys.exit(1)

try:
    from lightgbm import LGBMClassifier
except ImportError:
    print("ERROR: lightgbm is required. Install with: pip install lightgbm")
    sys.exit(1)

try:
    import shap
except ImportError:
    print("ERROR: shap is required. Install with: pip install shap")
    sys.exit(1)

try:
    from scipy.stats import spearmanr, skew, kurtosis
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    from bootstrap_stability import BootstrapStability
except ImportError:
    print("WARNING: bootstrap_stability not available. Section 13 will be skipped.")
    BootstrapStability = None

try:
    from bootstrap_stability import (
        SHAPStability,
        TrainOOTStability,
        SHAPMetricRunner,
        aggregate_shap_metrics,
        get_drift_grade,
    )
    SHAP_STABILITY_AVAILABLE = True
except ImportError as e:
    print(f"WARNING: SHAP stability modules not available. Sections 18-20 will be skipped. Error: {e}")
    SHAP_STABILITY_AVAILABLE = False
    SHAPStability = None
    TrainOOTStability = None

warnings.filterwarnings("ignore")

# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = os.path.join(
    "..", "default+of+credit+card+clients", "default of credit card clients.xls"
)
OUTPUT_DIR = "credit_card_analysis_results"
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "default payment next month"
DROP_COLS = ["ID"]

SEEDS = [42, 123, 456, 789, 1024]

LGBM_PARAMS = dict(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=RANDOM_STATE,
    verbose=-1,
)


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def section_timer(func):
    """Decorator to time a section and print elapsed time."""
    def wrapper(*args, **kwargs):
        print_header(func.__name__.replace("_", " ").upper())
        t0 = time.time()
        result = func(*args, **kwargs)
        elapsed = time.time() - t0
        print(f"\n  [{func.__name__}] completed in {elapsed:.1f}s")
        return result
    return wrapper


# =============================================================================
# Section 1: Data Loading & Model Training
# =============================================================================
@section_timer
def section01_load_and_train():
    """Load data, split, train LightGBM, and report performance."""
    df = pd.read_excel(DATA_PATH, header=1)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")

    df = df.drop(columns=DROP_COLS)
    print(f"After dropping {DROP_COLS}: {df.shape[1]} columns remaining")
    print(f"Columns: {list(df.columns)}")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")
    print(f"Target distribution (train): {dict(y_train.value_counts())}")

    model = LGBMClassifier(**LGBM_PARAMS)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)
    print(f"\nModel Performance:")
    print(f"  Accuracy : {acc:.4f}")
    print(f"  ROC-AUC  : {auc:.4f}")

    # Hold-out set (test features, no target) for SHAP
    X_holdout = X_test.copy()

    return {
        "model": model,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "X_holdout": X_holdout,
        "feature_names": feature_names,
        "df_train": pd.concat([X_train, y_train], axis=1),
    }


# =============================================================================
# Section 2: LightGBM Feature Importance (Split + Gain)
# =============================================================================
@section_timer
def section02_lgbm_importance(model, feature_names):
    """Extract and normalize split and gain importances."""
    raw_split = model.booster_.feature_importance("split")
    raw_gain = model.booster_.feature_importance("gain")

    split_norm = raw_split / raw_split.sum()
    gain_norm = raw_gain / raw_gain.sum()

    df_imp = pd.DataFrame({
        "feature": feature_names,
        "lgbm_split_raw": raw_split,
        "lgbm_split_norm": split_norm,
        "lgbm_gain_raw": raw_gain,
        "lgbm_gain_norm": gain_norm,
    })
    df_imp["lgbm_split_rank"] = df_imp["lgbm_split_norm"].rank(ascending=False).astype(int)
    df_imp["lgbm_gain_rank"] = df_imp["lgbm_gain_norm"].rank(ascending=False).astype(int)
    df_imp = df_imp.sort_values("lgbm_gain_norm", ascending=False).reset_index(drop=True)

    print("LightGBM Feature Importance (sorted by gain):")
    print(df_imp[["feature", "lgbm_gain_norm", "lgbm_gain_rank",
                   "lgbm_split_norm", "lgbm_split_rank"]].to_string(index=False))

    return df_imp


# =============================================================================
# Section 3: Gain vs Split Divergence
# =============================================================================
@section_timer
def section03_gain_split_divergence(df_imp):
    """Compute gain minus split divergence and label features."""
    df_imp = df_imp.copy()
    df_imp["gain_split_divergence"] = df_imp["lgbm_gain_norm"] - df_imp["lgbm_split_norm"]

    def label_divergence(val):
        if val > 0.005:
            return "Concentrated"
        elif val < -0.005:
            return "Diluted"
        else:
            return "Balanced"

    df_imp["divergence_label"] = df_imp["gain_split_divergence"].apply(label_divergence)
    df_div = df_imp.sort_values("gain_split_divergence", ascending=False).reset_index(drop=True)

    print("Gain vs Split Divergence (sorted by divergence):")
    print(df_div[["feature", "lgbm_gain_norm", "lgbm_split_norm",
                   "gain_split_divergence", "divergence_label"]].to_string(index=False))

    return df_div[["feature", "gain_split_divergence"]]


# =============================================================================
# Section 4: Feature Tree Coverage
# =============================================================================
@section_timer
def section04_tree_coverage(model, feature_names):
    """Compute what fraction of trees use each feature."""
    tree_df = model.booster_.trees_to_dataframe()
    total_trees = tree_df["tree_index"].nunique()

    feature_tree_count = {}
    for feat in feature_names:
        trees_using = tree_df[tree_df["split_feature"] == feat]["tree_index"].nunique()
        feature_tree_count[feat] = trees_using / total_trees

    df_cov = pd.DataFrame({
        "feature": feature_names,
        "tree_coverage": [feature_tree_count[f] for f in feature_names],
    })
    df_cov = df_cov.sort_values("tree_coverage", ascending=False).reset_index(drop=True)

    print(f"Total trees: {total_trees}")
    print("Feature Tree Coverage:")
    print(df_cov.to_string(index=False))

    return df_cov, tree_df


# =============================================================================
# Section 5: Feature Depth Statistics
# =============================================================================
@section_timer
def section05_depth_statistics(tree_df, feature_names):
    """Compute mean, median, min depth for each feature."""
    split_nodes = tree_df[tree_df["split_feature"].notna()].copy()

    depth_stats = {}
    for feat in feature_names:
        depths = split_nodes[split_nodes["split_feature"] == feat]["node_depth"]
        if len(depths) > 0:
            depth_stats[feat] = {
                "mean_depth": depths.mean(),
                "median_depth": depths.median(),
                "min_depth": depths.min(),
                "max_depth": depths.max(),
            }
        else:
            depth_stats[feat] = {
                "mean_depth": np.nan,
                "median_depth": np.nan,
                "min_depth": np.nan,
                "max_depth": np.nan,
            }

    df_depth = pd.DataFrame([
        {"feature": feat, **depth_stats[feat]} for feat in feature_names
    ])
    df_depth = df_depth.sort_values("mean_depth").reset_index(drop=True)

    print("Feature Depth Statistics (sorted by mean depth):")
    print(df_depth.to_string(index=False))

    return df_depth


# =============================================================================
# Section 6: Permutation Importance
# =============================================================================
@section_timer
def section06_permutation_importance_(model, X_test, y_test, feature_names):
    """Compute permutation importance with 95% CI."""
    result = permutation_importance(
        model, X_test, y_test,
        n_repeats=10,
        scoring="roc_auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    df_perm = pd.DataFrame({
        "feature": feature_names,
        "perm_importance_mean": result.importances_mean,
        "perm_importance_std": result.importances_std,
    })
    df_perm["perm_importance_ci_lower"] = (
        result.importances_mean - 1.96 * result.importances_std
    )
    df_perm["perm_importance_ci_upper"] = (
        result.importances_mean + 1.96 * result.importances_std
    )
    df_perm = df_perm.sort_values("perm_importance_mean", ascending=False).reset_index(drop=True)

    print("Permutation Importance (sorted by mean):")
    print(df_perm[["feature", "perm_importance_mean", "perm_importance_std",
                    "perm_importance_ci_lower", "perm_importance_ci_upper"]].to_string(index=False))

    return df_perm


# =============================================================================
# Section 7: SHAP Analysis
# =============================================================================
@section_timer
def section07_shap_analysis(model, X_holdout, feature_names):
    """Compute SHAP values and mean |SHAP| per feature."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_holdout)

    # For binary classification, shap_values may be a list [class0, class1]
    if isinstance(shap_values, list):
        sv = shap_values[1]  # positive class
    else:
        sv = shap_values

    mean_abs_shap = np.abs(sv).mean(axis=0)
    shap_norm = mean_abs_shap / mean_abs_shap.sum()

    df_shap = pd.DataFrame({
        "feature": feature_names,
        "shap_importance_raw": mean_abs_shap,
        "shap_importance_norm": shap_norm,
    })
    df_shap["shap_rank"] = df_shap["shap_importance_norm"].rank(ascending=False).astype(int)
    df_shap = df_shap.sort_values("shap_importance_norm", ascending=False).reset_index(drop=True)

    print("SHAP Feature Importance (sorted by mean |SHAP|):")
    print(df_shap[["feature", "shap_importance_norm", "shap_rank"]].to_string(index=False))

    return df_shap, sv


# =============================================================================
# Section 8: SHAP Value Variance & Coefficient of Variation
# =============================================================================
@section_timer
def section08_shap_variance(sv, feature_names):
    """Compute per-feature SHAP std, mean|SHAP|, and CV."""
    df_var = pd.DataFrame({
        "feature": feature_names,
        "shap_std": np.std(sv, axis=0),
        "shap_mean_abs": np.mean(np.abs(sv), axis=0),
    })
    df_var["shap_cv"] = df_var["shap_std"] / df_var["shap_mean_abs"]
    df_var = df_var.sort_values("shap_cv").reset_index(drop=True)

    print("SHAP Variance & CV (sorted by CV, most stable first):")
    print(df_var.to_string(index=False))

    return df_var


# =============================================================================
# Section 9: SHAP Interaction Values
# =============================================================================
@section_timer
def section09_shap_interactions(model, X_holdout, feature_names):
    """Compute SHAP interaction values on a sample of hold-out set."""
    n_sample = min(500, len(X_holdout))
    X_sample = X_holdout.iloc[:n_sample]
    print(f"Computing SHAP interaction values on {n_sample} samples...")

    explainer = shap.TreeExplainer(model)
    shap_inter = explainer.shap_interaction_values(X_sample)

    # For binary classification, take positive class
    if isinstance(shap_inter, list):
        shap_inter = shap_inter[1]

    # Mean absolute interaction per feature (sum across all other features)
    n_features = shap_inter.shape[1]
    mean_abs_inter = np.zeros(n_features)
    for i in range(n_features):
        # For feature i, sum |interaction(i,j)| for all j != i, then average over samples
        interactions_no_diag = np.abs(shap_inter[:, i, :].copy())
        interactions_no_diag[:, i] = 0  # zero out main effect diagonal
        mean_abs_inter[i] = interactions_no_diag.sum(axis=1).mean()

    df_inter = pd.DataFrame({
        "feature": feature_names,
        "shap_total_interaction": mean_abs_inter,
    })
    df_inter = df_inter.sort_values(
        "shap_total_interaction", ascending=False
    ).reset_index(drop=True)

    # Top 10 feature pairs
    pair_data = []
    for i in range(n_features):
        for j in range(i + 1, n_features):
            mean_pair = np.abs(shap_inter[:, i, j]).mean()
            pair_data.append((feature_names[i], feature_names[j], mean_pair))

    pair_data.sort(key=lambda x: x[2], reverse=True)
    print("\nTop 10 Feature Pairs by Interaction Strength:")
    for rank, (fi, fj, val) in enumerate(pair_data[:10], 1):
        print(f"  {rank:2d}. {fi} <-> {fj}: {val:.6f}")

    print("\nPer-Feature Total Interaction Strength:")
    print(df_inter.to_string(index=False))

    return df_inter


# =============================================================================
# Section 10: SHAP Distribution Shape Analysis
# =============================================================================
@section_timer
def section10_shap_distribution(sv, feature_names):
    """Compute skewness, kurtosis, bimodality coefficient for each feature's SHAP values."""
    rows = []
    for i, feat in enumerate(feature_names):
        vals = sv[:, i]
        s = skew(vals)
        k = kurtosis(vals, fisher=True)  # excess kurtosis
        # Bimodality coefficient: (skewness^2 + 1) / (kurtosis + 3)
        # Using excess kurtosis: BC = (s^2 + 1) / (k + 3)
        bc = (s ** 2 + 1) / (k + 3) if (k + 3) != 0 else np.nan
        rows.append({
            "feature": feat,
            "shap_skewness": s,
            "shap_kurtosis": k,
            "shap_bimodality_coeff": bc,
            "shap_bimodality_flag": bc > 0.555 if not np.isnan(bc) else False,
        })

    df_dist = pd.DataFrame(rows)
    df_dist = df_dist.sort_values(
        "shap_bimodality_coeff", ascending=False
    ).reset_index(drop=True)

    print("SHAP Distribution Shape Analysis (sorted by bimodality coefficient):")
    print(df_dist.to_string(index=False))

    return df_dist


# =============================================================================
# Section 11: SHAP Entropy / Concentration
# =============================================================================
@section_timer
def section11_shap_entropy(df_shap):
    """Compute entropy of the normalized mean |SHAP| distribution."""
    p = df_shap["shap_importance_norm"].values
    p = p[p > 0]  # filter zeros for log

    entropy = -np.sum(p * np.log(p))
    max_entropy = np.log(len(p))
    normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0

    print(f"SHAP Entropy: {entropy:.4f}")
    print(f"Max Entropy (uniform): {max_entropy:.4f}")
    print(f"Normalized Entropy: {normalized_entropy:.4f}")

    if normalized_entropy > 0.85:
        interpretation = "Distributed (features share importance relatively evenly)"
    elif normalized_entropy > 0.65:
        interpretation = "Moderately concentrated"
    else:
        interpretation = "Concentrated (a few features dominate the model)"

    print(f"Interpretation: {interpretation}")

    return {
        "entropy": entropy,
        "max_entropy": max_entropy,
        "normalized_entropy": normalized_entropy,
        "interpretation": interpretation,
    }


# =============================================================================
# Section 12: Cross-Seed SHAP Consistency
# =============================================================================
@section_timer
def section12_cross_seed_shap(X_train, y_train, X_holdout, feature_names):
    """Train 5 LightGBM models with different seeds and compare SHAP rankings."""
    seed_results = {}

    for idx, seed in enumerate(SEEDS):
        print(f"  Training model {idx + 1}/{len(SEEDS)} (seed={seed})...")
        params = dict(LGBM_PARAMS, random_state=seed)
        m = LGBMClassifier(**params)
        m.fit(X_train, y_train)

        explainer = shap.TreeExplainer(m)
        sv = explainer.shap_values(X_holdout)
        if isinstance(sv, list):
            sv = sv[1]

        mean_abs = np.abs(sv).mean(axis=0)
        ranks = pd.Series(mean_abs).rank(ascending=False).values

        seed_results[seed] = {
            "mean_abs_shap": mean_abs,
            "ranks": ranks,
        }

    # Aggregate across seeds
    rank_matrix = np.array([seed_results[s]["ranks"] for s in SEEDS])
    mean_ranks = rank_matrix.mean(axis=0)
    std_ranks = rank_matrix.std(axis=0)
    rank_range = rank_matrix.max(axis=0) - rank_matrix.min(axis=0)

    df_consistency = pd.DataFrame({
        "feature": feature_names,
        "cross_seed_mean_rank": mean_ranks,
        "cross_seed_std_rank": std_ranks,
        "cross_seed_rank_range": rank_range,
    })
    df_consistency = df_consistency.sort_values(
        "cross_seed_std_rank"
    ).reset_index(drop=True)

    print("Cross-Seed SHAP Consistency (sorted by std_rank, most consistent first):")
    print(df_consistency.to_string(index=False))

    return df_consistency


# =============================================================================
# Section 13: Bootstrap Stability Toolkit
# =============================================================================
@section_timer
def section13_bootstrap_stability(df_train, feature_names):
    """Run BootstrapStability.fit_panel() on training data."""
    if BootstrapStability is None:
        print("BootstrapStability not available. Skipping.")
        return pd.DataFrame({"feature": feature_names, "complexity_score": np.nan}), {}

    bs = BootstrapStability(n_resamples=15, random_state=RANDOM_STATE)

    print("Running BootstrapStability.fit_panel()...")
    print(f"  Features to analyze: {len(feature_names)}")
    panel_result = bs.fit_panel(df_train, target_col=TARGET_COL)

    summary = panel_result["summary"]
    print(f"\n  Successfully analyzed: {len(summary)} features")
    print(f"  Skipped (categorical/other): {len(feature_names) - len(summary)} features")

    if not summary.empty:
        print("\nBootstrap Stability Complexity Scores:")
        cols = ["feature", "complexity_score"]
        available_cols = [c for c in cols if c in summary.columns]
        print(summary[available_cols].to_string(index=False))

    return summary, panel_result


# =============================================================================
# Section 13b: OOT Bootstrap Stability Comparison
# =============================================================================
@section_timer
def section13b_oot_bootstrap_comparison(train_df, X_test_df, train_panel_results, feature_names):
    """
    Compare bootstrap stability between training and test (OOT) data.

    Args:
        train_df: Training DataFrame with target column
        X_test_df: Test DataFrame (features only, no target)
        train_panel_results: Results from Section 13's fit_panel on training data
        feature_names: List of all feature names

    Returns:
        dict with keys: 'oot_panel_results', 'comparison_df', 'shifted_features'
    """
    if BootstrapStability is None:
        print("BootstrapStability not available. Skipping OOT comparison.")
        return {
            "oot_panel_results": {},
            "comparison_df": pd.DataFrame(),
            "shifted_features": pd.DataFrame(),
        }

    # --- Run bootstrap stability on test (OOT) data (no target) ---
    print("Running BootstrapStability.fit_panel() on OOT (test) data...")
    print(f"  Features to analyze: {len(feature_names)}")
    bs_oot = BootstrapStability(n_resamples=15, random_state=RANDOM_STATE)
    oot_panel_result = bs_oot.fit_panel(X_test_df, target_col=None, feature_cols=feature_names)

    oot_summary = oot_panel_result["summary"]
    print(f"\n  Successfully analyzed (OOT): {len(oot_summary)} features")
    print(f"  Skipped (categorical/other): {len(feature_names) - len(oot_summary)} features")

    if not oot_summary.empty:
        print("\nOOT Bootstrap Stability Complexity Scores:")
        cols = ["feature", "complexity_score"]
        available_cols = [c for c in cols if c in oot_summary.columns]
        print(oot_summary[available_cols].to_string(index=False))

    # --- Build comparison between train and test ---
    train_summary = train_panel_results.get("summary", pd.DataFrame())
    if train_summary.empty or oot_summary.empty:
        print("\n  Cannot compare: one or both panel results are empty.")
        return {
            "oot_panel_results": oot_panel_result,
            "comparison_df": pd.DataFrame(),
            "shifted_features": pd.DataFrame(),
        }

    # Merge on feature
    comparison_df = train_summary[["feature", "complexity_score"]].rename(
        columns={"complexity_score": "train_complexity_score"}
    ).merge(
        oot_summary[["feature", "complexity_score"]].rename(
            columns={"complexity_score": "oot_complexity_score"}
        ),
        on="feature",
        how="inner",
    )

    # Compute shift: test - train (positive = degraded in test)
    comparison_df["oot_complexity_shift"] = (
        comparison_df["oot_complexity_score"] - comparison_df["train_complexity_score"]
    )
    comparison_df["abs_shift"] = comparison_df["oot_complexity_shift"].abs()
    comparison_df = comparison_df.sort_values(
        "abs_shift", ascending=False
    ).reset_index(drop=True)

    print("\nTrain vs OOT Complexity Comparison:")
    print(comparison_df[["feature", "train_complexity_score",
                         "oot_complexity_score", "oot_complexity_shift"]].to_string(index=False))

    # --- Visualization 1: Scatter plot train vs OOT complexity ---
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(
        comparison_df["train_complexity_score"],
        comparison_df["oot_complexity_score"],
        s=60, alpha=0.7, edgecolors="black", linewidth=0.5,
    )

    # Diagonal reference line
    all_vals = pd.concat([
        comparison_df["train_complexity_score"], comparison_df["oot_complexity_score"]
    ])
    lo, hi = all_vals.min(), all_vals.max()
    margin = (hi - lo) * 0.05
    ax.plot([lo - margin, hi + margin], [lo - margin, hi + margin],
            "k--", alpha=0.4, label="Perfect agreement")

    # Annotate points
    for _, row in comparison_df.iterrows():
        ax.annotate(
            row["feature"],
            (row["train_complexity_score"], row["oot_complexity_score"]),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("Train Complexity Score", fontsize=12)
    ax.set_ylabel("OOT (Test) Complexity Score", fontsize=12)
    ax.set_title("Train vs OOT Bootstrap Complexity", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(OUTPUT_DIR, "oot_complexity_scatter.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nOOT complexity scatter saved to: {output_path}")

    # --- Visualization 2: Bar chart of top 10 features by absolute shift ---
    top_shift = comparison_df.head(10).copy()
    top_shift = top_shift.sort_values("oot_complexity_shift", ascending=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = ["#d62728" if v > 0 else "#2ca02c" for v in top_shift["oot_complexity_shift"]]
    ax.barh(top_shift["feature"], top_shift["oot_complexity_shift"], color=colors, edgecolor="black")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_xlabel("Complexity Shift (OOT - Train)", fontsize=12)
    ax.set_ylabel("Feature", fontsize=12)
    ax.set_title("Top 10 Features by Absolute Complexity Shift (OOT vs Train)", fontsize=13)
    ax.grid(True, alpha=0.3, axis="x")

    output_path2 = os.path.join(OUTPUT_DIR, "oot_complexity_shift.png")
    fig.tight_layout()
    fig.savefig(output_path2, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"OOT complexity shift bar chart saved to: {output_path2}")

    # --- Print OOT insights ---
    n_features = len(comparison_df)
    print("\n" + "-" * 60)
    print("OOT BOOTSTRAP STABILITY INSIGHTS")
    print("-" * 60)

    # Features with largest stability DEGRADATION (positive shift)
    degraded = comparison_df.nlargest(5, "oot_complexity_shift")
    if not degraded.empty and degraded["oot_complexity_shift"].max() > 0:
        print("\n► Features with Largest Stability DEGRADATION in OOT (positive shift):")
        for _, row in degraded.iterrows():
            if row["oot_complexity_shift"] > 0:
                print(f"  - {row['feature']}: shift = +{row['oot_complexity_shift']:.4f} "
                      f"(train={row['train_complexity_score']:.4f} → oot={row['oot_complexity_score']:.4f})")

    # Features with largest stability IMPROVEMENT (negative shift)
    improved = comparison_df.nsmallest(5, "oot_complexity_shift")
    if not improved.empty and improved["oot_complexity_shift"].min() < 0:
        print("\n► Features with Largest Stability IMPROVEMENT in OOT (negative shift):")
        for _, row in improved.iterrows():
            if row["oot_complexity_shift"] < 0:
                print(f"  - {row['feature']}: shift = {row['oot_complexity_shift']:.4f} "
                      f"(train={row['train_complexity_score']:.4f} → oot={row['oot_complexity_score']:.4f})")

    # Features with consistent stability (small shift)
    median_abs_shift = comparison_df["abs_shift"].median()
    consistent = comparison_df[comparison_df["abs_shift"] <= median_abs_shift]
    if not consistent.empty:
        print(f"\n► Features with Consistent Stability (abs shift ≤ median {median_abs_shift:.4f}):")
        for _, row in consistent.iterrows():
            print(f"  - {row['feature']}: shift = {row['oot_complexity_shift']:+.4f}")

    print("\n► Interpretation:")
    print("  Features with large positive shifts are MORE UNSTABLE in test/OOT data,")
    print("  suggesting distributional drift that may degrade model performance in production.")
    print("  Features with negative shifts are MORE STABLE in OOT, indicating robust behavior.")
    print("  Features near zero shift show consistent stability across train and test.")

    return {
        "oot_panel_results": oot_panel_result,
        "comparison_df": comparison_df,
        "shifted_features": comparison_df,
    }


# =============================================================================
# Section 14: Master Comparison Table
# =============================================================================
@section_timer
def section14_master_table(
    feature_names,
    df_imp,
    df_divergence,
    df_coverage,
    df_depth,
    df_perm,
    df_shap,
    df_shap_var,
    df_inter,
    df_dist,
    df_consistency,
    df_bootstrap,
    oot_comparison=None,
):
    """Merge ALL metrics into a single DataFrame."""
    master = pd.DataFrame({"feature": feature_names})

    # LightGBM importance
    master = master.merge(
        df_imp[["feature", "lgbm_gain_norm", "lgbm_gain_rank",
                 "lgbm_split_norm", "lgbm_split_rank"]],
        on="feature", how="left",
    )

    # Gain vs Split divergence
    master = master.merge(df_divergence, on="feature", how="left")

    # Tree coverage
    master = master.merge(df_coverage[["feature", "tree_coverage"]], on="feature", how="left")

    # Depth statistics
    master = master.merge(df_depth[["feature", "mean_depth"]], on="feature", how="left")

    # Permutation importance
    master = master.merge(
        df_perm[["feature", "perm_importance_mean", "perm_importance_std"]],
        on="feature", how="left",
    )

    # SHAP importance
    master = master.merge(
        df_shap[["feature", "shap_importance_norm", "shap_rank"]],
        on="feature", how="left",
    )

    # SHAP CV
    master = master.merge(
        df_shap_var[["feature", "shap_cv"]],
        on="feature", how="left",
    )

    # SHAP interactions
    master = master.merge(
        df_inter[["feature", "shap_total_interaction"]],
        on="feature", how="left",
    )

    # SHAP distribution shape
    master = master.merge(
        df_dist[["feature", "shap_skewness", "shap_kurtosis", "shap_bimodality_flag"]],
        on="feature", how="left",
    )

    # Cross-seed consistency
    master = master.merge(
        df_consistency[["feature", "cross_seed_mean_rank", "cross_seed_std_rank"]],
        on="feature", how="left",
    )

    # Bootstrap stability
    if not df_bootstrap.empty and "complexity_score" in df_bootstrap.columns:
        bs_cols = ["feature", "complexity_score"]
        master = master.merge(
            df_bootstrap[bs_cols], on="feature", how="left",
        )
        master["complexity_rank"] = master["complexity_score"].rank(
            ascending=True, na_option="keep"
        )
        # Keep as float so NaN is preserved — cannot cast NaN to int
    else:
        master["complexity_score"] = np.nan
        master["complexity_rank"] = np.nan

    # OOT bootstrap stability
    if oot_comparison is not None and not oot_comparison.get("comparison_df", pd.DataFrame()).empty:
        oot_df = oot_comparison["comparison_df"][["feature", "oot_complexity_score", "oot_complexity_shift"]]
        master = master.merge(oot_df, on="feature", how="left")
    else:
        master["oot_complexity_score"] = np.nan
        master["oot_complexity_shift"] = np.nan

    # Sort by SHAP importance
    master = master.sort_values("shap_importance_norm", ascending=False).reset_index(drop=True)

    # Save to CSV
    output_path = os.path.join(OUTPUT_DIR, "master_comparison.csv")
    master.to_csv(output_path, index=False)
    print(f"Master comparison table saved to: {output_path}")
    print(f"Shape: {master.shape}")

    # Print full table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)
    print("\nMaster Comparison Table:")
    print(master.to_string(index=False))

    return master


# =============================================================================
# Section 15: Correlation Analysis
# =============================================================================
@section_timer
def section15_correlation_analysis(master):
    """Compute Spearman rank correlation matrix across ALL ranking metrics."""
    # Select ranking/numeric columns for correlation
    rank_cols = [
        "lgbm_gain_rank",
        "lgbm_split_rank",
        "tree_coverage",
        "mean_depth",
        "perm_importance_mean",
        "shap_rank",
        "shap_cv",
        "shap_total_interaction",
        "cross_seed_std_rank",
        "complexity_score",
        "oot_complexity_score",
        "oot_complexity_shift",
    ]

    # Filter to columns that exist and have enough non-null values
    available_cols = [c for c in rank_cols if c in master.columns]
    corr_data = master[available_cols].dropna()

    if len(corr_data) < 3:
        print("Not enough data for correlation analysis.")
        return

    corr_matrix = corr_data.corr(method="spearman")

    print("Spearman Rank Correlation Matrix:")
    print(corr_matrix.round(3).to_string())

    # Key correlations
    print("\nKey Correlations:")
    # Flatten upper triangle
    pairs = []
    cols = list(corr_matrix.columns)
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            pairs.append((cols[i], cols[j], corr_matrix.iloc[i, j]))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)

    print("\n  Highest Agreement (most correlated):")
    for c1, c2, r in pairs[:5]:
        print(f"    {c1} <-> {c2}: {r:.3f}")

    print("\n  Highest Disagreement (most negatively correlated):")
    neg_pairs = sorted(pairs, key=lambda x: x[2])
    for c1, c2, r in neg_pairs[:5]:
        print(f"    {c1} <-> {c2}: {r:.3f}")

    # Create heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    im = ax.imshow(corr_matrix.values, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")

    ax.set_xticks(range(len(cols)))
    ax.set_yticks(range(len(cols)))
    ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(cols, fontsize=9)

    # Annotate cells
    for i in range(len(cols)):
        for j in range(len(cols)):
            val = corr_matrix.values[i, j]
            color = "white" if abs(val) > 0.5 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=8, color=color)

    plt.colorbar(im, ax=ax, label="Spearman Correlation")
    ax.set_title("All Methods Correlation Heatmap", fontsize=14, pad=15)

    output_path = os.path.join(OUTPUT_DIR, "all_methods_correlation_heatmap.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nHeatmap saved to: {output_path}")


# =============================================================================
# Section 16: Feature Reliability Scoring
# =============================================================================
@section_timer
def section16_reliability_scoring(master):
    """Create composite reliability score for each feature."""
    df = master.copy()

    def normalize_to_01(series):
        """Min-max normalize a series to [0, 1]."""
        mn, mx = series.min(), series.max()
        if mx - mn == 0:
            return pd.Series(0.5, index=series.index)
        return (series - mn) / (mx - mn)

    # Component 1: Importance (high SHAP rank = low number = high importance)
    # Invert rank so higher = more important
    importance = normalize_to_01(df["shap_rank"].max() - df["shap_rank"] + 1)

    # Component 2: Stability (low CV, low cross-seed std, low complexity score)
    cv_inv = normalize_to_01(df["shap_cv"].max() - df["shap_cv"])  # invert
    seed_std_inv = normalize_to_01(
        df["cross_seed_std_rank"].max() - df["cross_seed_std_rank"]
    )  # invert
    complexity_inv = normalize_to_01(
        df["complexity_score"].max() - df["complexity_score"]
    ) if df["complexity_score"].notna().any() else pd.Series(0.5, index=df.index)
    stability = (cv_inv + seed_std_inv + complexity_inv) / 3.0

    # Component 3: Coverage
    coverage = normalize_to_01(df["tree_coverage"])

    # Component 4: Consistency (tight permutation CI = low std relative to mean)
    perm_cv = df["perm_importance_std"] / df["perm_importance_mean"].replace(0, np.nan)
    consistency = normalize_to_01(perm_cv.max() - perm_cv)  # invert

    # Weighted average
    w_imp = 0.3
    w_stab = 0.4
    w_cov = 0.15
    w_con = 0.15

    df["reliability_score"] = (
        w_imp * importance +
        w_stab * stability +
        w_cov * coverage +
        w_con * consistency
    )

    df = df.sort_values("reliability_score", ascending=False).reset_index(drop=True)

    print("Feature Reliability Ranking:")
    print(df[["feature", "reliability_score", "shap_importance_norm", "shap_rank",
              "shap_cv", "cross_seed_std_rank", "tree_coverage"]].to_string(index=False))

    # Scatter plot: reliability vs SHAP importance
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.scatter(
        df["shap_importance_norm"],
        df["reliability_score"],
        s=60, alpha=0.7, edgecolors="black", linewidth=0.5,
    )
    for _, row in df.iterrows():
        ax.annotate(
            row["feature"],
            (row["shap_importance_norm"], row["reliability_score"]),
            fontsize=7, alpha=0.8,
            xytext=(5, 5), textcoords="offset points",
        )

    ax.set_xlabel("SHAP Importance (normalized)", fontsize=12)
    ax.set_ylabel("Reliability Score", fontsize=12)
    ax.set_title("Feature Reliability vs SHAP Importance", fontsize=14)
    ax.grid(True, alpha=0.3)

    output_path = os.path.join(OUTPUT_DIR, "reliability_vs_importance.png")
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nScatter plot saved to: {output_path}")

    return df


# =============================================================================
# Section 17: Key Findings Summary
# =============================================================================
@section_timer
def section17_key_findings(master, df_reliability, entropy_info, df_dist, df_inter, oot_comparison=None):
    """Print comprehensive text summary of key findings."""
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)

    # Top 5 most reliable features
    print("\n► Top 5 Most Reliable Features:")
    for i, row in df_reliability.head(5).iterrows():
        print(f"  {i + 1}. {row['feature']}: reliability={row['reliability_score']:.3f}, "
              f"SHAP rank={row['shap_rank']:.0f}")

    # Top 5 most important but unreliable
    print("\n► Top 5 Most Important but Unreliable Features (high SHAP, low reliability):")
    df_reliability_sorted = df_reliability.sort_values("shap_rank").reset_index(drop=True)
    low_reliability = df_reliability_sorted.nlargest(20, "shap_rank")
    important_unreliable = low_reliability.nsmallest(5, "reliability_score")
    for i, (_, row) in enumerate(important_unreliable.iterrows()):
        print(f"  {i + 1}. {row['feature']}: SHAP rank={row['shap_rank']:.0f}, "
              f"reliability={row['reliability_score']:.3f}")

    # Top 5 most stable but unimportant
    print("\n► Top 5 Most Stable but Unimportant Features (low SHAP, high reliability):")
    unimportant = df_reliability.nlargest(10, "shap_rank")
    stable_unimportant = unimportant.nlargest(5, "reliability_score")
    for i, (_, row) in enumerate(stable_unimportant.iterrows()):
        print(f"  {i + 1}. {row['feature']}: SHAP rank={row['shap_rank']:.0f}, "
              f"reliability={row['reliability_score']:.3f}")

    # Methods with highest/lowest agreement
    print("\n► Methods Agreement:")
    print(f"  Overall model concentration: {entropy_info['interpretation']}")
    print(f"  Normalized entropy: {entropy_info['normalized_entropy']:.4f}")

    # Features flagged as bimodal
    bimodal = df_dist[df_dist["shap_bimodality_flag"] == True]
    if len(bimodal) > 0:
        print(f"\n► Features Flagged as Bimodal (potential subpopulations): {len(bimodal)}")
        for _, row in bimodal.iterrows():
            print(f"  - {row['feature']}: BC={row['shap_bimodality_coeff']:.3f}, "
                  f"skewness={row['shap_skewness']:.3f}, kurtosis={row['shap_kurtosis']:.3f}")
    else:
        print("\n► No features flagged as bimodal.")

    # Features with highest interaction effects
    print("\n► Features with Highest Interaction Effects:")
    for i, (_, row) in enumerate(df_inter.head(5).iterrows()):
        print(f"  {i + 1}. {row['feature']}: total interaction={row['shap_total_interaction']:.6f}")

    # Overall entropy
    print(f"\n► Overall Model Concentration:")
    print(f"  Entropy: {entropy_info['entropy']:.4f} / {entropy_info['max_entropy']:.4f}")
    print(f"  Normalized: {entropy_info['normalized_entropy']:.4f}")
    print(f"  Interpretation: {entropy_info['interpretation']}")

    # OOT stability insights
    if oot_comparison is not None and not oot_comparison.get("comparison_df", pd.DataFrame()).empty:
        oot_comp = oot_comparison["comparison_df"]
        print(f"\n► OOT (Out-of-Time) Bootstrap Stability Insights:")
        if "oot_complexity_shift" in oot_comp.columns:
            max_degraded = oot_comp.loc[oot_comp["oot_complexity_shift"].idxmax()]
            max_improved = oot_comp.loc[oot_comp["oot_complexity_shift"].idxmin()]
            mean_abs_shift = oot_comp["oot_complexity_shift"].abs().mean()
            print(f"  Mean absolute complexity shift (train→OOT): {mean_abs_shift:.4f}")
            print(f"  Most degraded in OOT: {max_degraded['feature']} "
                  f"(shift = +{max_degraded['oot_complexity_shift']:.4f})")
            print(f"  Most improved in OOT: {max_improved['feature']} "
                  f"(shift = {max_improved['oot_complexity_shift']:.4f})")
            n_degraded = (oot_comp["oot_complexity_shift"] > 0).sum()
            n_improved = (oot_comp["oot_complexity_shift"] < 0).sum()
            print(f"  Features degraded in OOT: {n_degraded} / {len(oot_comp)}")
            print(f"  Features improved in OOT: {n_improved} / {len(oot_comp)}")
    else:
        print(f"\n► OOT Bootstrap Stability: not available (skipped or empty)")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


# =============================================================================
# Section 18: SHAP Stability Learning Curves
# =============================================================================
@section_timer
def section18_shap_stability_learning_curves(
    X_train, y_train, X_holdout, feature_names, df_bootstrap
):
    """
    Compute SHAP-based stability with learning curves using SHAPStability.
    
    This section:
    - Trains models on bootstrap samples
    - Computes SHAP stability metrics across pool sizes
    - Fits learning curves to get floor parameters
    - Compares with marginal stability floors
    """
    if not SHAP_STABILITY_AVAILABLE or SHAPStability is None:
        print("SHAPStability not available. Skipping Section 18.")
        return None
    
    print("Setting up SHAP Stability analysis...")
    
    # Create model factory for LightGBM
    def model_factory():
        return LGBMClassifier(**{**LGBM_PARAMS, "verbose": -1})
    
    # Initialize SHAPStability with reasonable parameters for credit card dataset
    shap_stab = SHAPStability(
        model_factory=model_factory,
        explainer_type='tree',
        n_resamples=10,  # Reduced for computational efficiency
        min_pool=200,
        n_points=10,
        eval_set_strategy='holdout',
        eval_set_size=0.2,
        shap_subsample=200,  # Subsample for SHAP computation
        retrain_per_bootstrap=False,  # Option A: faster
        r2_threshold=0.70,
        extrapolate_to=[500, 1000, 2000],
        n_jobs=-1,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    print(f"  Pool sizes: will be determined automatically (min={200}, n_points={10})")
    print(f"  Bootstrap resamples per pool: {10}")
    print(f"  SHAP subsample: {200}")
    print(f"  This may take several minutes...")
    
    try:
        # Fit on training data
        results = shap_stab.fit(X_train, y_train, feature_names=feature_names)
        
        print("\nSHAP Stability Learning Curves Results:")
        print(f"  Features analyzed: {len(results.get('feature_results', []))}")
        
        # Extract and validate complexity score
        complexity_score = results.get('complexity_score', np.nan)
        print(f"\n  Overall SHAP Complexity Score: {complexity_score:.4f}" if not np.isnan(complexity_score) else "\n  Overall SHAP Complexity Score: NaN")
        
        # Extract per-metric floor parameters
        per_metric_floors = results.get('per_metric_floors', {})
        if per_metric_floors:
            print(f"\n  Per-Metric Floor Parameters (asymptotic stability limits):")
            for param, value in sorted(per_metric_floors.items()):
                if isinstance(value, (int, float)):
                    if np.isfinite(value):
                        print(f"    {param}: {value:.4f}")
                    else:
                        print(f"    {param}: NaN (fit failed or anomalous)")
        
        # Validate that complexity score was computed
        if np.isnan(complexity_score):
            print("\n  WARNING: SHAP complexity score is NaN!")
            print("  Checking learning curve fits...")
            learning_curves = results.get('learning_curves', {})
            for metric, lc_data in learning_curves.items():
                fit_info = lc_data.get('fit', {})
                if fit_info.get('fit_failed'):
                    print(f"    {metric}: FIT FAILED")
                elif fit_info.get('anomalous'):
                    print(f"    {metric}: ANOMALOUS (floor={fit_info.get('floor', 'N/A')})")
                elif 'floor' in fit_info:
                    print(f"    {metric}: floor={fit_info['floor']:.4f}")
        
        # Create comparison DataFrame with marginal stability
        # Note: fit() returns results directly; fit_panel() returns results with 'summary' key
        shap_summary = results.get('summary', pd.DataFrame())
        if shap_summary.empty:
            # Build summary from feature_results if not present
            feature_results = results.get('feature_results', [])
            if feature_results:
                summary_rows = []
                for fr in feature_results:
                    row = {
                        'feature': fr.get('feature', 'unknown'),
                        'complexity_score': complexity_score,  # Global score
                        'direction_consistency': fr.get('direction_consistency_mean', np.nan),
                        'rank_stability': fr.get('rank_stability_mean', np.nan),
                    }
                    summary_rows.append(row)
                shap_summary = pd.DataFrame(summary_rows)
                print(f"\n  Built summary DataFrame with {len(shap_summary)} features")
        
        if not shap_summary.empty and not df_bootstrap.empty:
            comparison_df = pd.DataFrame({
                'feature': feature_names,
            })
            
            # Add SHAP complexity if available
            if 'complexity_score' in shap_summary.columns:
                shap_comp = shap_summary[['feature', 'complexity_score']].rename(
                    columns={'complexity_score': 'shap_complexity_score'}
                )
                comparison_df = comparison_df.merge(shap_comp, on='feature', how='left')
            
            # Add marginal complexity
            if 'complexity_score' in df_bootstrap.columns:
                marg_comp = df_bootstrap[['feature', 'complexity_score']].rename(
                    columns={'complexity_score': 'marginal_complexity_score'}
                )
                comparison_df = comparison_df.merge(marg_comp, on='feature', how='left')
            
            # Compute difference
            if 'shap_complexity_score' in comparison_df.columns and 'marginal_complexity_score' in comparison_df.columns:
                comparison_df['complexity_diff'] = (
                    comparison_df['shap_complexity_score'] - comparison_df['marginal_complexity_score']
                ).fillna(0)
            
            print("\n  SHAP vs Marginal Complexity Comparison:")
            cols_to_show = ['feature', 'shap_complexity_score', 'marginal_complexity_score', 'complexity_diff']
            available_cols = [c for c in cols_to_show if c in comparison_df.columns]
            print(comparison_df[available_cols].to_string(index=False))
        
        # Visualization: Learning curves for top features
        # Note: feature_results is a list of dicts, not a dict of dicts
        feature_results = results.get('feature_results', [])
        if feature_results:
            # Select top 6 features
            top_features = feature_results[:6]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.flatten()
            
            for idx, feat_result in enumerate(top_features):
                if idx >= len(axes):
                    break
                    
                ax = axes[idx]
                feat_name = feat_result.get('feature', f'feature_{idx}')
                
                # Plot learning curve - use global learning curves from results
                pool_sizes = results.get('pool_sequence', [])
                learning_curves = results.get('learning_curves', {})
                
                if pool_sizes and learning_curves:
                    # Plot rank stability
                    if 'rank_stability' in learning_curves:
                        means = learning_curves['rank_stability'].get('means', [])
                        if means:
                            ax.plot(pool_sizes[:len(means)], means, 'b-o',
                                    label='Rank Stability', markersize=4)
                    
                    # Plot direction consistency
                    if 'direction_consistency' in learning_curves:
                        means = learning_curves['direction_consistency'].get('means', [])
                        if means:
                            ax.plot(pool_sizes[:len(means)], means, 'g-s',
                                    label='Direction Consistency', markersize=4)
                    
                    # Add floor line from per-metric floors
                    per_metric_floors = results.get('per_metric_floors', {})
                    if 'rank_stability' in per_metric_floors:
                        floor = per_metric_floors['rank_stability']
                        if np.isfinite(floor):
                            ax.axhline(y=1-floor, color='r', linestyle='--',
                                       label=f'Floor (instability): {1-floor:.3f}', alpha=0.7)
                    
                    ax.set_xlabel('Pool Size', fontsize=10)
                    ax.set_ylabel('Stability Metric', fontsize=10)
                    ax.set_title(f'{feat_name}', fontsize=11)
                    ax.legend(fontsize=8)
                    ax.grid(True, alpha=0.3)
                    ax.set_ylim([0, 1.05])
            
            # Hide unused subplots
            for idx in range(len(top_features), len(axes)):
                axes[idx].set_visible(False)
            
            plt.suptitle('Section 18: SHAP Stability Learning Curves', fontsize=14, y=1.02)
            fig.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, "section18_shap_stability_learning_curves.png")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nLearning curves plot saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"ERROR in SHAP Stability analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Section 19: Train/OOT Drift Analysis
# =============================================================================
@section_timer
def section19_train_oot_drift(X_train, y_train, X_test, y_test, feature_names):
    """
    Compare train vs OOT (out-of-time/test) SHAP stability.
    
    This section:
    - Uses TrainOOTStability to compare SHAP patterns
    - Computes drift metrics between periods
    - Generates drift grades (A-F) per feature
    """
    if not SHAP_STABILITY_AVAILABLE or TrainOOTStability is None:
        print("TrainOOTStability not available. Skipping Section 19.")
        return None
    
    print("Setting up Train/OOT Drift Analysis...")
    print(f"  Train set size: {len(X_train)}")
    print(f"  OOT (test) set size: {len(X_test)}")
    
    # Create model factory
    def model_factory():
        return LGBMClassifier(**{**LGBM_PARAMS, "verbose": -1})
    
    # Initialize TrainOOTStability
    oot_stability = TrainOOTStability(
        model_factory=model_factory,
        explainer_type='tree',
        shap_subsample=500,  # Subsample for efficiency
        top_k=10,
        random_state=RANDOM_STATE,
        verbose=1,
    )
    
    try:
        print("\nTraining model on train set and computing SHAP values...")
        print("Computing SHAP values on OOT set...")
        
        # Fit and compare
        results = oot_stability.fit(
            X_train, y_train,
            X_test, y_test,
            feature_names=feature_names
        )
        
        # Print summary
        print("\n" + "=" * 60)
        print("TRAIN/OOT DRIFT ANALYSIS RESULTS")
        print("=" * 60)
        
        # Overall drift
        drift_score = results.get('overall_drift_score', 0)
        drift_grade = results.get('drift_grade', 'N/A')
        print(f"\n► Overall Drift Score: {drift_score:.4f}")
        print(f"► Overall Drift Grade: {drift_grade}")
        
        # Drift metrics
        drift_metrics = results.get('drift_metrics', {})
        if drift_metrics:
            print("\n► Drift Metrics:")
            for metric_name, metric_value in drift_metrics.items():
                if isinstance(metric_value, (int, float)):
                    print(f"    {metric_name}: {metric_value:.4f}")
        
        # Per-feature drift
        feature_drift = results.get('feature_drift', {})
        if feature_drift:
            # Create DataFrame for visualization
            drift_records = []
            for feat, metrics in feature_drift.items():
                drift_records.append({
                    'feature': feat,
                    'rank_train': metrics.get('rank_train', 0),
                    'rank_oot': metrics.get('rank_oot', 0),
                    'rank_change': metrics.get('rank_change', 0),
                    'magnitude_ratio': metrics.get('magnitude_ratio', 1.0),
                    'direction_consistent': metrics.get('direction_consistent', True),
                    'wasserstein': metrics.get('wasserstein', 0),
                    'drift_score': metrics.get('drift_score', 0),
                })
            
            drift_df = pd.DataFrame(drift_records)
            drift_df = drift_df.sort_values('drift_score', ascending=False)
            
            print("\n► Per-Feature Drift (sorted by drift score):")
            print(drift_df.to_string(index=False))
            
            # Identify problematic features
            high_drift = drift_df[drift_df['drift_score'] > 0.3]
            direction_flips = drift_df[~drift_df['direction_consistent']]
            
            if len(high_drift) > 0:
                print(f"\n► High Drift Features (drift_score > 0.3): {len(high_drift)}")
                for _, row in high_drift.head(5).iterrows():
                    print(f"    - {row['feature']}: drift_score={row['drift_score']:.3f}")
            
            if len(direction_flips) > 0:
                print(f"\n► Direction Flip Features: {len(direction_flips)}")
                for _, row in direction_flips.iterrows():
                    print(f"    - {row['feature']}: SHAP direction changed between train and OOT")
        
        # Visualization: Drift heatmap
        if feature_drift:
            fig, axes = plt.subplots(1, 2, figsize=(14, 6))
            
            # Plot 1: Rank comparison
            ax1 = axes[0]
            train_ranks = [feature_drift[f].get('rank_train', 0) for f in feature_names]
            oot_ranks = [feature_drift[f].get('rank_oot', 0) for f in feature_names]
            
            ax1.scatter(train_ranks, oot_ranks, s=60, alpha=0.7, edgecolors='black')
            max_rank = max(max(train_ranks), max(oot_ranks))
            ax1.plot([0, max_rank], [0, max_rank], 'k--', alpha=0.4, label='Perfect agreement')
            
            for i, feat in enumerate(feature_names):
                ax1.annotate(feat, (train_ranks[i], oot_ranks[i]),
                            fontsize=6, alpha=0.7, xytext=(3, 3),
                            textcoords='offset points')
            
            ax1.set_xlabel('Train Rank', fontsize=12)
            ax1.set_ylabel('OOT Rank', fontsize=12)
            ax1.set_title('Feature Importance Rank: Train vs OOT', fontsize=13)
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Plot 2: Drift scores bar chart
            ax2 = axes[1]
            drift_scores = [feature_drift[f].get('drift_score', 0) for f in feature_names]
            colors = ['#d62728' if s > 0.3 else '#2ca02c' for s in drift_scores]
            
            y_pos = np.arange(len(feature_names))
            ax2.barh(y_pos, drift_scores, color=colors, edgecolor='black', alpha=0.8)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(feature_names, fontsize=8)
            ax2.set_xlabel('Drift Score', fontsize=12)
            ax2.set_title('Per-Feature Drift Score (red > 0.3)', fontsize=13)
            ax2.axvline(x=0.3, color='red', linestyle='--', alpha=0.5, label='High drift threshold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='x')
            
            plt.suptitle(f'Section 19: Train/OOT Drift Analysis (Overall Grade: {drift_grade})',
                        fontsize=14, y=1.02)
            fig.tight_layout()
            output_path = os.path.join(OUTPUT_DIR, "section19_train_oot_drift.png")
            fig.savefig(output_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            print(f"\nDrift analysis plot saved to: {output_path}")
        
        return results
        
    except Exception as e:
        print(f"ERROR in Train/OOT Drift analysis: {e}")
        import traceback
        traceback.print_exc()
        return None


# =============================================================================
# Section 20: Marginal vs SHAP Comparison Summary
# =============================================================================
@section_timer
def section20_marginal_vs_shap_comparison(
    df_bootstrap, shap_stability_results, train_oot_results, feature_names, master
):
    """
    Compare marginal complexity scores with SHAP complexity scores.
    
    This section:
    - Identifies false alarms (stable SHAP, unstable marginal)
    - Identifies missed risks (unstable SHAP, stable marginal)
    - Provides actionable recommendations
    """
    print("=" * 60)
    print("MARGINAL VS SHAP STABILITY COMPARISON")
    print("=" * 60)
    
    # Build comparison DataFrame
    comparison_data = []
    
    for feat in feature_names:
        row = {'feature': feat}
        
        # Get marginal complexity
        if not df_bootstrap.empty and 'complexity_score' in df_bootstrap.columns:
            marg_row = df_bootstrap[df_bootstrap['feature'] == feat]
            if not marg_row.empty:
                row['marginal_complexity'] = marg_row['complexity_score'].values[0]
            else:
                row['marginal_complexity'] = np.nan
        else:
            row['marginal_complexity'] = np.nan
        
        # Get SHAP complexity from Section 18
        # The complexity_score is a global value in SHAP stability (not per-feature)
        if shap_stability_results is not None:
            # Try to get the global complexity score directly from results
            global_shap_complexity = shap_stability_results.get('complexity_score', np.nan)
            
            # Also try summary DataFrame if available
            shap_summary = shap_stability_results.get('summary', pd.DataFrame())
            if not shap_summary.empty and 'complexity_score' in shap_summary.columns:
                # Use per-feature complexity from summary if available
                shap_row = shap_summary[shap_summary['feature'] == feat]
                if not shap_row.empty:
                    row['shap_complexity'] = shap_row['complexity_score'].values[0]
                else:
                    # Fall back to global complexity score
                    row['shap_complexity'] = global_shap_complexity
            else:
                # Use global complexity score
                row['shap_complexity'] = global_shap_complexity
                
            # Log if NaN
            if np.isnan(row['shap_complexity']):
                print(f"  DEBUG: SHAP complexity is NaN for {feat}")
        else:
            row['shap_complexity'] = np.nan
        
        # Get OOT drift score from Section 19
        if train_oot_results is not None:
            feature_drift = train_oot_results.get('feature_drift', {})
            if feat in feature_drift:
                row['oot_drift_score'] = feature_drift[feat].get('drift_score', np.nan)
                row['direction_consistent'] = feature_drift[feat].get('direction_consistent', True)
            else:
                row['oot_drift_score'] = np.nan
                row['direction_consistent'] = True
        else:
            row['oot_drift_score'] = np.nan
            row['direction_consistent'] = True
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Define thresholds
    MARGINAL_THRESHOLD = 0.5  # High complexity = unstable
    SHAP_THRESHOLD = 0.5  # High complexity = unstable
    DRIFT_THRESHOLD = 0.3  # High drift = problematic
    
    # Classify features
    def classify_feature(row):
        marg = row.get('marginal_complexity', np.nan)
        shap = row.get('shap_complexity', np.nan)
        drift = row.get('oot_drift_score', np.nan)
        
        # Handle NaN
        marg_high = not np.isnan(marg) and marg > MARGINAL_THRESHOLD
        shap_high = not np.isnan(shap) and shap > SHAP_THRESHOLD
        drift_high = not np.isnan(drift) and drift > DRIFT_THRESHOLD
        
        if marg_high and not shap_high:
            return "FALSE_ALARM"  # Marginal says unstable, SHAP says stable
        elif shap_high and not marg_high:
            return "MISSED_RISK"  # SHAP says unstable, marginal missed it
        elif marg_high and shap_high:
            return "CONFIRMED_UNSTABLE"  # Both agree: unstable
        elif drift_high:
            return "OOT_DRIFT"  # High drift in OOT
        else:
            return "STABLE"  # Both agree: stable
    
    comparison_df['classification'] = comparison_df.apply(classify_feature, axis=1)
    
    # Print summary
    print("\n► Classification Summary:")
    for cls in ["STABLE", "CONFIRMED_UNSTABLE", "FALSE_ALARM", "MISSED_RISK", "OOT_DRIFT"]:
        count = (comparison_df['classification'] == cls).sum()
        print(f"    {cls}: {count} features")
    
    # Detailed breakdown
    false_alarms = comparison_df[comparison_df['classification'] == 'FALSE_ALARM']
    missed_risks = comparison_df[comparison_df['classification'] == 'MISSED_RISK']
    confirmed_unstable = comparison_df[comparison_df['classification'] == 'CONFIRMED_UNSTABLE']
    oot_drift = comparison_df[comparison_df['classification'] == 'OOT_DRIFT']
    
    if len(false_alarms) > 0:
        print(f"\n► FALSE ALARMS (Marginal unstable, SHAP stable): {len(false_alarms)}")
        print("  These features may have complex marginal distributions but stable model contributions.")
        for _, row in false_alarms.iterrows():
            print(f"    - {row['feature']}: marginal={row['marginal_complexity']:.3f}, "
                  f"shap={row['shap_complexity']:.3f}")
    
    if len(missed_risks) > 0:
        print(f"\n► MISSED RISKS (SHAP unstable, Marginal stable): {len(missed_risks)}")
        print("  These features have stable distributions but unstable model contributions.")
        print("  ** ACTION REQUIRED: Monitor these features closely in production. **")
        for _, row in missed_risks.iterrows():
            print(f"    - {row['feature']}: marginal={row['marginal_complexity']:.3f}, "
                  f"shap={row['shap_complexity']:.3f}")
    
    if len(confirmed_unstable) > 0:
        print(f"\n► CONFIRMED UNSTABLE (Both methods agree): {len(confirmed_unstable)}")
        print("  These features are unstable by both metrics.")
        print("  ** ACTION REQUIRED: Consider feature engineering or removal. **")
        for _, row in confirmed_unstable.iterrows():
            print(f"    - {row['feature']}: marginal={row['marginal_complexity']:.3f}, "
                  f"shap={row['shap_complexity']:.3f}")
    
    if len(oot_drift) > 0:
        print(f"\n► OOT DRIFT (High drift in test period): {len(oot_drift)}")
        print("  These features show different behavior in OOT data.")
        for _, row in oot_drift.iterrows():
            print(f"    - {row['feature']}: drift_score={row['oot_drift_score']:.3f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Plot 1: Marginal vs SHAP complexity scatter
    ax1 = axes[0, 0]
    valid_mask = comparison_df['marginal_complexity'].notna() & comparison_df['shap_complexity'].notna()
    if valid_mask.any():
        scatter_data = comparison_df[valid_mask]
        colors = []
        for cls in scatter_data['classification']:
            if cls == 'STABLE':
                colors.append('#2ca02c')
            elif cls == 'CONFIRMED_UNSTABLE':
                colors.append('#d62728')
            elif cls == 'FALSE_ALARM':
                colors.append('#ff7f0e')
            elif cls == 'MISSED_RISK':
                colors.append('#9467bd')
            else:
                colors.append('#8c564b')
        
        ax1.scatter(scatter_data['marginal_complexity'],
                   scatter_data['shap_complexity'],
                   c=colors, s=80, alpha=0.7, edgecolors='black')
        
        # Reference lines
        ax1.axhline(y=SHAP_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
        ax1.axvline(x=MARGINAL_THRESHOLD, color='gray', linestyle='--', alpha=0.5)
        
        # Labels
        for _, row in scatter_data.iterrows():
            ax1.annotate(row['feature'],
                        (row['marginal_complexity'], row['shap_complexity']),
                        fontsize=6, alpha=0.7, xytext=(3, 3),
                        textcoords='offset points')
        
        ax1.set_xlabel('Marginal Complexity Score', fontsize=12)
        ax1.set_ylabel('SHAP Complexity Score', fontsize=12)
        ax1.set_title('Marginal vs SHAP Complexity', fontsize=13)
        ax1.grid(True, alpha=0.3)
        
        # Legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#2ca02c', label='Stable'),
            Patch(facecolor='#d62728', label='Confirmed Unstable'),
            Patch(facecolor='#ff7f0e', label='False Alarm'),
            Patch(facecolor='#9467bd', label='Missed Risk'),
        ]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    # Plot 2: Classification counts
    ax2 = axes[0, 1]
    class_counts = comparison_df['classification'].value_counts()
    colors_bar = {
        'STABLE': '#2ca02c',
        'CONFIRMED_UNSTABLE': '#d62728',
        'FALSE_ALARM': '#ff7f0e',
        'MISSED_RISK': '#9467bd',
        'OOT_DRIFT': '#8c564b',
    }
    bar_colors = [colors_bar.get(c, '#7f7f7f') for c in class_counts.index]
    ax2.bar(class_counts.index, class_counts.values, color=bar_colors, edgecolor='black')
    ax2.set_ylabel('Number of Features', fontsize=12)
    ax2.set_title('Feature Classification Summary', fontsize=13)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: OOT Drift Score Distribution
    ax3 = axes[1, 0]
    if comparison_df['oot_drift_score'].notna().any():
        drift_data = comparison_df[comparison_df['oot_drift_score'].notna()].sort_values('oot_drift_score', ascending=True)
        colors_drift = ['#d62728' if x > DRIFT_THRESHOLD else '#2ca02c'
                       for x in drift_data['oot_drift_score']]
        ax3.barh(drift_data['feature'], drift_data['oot_drift_score'],
                color=colors_drift, edgecolor='black', alpha=0.8)
        ax3.axvline(x=DRIFT_THRESHOLD, color='red', linestyle='--', alpha=0.5)
        ax3.set_xlabel('OOT Drift Score', fontsize=12)
        ax3.set_title('Train vs OOT Drift Score', fontsize=13)
        ax3.grid(True, alpha=0.3, axis='x')
    
    # Plot 4: Recommendations
    ax4 = axes[1, 1]
    ax4.axis('off')
    
    recommendations = []
    if len(missed_risks) > 0:
        recommendations.append(f"⚠️ {len(missed_risks)} MISSED RISKS: Features with stable distributions\n"
                              f"   but unstable model contributions. Monitor closely!")
    if len(confirmed_unstable) > 0:
        recommendations.append(f"🔴 {len(confirmed_unstable)} CONFIRMED UNSTABLE: Consider\n"
                              f"   feature engineering or removal.")
    if len(false_alarms) > 0:
        recommendations.append(f"🟡 {len(false_alarms)} FALSE ALARMS: Marginal analysis flagged\n"
                              f"   these, but SHAP shows stable contributions.")
    if len(oot_drift) > 0:
        recommendations.append(f"📊 {len(oot_drift)} OOT DRIFT: Features showing different\n"
                              f"   behavior in test period.")
    
    if recommendations:
        rec_text = "RECOMMENDATIONS\n" + "=" * 40 + "\n\n" + "\n\n".join(recommendations)
    else:
        rec_text = "All features show stable behavior.\nNo immediate action required."
    
    ax4.text(0.1, 0.9, rec_text, transform=ax4.transAxes, fontsize=11,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.set_title('Actionable Recommendations', fontsize=13)
    
    plt.suptitle('Section 20: Marginal vs SHAP Stability Comparison', fontsize=14, y=1.02)
    fig.tight_layout()
    output_path = os.path.join(OUTPUT_DIR, "section20_marginal_vs_shap_comparison.png")
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nComparison plot saved to: {output_path}")
    
    # Save comparison table to CSV
    csv_path = os.path.join(OUTPUT_DIR, "marginal_vs_shap_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Comparison table saved to: {csv_path}")
    
    return comparison_df


# =============================================================================
# Main
# =============================================================================
def main():
    total_start = time.time()

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    print()

    # Section 1: Data Loading & Model Training
    data = section01_load_and_train()
    model = data["model"]
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    X_holdout = data["X_holdout"]
    feature_names = data["feature_names"]
    df_train = data["df_train"]

    # Section 2: LightGBM Feature Importance
    df_imp = section02_lgbm_importance(model, feature_names)

    # Section 3: Gain vs Split Divergence
    df_divergence = section03_gain_split_divergence(df_imp)

    # Section 4: Feature Tree Coverage
    df_coverage, tree_df = section04_tree_coverage(model, feature_names)

    # Section 5: Feature Depth Statistics
    df_depth = section05_depth_statistics(tree_df, feature_names)

    # Section 6: Permutation Importance
    df_perm = section06_permutation_importance_(model, X_test, y_test, feature_names)

    # Section 7: SHAP Analysis
    df_shap, sv = section07_shap_analysis(model, X_holdout, feature_names)

    # Section 8: SHAP Variance & CV
    df_shap_var = section08_shap_variance(sv, feature_names)

    # Section 9: SHAP Interaction Values
    df_inter = section09_shap_interactions(model, X_holdout, feature_names)

    # Section 10: SHAP Distribution Shape Analysis
    df_dist = section10_shap_distribution(sv, feature_names)

    # Section 11: SHAP Entropy / Concentration
    entropy_info = section11_shap_entropy(df_shap)

    # Section 12: Cross-Seed SHAP Consistency
    df_consistency = section12_cross_seed_shap(X_train, y_train, X_holdout, feature_names)

    # Section 13: Bootstrap Stability Toolkit
    df_bootstrap, train_panel_results = section13_bootstrap_stability(df_train, feature_names)

    # Section 13b: OOT Bootstrap Stability Comparison
    oot_results = section13b_oot_bootstrap_comparison(
        df_train, X_test, train_panel_results, feature_names,
    )

    # Section 14: Master Comparison Table
    master = section14_master_table(
        feature_names,
        df_imp,
        df_divergence,
        df_coverage,
        df_depth,
        df_perm,
        df_shap,
        df_shap_var,
        df_inter,
        df_dist,
        df_consistency,
        df_bootstrap,
        oot_comparison=oot_results,
    )

    # Section 15: Correlation Analysis
    section15_correlation_analysis(master)

    # Section 16: Feature Reliability Scoring
    df_reliability = section16_reliability_scoring(master)

    # Section 17: Key Findings Summary
    section17_key_findings(master, df_reliability, entropy_info, df_dist, df_inter, oot_comparison=oot_results)

    # =========================================================================
    # NEW SHAP Stability Sections (18-20)
    # =========================================================================
    
    # Section 18: SHAP Stability Learning Curves
    shap_stability_results = section18_shap_stability_learning_curves(
        X_train, y_train, X_holdout, feature_names, df_bootstrap
    )
    
    # Section 19: Train/OOT Drift Analysis
    train_oot_results_new = section19_train_oot_drift(
        X_train, y_train, X_test, y_test, feature_names
    )
    
    # Section 20: Marginal vs SHAP Comparison Summary
    comparison_df = section20_marginal_vs_shap_comparison(
        df_bootstrap, shap_stability_results, train_oot_results_new, feature_names, master
    )
    
    # =========================================================================
    # Final Summary Write to File
    # =========================================================================
    _write_comprehensive_summary(
        master, df_reliability, entropy_info, df_dist, df_inter,
        oot_results, shap_stability_results, train_oot_results_new, comparison_df
    )

    total_elapsed = time.time() - total_start
    print(f"\nTotal analysis time: {total_elapsed:.1f}s ({total_elapsed / 60:.1f}min)")


def _write_comprehensive_summary(
    master, df_reliability, entropy_info, df_dist, df_inter,
    oot_results, shap_stability_results, train_oot_results, comparison_df
):
    """Write comprehensive analysis summary to text file."""
    summary_path = os.path.join(OUTPUT_DIR, "comprehensive_analysis_summary.txt")
    
    with open(summary_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("COMPREHENSIVE FEATURE ANALYSIS SUMMARY\n")
        f.write("=" * 80 + "\n\n")
        
        # Section: Top Features
        f.write("TOP 10 MOST RELIABLE FEATURES\n")
        f.write("-" * 40 + "\n")
        for i, row in df_reliability.head(10).iterrows():
            f.write(f"  {i+1}. {row['feature']}: reliability={row['reliability_score']:.3f}\n")
        f.write("\n")
        
        # Section: SHAP Entropy
        f.write("MODEL CONCENTRATION (SHAP Entropy)\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Normalized Entropy: {entropy_info['normalized_entropy']:.4f}\n")
        f.write(f"  Interpretation: {entropy_info['interpretation']}\n\n")
        
        # Section: Marginal vs SHAP Comparison
        if comparison_df is not None and not comparison_df.empty:
            f.write("MARGINAL VS SHAP STABILITY COMPARISON\n")
            f.write("-" * 40 + "\n")
            
            # Count classifications
            for cls in ["STABLE", "CONFIRMED_UNSTABLE", "FALSE_ALARM", "MISSED_RISK", "OOT_DRIFT"]:
                count = (comparison_df['classification'] == cls).sum()
                f.write(f"  {cls}: {count} features\n")
            f.write("\n")
            
            # List problematic features
            confirmed_unstable = comparison_df[comparison_df['classification'] == 'CONFIRMED_UNSTABLE']
            missed_risks = comparison_df[comparison_df['classification'] == 'MISSED_RISK']
            
            if len(confirmed_unstable) > 0:
                f.write("  CONFIRMED UNSTABLE (consider removal/engineering):\n")
                for _, row in confirmed_unstable.iterrows():
                    f.write(f"    - {row['feature']}\n")
                f.write("\n")
            
            if len(missed_risks) > 0:
                f.write("  MISSED RISKS (monitor closely in production):\n")
                for _, row in missed_risks.iterrows():
                    f.write(f"    - {row['feature']}\n")
                f.write("\n")
        
        # Section: Train/OOT Drift
        if train_oot_results is not None:
            f.write("TRAIN/OOT DRIFT ANALYSIS\n")
            f.write("-" * 40 + "\n")
            f.write(f"  Overall Drift Score: {train_oot_results.get('overall_drift_score', 'N/A')}\n")
            f.write(f"  Overall Drift Grade: {train_oot_results.get('drift_grade', 'N/A')}\n\n")
        
        # Section: Key Recommendations
        f.write("KEY RECOMMENDATIONS\n")
        f.write("-" * 40 + "\n")
        
        if comparison_df is not None:
            missed = (comparison_df['classification'] == 'MISSED_RISK').sum()
            confirmed = (comparison_df['classification'] == 'CONFIRMED_UNSTABLE').sum()
            
            if missed > 0:
                f.write(f"  1. MONITOR {missed} 'MISSED RISK' features closely in production\n")
            if confirmed > 0:
                f.write(f"  2. CONSIDER removing or engineering {confirmed} 'CONFIRMED UNSTABLE' features\n")
            
            stable = (comparison_df['classification'] == 'STABLE').sum()
            f.write(f"  3. {stable} features are stable and reliable for production use\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("END OF SUMMARY\n")
        f.write("=" * 80 + "\n")
    
    print(f"\nComprehensive summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
