"""
LightGBM + SHAP Feature Importance Experiment
==============================================

This script performs a comprehensive feature importance analysis on the
Taiwanese credit card default dataset using LightGBM and SHAP.

The experiment:
1. Loads and prepares the credit card default dataset
2. Trains a LightGBM classifier with tuned hyperparameters
3. Extracts and visualizes LightGBM feature importances (split & gain)
4. Computes SHAP values and creates summary/bar plots
5. Compares rankings across all three methods (gain, split, SHAP)
6. Summarizes key findings including rank disagreements

All output files are saved in the feature-bootstrapping-toolkit/ directory.
"""

import sys
import os
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
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
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
    from scipy.stats import spearmanr
except ImportError:
    print("ERROR: scipy is required. Install with: pip install scipy")
    sys.exit(1)

try:
    from bootstrap_stability import BootstrapStability
except ImportError:
    print("WARNING: bootstrap_stability not available. Section 7 will be skipped.")
    BootstrapStability = None

warnings.filterwarnings("ignore")


# =============================================================================
# Configuration
# =============================================================================
DATA_PATH = os.path.join("..", "default+of+credit+card+clients", "default of credit card clients.xls")
OUTPUT_DIR = "."
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET_COL = "default payment next month"
DROP_COLS = ["ID"]


def print_header(title):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


# =============================================================================
# 1. Load and Prepare Data
# =============================================================================
def load_and_prepare_data():
    """Load the credit card default dataset and split into train/test/hold-out sets."""
    print_header("1. LOADING AND PREPARING DATA")

    df = pd.read_excel(DATA_PATH, header=1)
    print(f"Loaded dataset: {df.shape[0]} rows x {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # Drop ID column
    df = df.drop(columns=DROP_COLS)
    print(f"\nAfter dropping {DROP_COLS}: {df.shape[1]} columns remaining")

    # Split into features and target
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    feature_names = list(X.columns)

    print(f"\nTarget distribution:")
    print(y.value_counts().to_string())
    print(f"Default rate: {y.mean():.4f}")

    # Train/test split (80/20, stratified)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    print(f"\nTrain set: {X_train.shape[0]} samples")
    print(f"Test set:  {X_test.shape[0]} samples")

    # Create hold-out set (test features without target)
    X_holdout = X_test.copy()
    print(f"Hold-out set: {X_holdout.shape[0]} samples (no target)")

    return X_train, X_test, y_train, y_test, X_holdout, feature_names


# =============================================================================
# 2. Train LightGBM Classifier
# =============================================================================
def train_lightgbm(X_train, y_train, X_test, y_test):
    """Train a LightGBM classifier and evaluate on the test set."""
    print_header("2. TRAINING LIGHTGBM CLASSIFIER")

    model = LGBMClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=RANDOM_STATE,
        verbose=-1,
    )

    model.fit(X_train, y_train)
    print("Model trained successfully.")

    # Evaluate on test set
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"\nTest Set Performance:")
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  ROC-AUC:   {roc_auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No Default", "Default"]))

    return model


# =============================================================================
# 3. LightGBM Feature Importance
# =============================================================================
def analyze_lgbm_importance(model, feature_names):
    """Extract and visualize LightGBM feature importances (split and gain)."""
    print_header("3. LIGHTGBM FEATURE IMPORTANCE")

    # Extract importances
    split_imp = model.booster_.feature_importance(importance_type="split")
    gain_imp = model.booster_.feature_importance(importance_type="gain")

    # Create DataFrames
    split_df = pd.DataFrame({"feature": feature_names, "split_importance": split_imp})
    gain_df = pd.DataFrame({"feature": feature_names, "gain_importance": gain_imp})

    # Sort by gain
    split_df = split_df.sort_values("split_importance", ascending=True).reset_index(drop=True)
    gain_df = gain_df.sort_values("gain_importance", ascending=True).reset_index(drop=True)

    # Plot side-by-side horizontal bar charts
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 12))

    # Split importance
    ax1.barh(split_df["feature"], split_df["split_importance"], color="steelblue", edgecolor="navy", alpha=0.8)
    ax1.set_xlabel("Split Importance (count)", fontsize=12)
    ax1.set_title("LightGBM Split Importance", fontsize=14, fontweight="bold")
    ax1.tick_params(axis="y", labelsize=9)

    # Gain importance
    ax2.barh(gain_df["feature"], gain_df["gain_importance"], color="coral", edgecolor="darkred", alpha=0.8)
    ax2.set_xlabel("Gain Importance (total gain)", fontsize=12)
    ax2.set_title("LightGBM Gain Importance", fontsize=14, fontweight="bold")
    ax2.tick_params(axis="y", labelsize=9)

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "lgbm_feature_importance.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {save_path}")

    # Print ranked table by gain importance
    gain_ranked = gain_df.sort_values("gain_importance", ascending=False).reset_index(drop=True)
    gain_ranked.index += 1
    gain_ranked.index.name = "Rank"
    print(f"\nFeatures ranked by Gain Importance:")
    print(gain_ranked.to_string())

    return split_df, gain_df


# =============================================================================
# 4. SHAP Analysis
# =============================================================================
def analyze_shap(model, X_holdout, feature_names):
    """Compute SHAP values and create summary/bar plots."""
    print_header("4. SHAP ANALYSIS")

    # Compute SHAP values using TreeExplainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_holdout)

    # For binary classification, shap_values is a list of two arrays;
    # we use the positive class (index 1) SHAP values
    if isinstance(shap_values, list):
        shap_vals = shap_values[1]
    else:
        shap_vals = shap_values

    print(f"SHAP values computed for {X_holdout.shape[0]} hold-out samples.")
    print(f"SHAP values shape: {shap_vals.shape}")

    # --- SHAP Summary Plot (Beeswarm) ---
    fig_summary = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_holdout, show=False)
    plt.title("SHAP Summary Plot (Beeswarm)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "shap_summary_plot.png")
    fig_summary.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_summary)
    print(f"Saved: {save_path}")

    # --- SHAP Bar Plot (Mean Absolute SHAP) ---
    fig_bar = plt.figure(figsize=(12, 10))
    shap.summary_plot(shap_vals, X_holdout, plot_type="bar", show=False)
    plt.title("SHAP Bar Plot (Mean |SHAP|)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "shap_bar_plot.png")
    fig_bar.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig_bar)
    print(f"Saved: {save_path}")

    # Compute mean absolute SHAP values per feature
    mean_abs_shap = np.abs(shap_vals).mean(axis=0)
    shap_df = pd.DataFrame({"feature": feature_names, "mean_abs_shap": mean_abs_shap})
    shap_df = shap_df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    shap_df.index += 1
    shap_df.index.name = "Rank"

    print(f"\nFeatures ranked by Mean Absolute SHAP Value:")
    print(shap_df.to_string())

    return shap_df, shap_vals


# =============================================================================
# 5. Comparison Analysis
# =============================================================================
def compare_importances(split_df, gain_df, shap_df, feature_names):
    """Compare feature importance rankings across LGBM and SHAP methods."""
    print_header("5. COMPARISON ANALYSIS")

    # Build comparison DataFrame with normalized importances
    # Normalize each to sum to 1
    split_norm = split_df.set_index("feature")["split_importance"]
    split_norm = split_norm / split_norm.sum()

    gain_norm = gain_df.set_index("feature")["gain_importance"]
    gain_norm = gain_norm / gain_norm.sum()

    shap_norm = shap_df.set_index("feature")["mean_abs_shap"]
    shap_norm = shap_norm / shap_norm.sum()

    # Compute ranks (1 = most important)
    gain_rank = gain_norm.rank(ascending=False).astype(int)
    split_rank = split_norm.rank(ascending=False).astype(int)
    shap_rank = shap_norm.rank(ascending=False).astype(int)

    comparison = pd.DataFrame({
        "feature": feature_names,
        "gain_importance_norm": gain_norm.reindex(feature_names).values,
        "split_importance_norm": split_norm.reindex(feature_names).values,
        "mean_abs_shap_norm": shap_norm.reindex(feature_names).values,
        "gain_rank": gain_rank.reindex(feature_names).values,
        "split_rank": split_rank.reindex(feature_names).values,
        "shap_rank": shap_rank.reindex(feature_names).values,
    })

    # Sort by SHAP rank (ascending = most important first)
    comparison = comparison.sort_values("shap_rank").reset_index(drop=True)
    comparison.index += 1
    comparison.index.name = "Rank"

    print("\nComparison Table (sorted by SHAP rank):")
    print(comparison.to_string())

    # Spearman rank correlations
    corr_gain_shap, p_gain_shap = spearmanr(comparison["gain_rank"], comparison["shap_rank"])
    corr_split_shap, p_split_shap = spearmanr(comparison["split_rank"], comparison["shap_rank"])
    corr_gain_split, p_gain_split = spearmanr(comparison["gain_rank"], comparison["split_rank"])

    corr_matrix = pd.DataFrame({
        "LGBM Gain": [1.0, corr_gain_split, corr_gain_shap],
        "LGBM Split": [corr_gain_split, 1.0, corr_split_shap],
        "SHAP": [corr_gain_shap, corr_split_shap, 1.0],
    }, index=["LGBM Gain", "LGBM Split", "SHAP"])

    print(f"\nSpearman Rank Correlations:")
    print(corr_matrix.to_string(float_format="{:.4f}".format))
    print(f"\nP-values:")
    print(f"  Gain vs SHAP:   {p_gain_shap:.6f}")
    print(f"  Split vs SHAP:  {p_split_shap:.6f}")
    print(f"  Gain vs Split:  {p_gain_split:.6f}")

    # --- Comparison Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(22, 8))

    # Subplot 1: Scatter of LGBM gain rank vs SHAP rank
    ax1 = axes[0]
    ax1.scatter(comparison["gain_rank"], comparison["shap_rank"],
                c="steelblue", edgecolors="navy", s=80, alpha=0.7, zorder=3)
    max_rank = max(comparison["gain_rank"].max(), comparison["shap_rank"].max())
    ax1.plot([0, max_rank + 1], [0, max_rank + 1], "k--", alpha=0.5, label="Perfect agreement")
    ax1.set_xlabel("LGBM Gain Rank", fontsize=12)
    ax1.set_ylabel("SHAP Rank", fontsize=12)
    ax1.set_title(f"Gain Rank vs SHAP Rank\n(ρ = {corr_gain_shap:.3f})", fontsize=13, fontweight="bold")
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Grouped horizontal bar chart for top 15 features
    ax2 = axes[1]
    top15 = comparison.head(15).sort_values("shap_rank", ascending=True)
    y_pos = np.arange(len(top15))
    bar_height = 0.25

    ax2.barh(y_pos - bar_height, top15["gain_importance_norm"], bar_height,
             label="LGBM Gain", color="coral", alpha=0.8)
    ax2.barh(y_pos, top15["split_importance_norm"], bar_height,
             label="LGBM Split", color="steelblue", alpha=0.8)
    ax2.barh(y_pos + bar_height, top15["mean_abs_shap_norm"], bar_height,
             label="SHAP", color="forestgreen", alpha=0.8)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(top15["feature"], fontsize=9)
    ax2.set_xlabel("Normalized Importance", fontsize=12)
    ax2.set_title("Top 15 Features: Method Comparison", fontsize=13, fontweight="bold")
    ax2.legend(fontsize=9, loc="lower right")
    ax2.grid(True, axis="x", alpha=0.3)

    # Subplot 3: Heatmap of Spearman correlations
    ax3 = axes[2]
    im = ax3.imshow(corr_matrix.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(len(corr_matrix.columns)))
    ax3.set_xticklabels(corr_matrix.columns, fontsize=10, rotation=30, ha="right")
    ax3.set_yticks(range(len(corr_matrix.index)))
    ax3.set_yticklabels(corr_matrix.index, fontsize=10)
    ax3.set_title("Spearman Rank Correlations", fontsize=13, fontweight="bold")

    # Annotate heatmap cells
    for i in range(len(corr_matrix.index)):
        for j in range(len(corr_matrix.columns)):
            val = corr_matrix.values[i, j]
            ax3.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=12, fontweight="bold",
                     color="white" if val > 0.85 else "black")

    fig.colorbar(im, ax=ax3, shrink=0.8, label="Spearman ρ")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "importance_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    return comparison, corr_matrix


# =============================================================================
# 6. Key Findings Summary
# =============================================================================
def print_findings(comparison, corr_matrix):
    """Print a summary of key findings from the analysis."""
    print_header("6. KEY FINDINGS SUMMARY")

    # Top 5 features by SHAP
    top5 = comparison.head(5)
    print("\nTop 5 Most Important Features (by Mean Absolute SHAP):")
    print("-" * 60)
    for idx, row in top5.iterrows():
        print(f"  {idx}. {row['feature']:<30s}  SHAP rank: {int(row['shap_rank']):>2d}  "
              f"Gain rank: {int(row['gain_rank']):>2d}  Split rank: {int(row['split_rank']):>2d}")

    # Biggest rank disagreements between LGBM Gain and SHAP
    comparison_copy = comparison.copy()
    comparison_copy["rank_diff_gain_shap"] = (
        comparison_copy["gain_rank"] - comparison_copy["shap_rank"]
    ).abs()
    disagreements = comparison_copy.nlargest(5, "rank_diff_gain_shap")

    print(f"\nTop 5 Biggest Rank Disagreements (LGBM Gain vs SHAP):")
    print("-" * 60)
    for _, row in disagreements.iterrows():
        direction = "↑" if row["gain_rank"] < row["shap_rank"] else "↓"
        print(f"  {row['feature']:<30s}  Gain rank: {int(row['gain_rank']):>2d}  "
              f"SHAP rank: {int(row['shap_rank']):>2d}  "
              f"Diff: {int(row['rank_diff_gain_shap']):>2d} {direction}")

    # Overall agreement level
    rho_gain_shap = corr_matrix.loc["LGBM Gain", "SHAP"]
    rho_split_shap = corr_matrix.loc["LGBM Split", "SHAP"]
    rho_gain_split = corr_matrix.loc["LGBM Gain", "LGBM Split"]

    print(f"\nOverall Agreement Level (Spearman ρ):")
    print("-" * 60)
    print(f"  LGBM Gain  vs SHAP:   ρ = {rho_gain_shap:.4f}")
    print(f"  LGBM Split vs SHAP:   ρ = {rho_split_shap:.4f}")
    print(f"  LGBM Gain  vs Split:  ρ = {rho_gain_split:.4f}")

    # Interpret correlations
    def interpret_correlation(rho):
        rho_abs = abs(rho)
        if rho_abs >= 0.9:
            return "Very strong agreement"
        elif rho_abs >= 0.7:
            return "Strong agreement"
        elif rho_abs >= 0.5:
            return "Moderate agreement"
        elif rho_abs >= 0.3:
            return "Weak agreement"
        else:
            return "Very weak / no agreement"

    print(f"\nInterpretation:")
    print(f"  • LGBM Gain vs SHAP:   {interpret_correlation(rho_gain_shap)}")
    print(f"  • LGBM Split vs SHAP:  {interpret_correlation(rho_split_shap)}")
    print(f"  • LGBM Gain vs Split:  {interpret_correlation(rho_gain_split)}")

    avg_corr = (rho_gain_shap + rho_split_shap + rho_gain_split) / 3
    print(f"\n  Average pairwise correlation: {avg_corr:.4f}")
    if avg_corr >= 0.8:
        print("  → The three methods show HIGH overall agreement on feature rankings.")
    elif avg_corr >= 0.6:
        print("  → The three methods show MODERATE overall agreement on feature rankings.")
    else:
        print("  → The three methods show LOW overall agreement — rankings differ substantially.")

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE")
    print("=" * 80)


# =============================================================================
# 7. Bootstrap Stability Comparison
# =============================================================================
def compare_with_bootstrap(X_train_df, comparison_df):
    """Compare LightGBM and SHAP results against bootstrap stability complexity scores.

    Parameters
    ----------
    X_train_df : pd.DataFrame
        Training data including the target column.
    comparison_df : pd.DataFrame
        The comparison DataFrame from compare_importances().
    """
    print_header("7. BOOTSTRAP STABILITY COMPARISON")

    if BootstrapStability is None:
        print("Skipping Section 7: bootstrap_stability package not available.")
        return

    # --- 7a. Run Bootstrap Stability Panel Analysis ---
    print("\nRunning Bootstrap Stability Panel Analysis (n_resamples=15)...")
    bs = BootstrapStability(n_resamples=15, random_state=42)
    panel_results = bs.fit_panel(X_train_df, target_col=TARGET_COL)

    summary_df = panel_results["summary"]
    print(f"\nBootstrap stability analysis completed for {len(summary_df)} features.")

    excluded_features = set(X_train_df.columns) - set(summary_df["feature"].tolist()) - {TARGET_COL}
    if excluded_features:
        print(f"\nNote: The following features were excluded by the bootstrap toolkit")
        print(f"      (likely detected as categorical with ≤10 unique values):")
        for feat in sorted(excluded_features):
            n_unique = X_train_df[feat].nunique()
            print(f"        - {feat} ({n_unique} unique values)")

    # --- 7b. Extract complexity scores from panel results ---
    bs_scores = summary_df[["feature", "complexity_score", "censoring_flag"]].copy()
    bs_scores = bs_scores.rename(columns={"complexity_score": "bs_complexity_score"})

    # --- 7c. Merge all rankings into one comparison DataFrame ---
    # Start from the existing comparison DataFrame
    merged = comparison_df[["feature"]].copy()

    # Add LGBM gain importance and rank
    gain_norm = comparison_df.set_index("feature")["gain_importance_norm"]
    gain_rank = comparison_df.set_index("feature")["gain_rank"]
    merged["lgbm_gain_importance"] = gain_norm.reindex(merged["feature"]).values
    merged["lgbm_gain_rank"] = gain_rank.reindex(merged["feature"]).values.astype(int)

    # Add LGBM split importance and rank
    split_norm = comparison_df.set_index("feature")["split_importance_norm"]
    split_rank = comparison_df.set_index("feature")["split_rank"]
    merged["lgbm_split_importance"] = split_norm.reindex(merged["feature"]).values
    merged["lgbm_split_rank"] = split_rank.reindex(merged["feature"]).values.astype(int)

    # Add SHAP importance and rank
    shap_norm = comparison_df.set_index("feature")["mean_abs_shap_norm"]
    shap_rank = comparison_df.set_index("feature")["shap_rank"]
    merged["shap_importance"] = shap_norm.reindex(merged["feature"]).values
    merged["shap_rank"] = shap_rank.reindex(merged["feature"]).values.astype(int)

    # Add bootstrap stability complexity score and censoring flag
    bs_indexed = bs_scores.set_index("feature")
    merged["complexity_score"] = bs_indexed["bs_complexity_score"].reindex(merged["feature"]).values
    merged["censoring_flag"] = bs_indexed["censoring_flag"].reindex(merged["feature"]).values

    # Compute complexity rank (ascending — lower complexity = more stable = rank 1)
    # Only rank features that have a valid complexity score
    valid_complexity = merged["complexity_score"].notna()
    merged["complexity_rank"] = np.nan
    if valid_complexity.sum() > 0:
        merged.loc[valid_complexity, "complexity_rank"] = (
            merged.loc[valid_complexity, "complexity_score"]
            .rank(ascending=True, method="min")
            .values
        )

    # Sort by SHAP rank (most important first)
    merged = merged.sort_values("shap_rank").reset_index(drop=True)
    merged.index += 1
    merged.index.name = "Rank"

    print("\nFull Comparison Table (sorted by SHAP rank):")
    print(merged.to_string())

    # --- 7d. Compute Spearman rank correlations across ALL four methods ---
    # Only use features with valid complexity scores
    valid_mask = merged["complexity_rank"].notna()
    valid_merged = merged[valid_mask].copy()

    if len(valid_merged) < 3:
        print("\nNot enough features with valid complexity scores for correlation analysis.")
        save_path = os.path.join(OUTPUT_DIR, "full_comparison.csv")
        merged.to_csv(save_path, index=True)
        print(f"Saved: {save_path}")
        return

    corr_gain_shap, _ = spearmanr(valid_merged["lgbm_gain_rank"], valid_merged["shap_rank"])
    corr_gain_split, _ = spearmanr(valid_merged["lgbm_gain_rank"], valid_merged["lgbm_split_rank"])
    corr_gain_complexity, _ = spearmanr(valid_merged["lgbm_gain_rank"], valid_merged["complexity_rank"])
    corr_split_shap, _ = spearmanr(valid_merged["lgbm_split_rank"], valid_merged["shap_rank"])
    corr_split_complexity, _ = spearmanr(valid_merged["lgbm_split_rank"], valid_merged["complexity_rank"])
    corr_shap_complexity, p_shap_complexity = spearmanr(valid_merged["shap_rank"], valid_merged["complexity_rank"])

    corr_matrix_4 = pd.DataFrame({
        "LGBM Gain": [1.0, corr_gain_split, corr_gain_shap, corr_gain_complexity],
        "LGBM Split": [corr_gain_split, 1.0, corr_split_shap, corr_split_complexity],
        "SHAP": [corr_gain_shap, corr_split_shap, 1.0, corr_shap_complexity],
        "Complexity": [corr_gain_complexity, corr_split_complexity, corr_shap_complexity, 1.0],
    }, index=["LGBM Gain", "LGBM Split", "SHAP", "Complexity"])

    print(f"\nSpearman Rank Correlations (4 methods, {len(valid_merged)} features):")
    print(corr_matrix_4.to_string(float_format="{:.4f}".format))
    print(f"\nKey correlation — SHAP vs Complexity: ρ = {corr_shap_complexity:.4f} (p = {p_shap_complexity:.6f})")

    # --- 7e. Create 4-method comparison plot ---
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))

    # Subplot 1: Scatter of SHAP rank vs complexity rank
    ax1 = axes[0, 0]
    ax1.scatter(
        valid_merged["shap_rank"], valid_merged["complexity_rank"],
        c="forestgreen", edgecolors="darkgreen", s=80, alpha=0.7, zorder=3,
    )
    max_rank = max(valid_merged["shap_rank"].max(), valid_merged["complexity_rank"].max())
    ax1.plot([0, max_rank + 1], [0, max_rank + 1], "k--", alpha=0.5, label="Perfect agreement")
    ax1.set_xlabel("SHAP Rank", fontsize=12)
    ax1.set_ylabel("Complexity Rank (lower = more stable)", fontsize=12)
    ax1.set_title(
        f"SHAP Rank vs Complexity Rank\n(ρ = {corr_shap_complexity:.3f}, p = {p_shap_complexity:.4f})",
        fontsize=13, fontweight="bold",
    )
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)

    # Subplot 2: Scatter of LGBM gain rank vs complexity rank
    ax2 = axes[0, 1]
    ax2.scatter(
        valid_merged["lgbm_gain_rank"], valid_merged["complexity_rank"],
        c="coral", edgecolors="darkred", s=80, alpha=0.7, zorder=3,
    )
    max_rank2 = max(valid_merged["lgbm_gain_rank"].max(), valid_merged["complexity_rank"].max())
    ax2.plot([0, max_rank2 + 1], [0, max_rank2 + 1], "k--", alpha=0.5, label="Perfect agreement")
    ax2.set_xlabel("LGBM Gain Rank", fontsize=12)
    ax2.set_ylabel("Complexity Rank (lower = more stable)", fontsize=12)
    ax2.set_title(
        f"LGBM Gain Rank vs Complexity Rank\n(ρ = {corr_gain_complexity:.3f})",
        fontsize=13, fontweight="bold",
    )
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)

    # Subplot 3: Heatmap of Spearman correlations across all 4 methods
    ax3 = axes[1, 0]
    im = ax3.imshow(corr_matrix_4.values, cmap="RdYlGn", vmin=-1.0, vmax=1.0, aspect="auto")
    ax3.set_xticks(range(len(corr_matrix_4.columns)))
    ax3.set_xticklabels(corr_matrix_4.columns, fontsize=10, rotation=30, ha="right")
    ax3.set_yticks(range(len(corr_matrix_4.index)))
    ax3.set_yticklabels(corr_matrix_4.index, fontsize=10)
    ax3.set_title("Spearman Rank Correlations (4 Methods)", fontsize=13, fontweight="bold")

    for i in range(len(corr_matrix_4.index)):
        for j in range(len(corr_matrix_4.columns)):
            val = corr_matrix_4.values[i, j]
            ax3.text(j, i, f"{val:.3f}", ha="center", va="center",
                     fontsize=11, fontweight="bold",
                     color="white" if abs(val) > 0.7 else "black")

    fig.colorbar(im, ax=ax3, shrink=0.8, label="Spearman ρ")

    # Subplot 4: Grouped horizontal bar chart for top 15 features
    ax4 = axes[1, 1]
    top15 = merged.head(15).copy()
    # Filter to features with valid complexity scores
    top15_valid = top15[top15["complexity_score"].notna()].head(15)
    if len(top15_valid) > 0:
        y_pos = np.arange(len(top15_valid))
        bar_height = 0.35

        # SHAP importance on left x-axis
        ax4.barh(y_pos - bar_height / 2, top15_valid["shap_importance"].values, bar_height,
                 label="SHAP Importance (norm)", color="forestgreen", alpha=0.8)

        # Complexity score on right x-axis
        ax4_right = ax4.twiny()
        ax4_right.barh(y_pos + bar_height / 2, top15_valid["complexity_score"].values, bar_height,
                       label="Complexity Score", color="mediumpurple", alpha=0.8)

        ax4.set_yticks(y_pos)
        ax4.set_yticklabels(top15_valid["feature"].values, fontsize=9)
        ax4.set_xlabel("Normalized SHAP Importance", fontsize=11, color="forestgreen")
        ax4_right.set_xlabel("Complexity Score", fontsize=11, color="mediumpurple")
        ax4.set_title("Top Features: SHAP vs Complexity", fontsize=13, fontweight="bold")

        # Combined legend
        lines1, labels1 = ax4.get_legend_handles_labels()
        lines2, labels2 = ax4_right.get_legend_handles_labels()
        ax4.legend(lines1 + lines2, labels1 + labels2, fontsize=9, loc="lower right")
        ax4.grid(True, axis="x", alpha=0.3)
    else:
        ax4.text(0.5, 0.5, "No features with valid complexity scores",
                 ha="center", va="center", transform=ax4.transAxes, fontsize=14)
        ax4.set_title("Top Features: SHAP vs Complexity", fontsize=13, fontweight="bold")

    plt.tight_layout()
    save_path = os.path.join(OUTPUT_DIR, "bootstrap_vs_shap_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\nSaved: {save_path}")

    # --- 7f. Print bootstrap stability insights ---
    print("\n" + "-" * 60)
    print("  BOOTSTRAP STABILITY INSIGHTS")
    print("-" * 60)

    # Top 5 most stable features (lowest complexity score)
    stable = valid_merged.nsmallest(5, "complexity_score")
    print("\nTop 5 Most Stable Features (lowest complexity score):")
    for _, row in stable.iterrows():
        print(f"  • {row['feature']:<30s}  complexity={row['complexity_score']:.4f}  "
              f"SHAP rank={int(row['shap_rank']):>2d}  Gain rank={int(row['lgbm_gain_rank']):>2d}")

    # Top 5 least stable features (highest complexity score)
    unstable = valid_merged.nlargest(5, "complexity_score")
    print("\nTop 5 Least Stable Features (highest complexity score):")
    for _, row in unstable.iterrows():
        print(f"  • {row['feature']:<30s}  complexity={row['complexity_score']:.4f}  "
              f"SHAP rank={int(row['shap_rank']):>2d}  Gain rank={int(row['lgbm_gain_rank']):>2d}")

    # Features where bootstrap stability disagrees with SHAP/LGBM
    print("\nFeatures Where Bootstrap Stability Disagrees with SHAP/LGBM:")
    print("  (High importance but high instability, or low importance but high stability)")

    # Normalize ranks to [0, 1] for comparison
    n_features = len(valid_merged)
    disagreement_df = valid_merged.copy()
    disagreement_df["shap_rank_norm"] = disagreement_df["shap_rank"] / n_features
    disagreement_df["complexity_rank_norm"] = disagreement_df["complexity_rank"] / n_features
    disagreement_df["importance_stability_gap"] = (
        disagreement_df["shap_rank_norm"] - disagreement_df["complexity_rank_norm"]
    ).abs()

    # High importance (low SHAP rank) but high instability (high complexity rank)
    high_imp_unstable = disagreement_df[
        (disagreement_df["shap_rank"] <= n_features // 3)
        & (disagreement_df["complexity_rank"] > 2 * n_features // 3)
    ]
    if len(high_imp_unstable) > 0:
        print("\n  ⚠ High importance but UNSTABLE:")
        for _, row in high_imp_unstable.iterrows():
            print(f"    - {row['feature']:<30s}  SHAP rank={int(row['shap_rank']):>2d}  "
                  f"Complexity rank={int(row['complexity_rank']):>2d}  "
                  f"(important but unreliable across bootstraps)")
    else:
        print("\n  ✓ No features with high importance and high instability found.")

    # Low importance (high SHAP rank) but high stability (low complexity rank)
    low_imp_stable = disagreement_df[
        (disagreement_df["shap_rank"] > 2 * n_features // 3)
        & (disagreement_df["complexity_rank"] <= n_features // 3)
    ]
    if len(low_imp_stable) > 0:
        print("\n  ℹ Low importance but STABLE:")
        for _, row in low_imp_stable.iterrows():
            print(f"    - {row['feature']:<30s}  SHAP rank={int(row['shap_rank']):>2d}  "
                  f"Complexity rank={int(row['complexity_rank']):>2d}  "
                  f"(stable but not driving predictions)")
    else:
        print("\n  ✓ No features with low importance and high stability found.")

    # Interpretation
    print("\n" + "-" * 60)
    print("  INTERPRETATION")
    print("-" * 60)

    # Check if top important features are also the most stable
    top5_shap = set(valid_merged.nsmallest(5, "shap_rank")["feature"].tolist())
    top5_stable = set(valid_merged.nsmallest(5, "complexity_score")["feature"].tolist())
    overlap = top5_shap & top5_stable

    print(f"\n  Overlap between Top-5 SHAP features and Top-5 most stable: {len(overlap)}/5")
    if overlap:
        for feat in overlap:
            print(f"    ✓ {feat}")
    else:
        print("    (no overlap)")

    if corr_shap_complexity > 0.7:
        print("\n  → Strong positive correlation between SHAP importance and complexity.")
        print("    The most important features tend to be MORE complex/less stable.")
        print("    This suggests model predictions may be sensitive to training data variation.")
    elif corr_shap_complexity > 0.3:
        print("\n  → Moderate positive correlation between SHAP importance and complexity.")
        print("    Some important features show instability — worth investigating individually.")
    elif corr_shap_complexity > -0.3:
        print("\n  → Weak/no correlation between SHAP importance and complexity.")
        print("    Feature importance and stability are largely independent dimensions.")
    else:
        print("\n  → Negative correlation between SHAP importance and complexity.")
        print("    The most important features tend to be MORE stable.")
        print("    This is a positive signal for model reliability.")

    print("\n" + "=" * 80)
    print("  EXPERIMENT COMPLETE (WITH BOOTSTRAP STABILITY)")
    print("=" * 80)

    # --- 7g. Save the full comparison table ---
    save_path = os.path.join(OUTPUT_DIR, "full_comparison.csv")
    merged.to_csv(save_path, index=True)
    print(f"Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================
if __name__ == "__main__":
    # 1. Load and prepare data
    X_train, X_test, y_train, y_test, X_holdout, feature_names = load_and_prepare_data()

    # 2. Train LightGBM classifier
    model = train_lightgbm(X_train, y_train, X_test, y_test)

    # 3. LightGBM Feature Importance
    split_df, gain_df = analyze_lgbm_importance(model, feature_names)

    # 4. SHAP Analysis
    shap_df, shap_vals = analyze_shap(model, X_holdout, feature_names)

    # 5. Comparison Analysis
    comparison, corr_matrix = compare_importances(split_df, gain_df, shap_df, feature_names)

    # 6. Key Findings Summary
    print_findings(comparison, corr_matrix)

    # 7. Bootstrap Stability Comparison
    # Build a training DataFrame that includes the target column
    X_train_df = X_train.copy()
    X_train_df[TARGET_COL] = y_train.values
    compare_with_bootstrap(X_train_df, comparison)
