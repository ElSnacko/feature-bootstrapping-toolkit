"""
Marginal vs SHAP Stability Comparison Demo

This script demonstrates the key insight from the critique:
"train stability in marginal space has no reason to predict holdout stability
in the space where the model actually makes decisions"

The script compares:
1. Marginal Stability (existing toolkit) - measures feature distribution stability
2. SHAP Stability (new module) - measures model contribution stability

Key scenarios to identify:
- False alarms: High marginal instability but low SHAP instability
- Missed risks: Low marginal instability but high SHAP instability
- Agreement: Both methods agree
"""

import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# Set matplotlib backend before importing
import matplotlib
matplotlib.use('Agg')

# Try importing optional dependencies
try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False

from bootstrap_stability import (
    BootstrapStability,
    SHAPStability,
    TrainHoldoutStability,
    print_report,
)


def load_credit_card_data(filepath: str) -> pd.DataFrame:
    """Load and preprocess the credit card dataset."""
    print(f"Loading data from: {filepath}")
    
    # Read the Excel file
    df = pd.read_excel(filepath, header=1)
    
    # Clean column names
    df.columns = [c.lower().replace(' ', '_') for c in df.columns]
    
    # Rename target column
    if 'default_payment_next_month' in df.columns:
        df = df.rename(columns={'default_payment_next_month': 'default'})
    
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def create_temporal_split(df: pd.DataFrame, train_frac: float = 0.8):
    """
    Create a temporal train/holdout split.
    
    Since the credit card data doesn't have explicit time ordering,
    we simulate temporal drift by:
    1. Using the first 80% as "train" (older period)
    2. Using the last 20% as "holdout" (newer period)
    """
    n = len(df)
    train_end = int(n * train_frac)
    
    train_df = df.iloc[:train_end].copy()
    holdout_df = df.iloc[train_end:].copy()
    
    print(f"\nTemporal Split:")
    print(f"  Train: {len(train_df)} samples (indices 0-{train_end-1})")
    print(f"  holdout:   {len(holdout_df)} samples (indices {train_end}-{n-1})")
    
    return train_df, holdout_df


def get_feature_columns(df: pd.DataFrame) -> list:
    """Get feature columns excluding ID and target."""
    exclude_cols = ['id', 'default']
    feature_cols = [c for c in df.columns if c not in exclude_cols]
    return feature_cols


def run_marginal_stability_analysis(df: pd.DataFrame, feature_cols: list) -> dict:
    """
    Run marginal stability analysis using the existing toolkit.
    
    This measures how stable the feature DISTRIBUTIONS are across
    bootstrap resamples - NOT how stable their contributions to the model are.
    """
    print("\n" + "="*60)
    print("MARGINAL STABILITY ANALYSIS (Existing Toolkit)")
    print("="*60)
    print("Measuring stability of feature distributions...")
    
    analyzer = BootstrapStability(
        resample_frac=0.8,
        n_resamples=20,
        n_bins=5,
        min_events=20,
        n_jobs=-1,
        random_state=42,
    )
    
    results = {}
    
    for feature in feature_cols:
        print(f"\nAnalyzing {feature}...")
        try:
            result = analyzer.fit(df, feature_col=feature, target_col='default')
            results[feature] = {
                'complexity_score': result.get('complexity_score', np.nan),
                'extrapolated_complexity': result.get('extrapolated_complexity', {}),
                'r2': result.get('r2', np.nan),
                'feature_type': result.get('feature_type', 'unknown'),
            }
        except Exception as e:
            print(f"  Error analyzing {feature}: {e}")
            results[feature] = {
                'complexity_score': np.nan,
                'extrapolated_complexity': {},
                'r2': np.nan,
                'feature_type': 'error',
            }
    
    return results


def create_model_factory():
    """Create a model factory for LightGBM."""
    def factory():
        if not HAS_LGB:
            raise ImportError("LightGBM is required. Install with: pip install lightgbm")
        return lgb.LGBMClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            random_state=42,
            verbose=-1,
        )
    return factory


def run_shap_stability_analysis(
    train_df: pd.DataFrame, 
    holdout_df: pd.DataFrame,
    feature_cols: list
) -> dict:
    """
    Run SHAP-based stability analysis using the new module.
    
    This measures how stable the feature CONTRIBUTIONS to the model are
    between train and holdout sets - directly measuring what matters for
    production model behavior.
    """
    print("\n" + "="*60)
    print("SHAP STABILITY ANALYSIS (New Module)")
    print("="*60)
    print("Measuring stability of feature contributions to model...")
    
    X_train = train_df[feature_cols].values
    y_train = train_df['default'].values
    X_holdout = holdout_df[feature_cols].values
    y_holdout = holdout_df['default'].values
    
    # Run Train/holdout SHAP stability analysis
    holdout_stability = TrainHoldoutStability(
        model_factory=create_model_factory(),
        explainer_type='tree',
        shap_subsample=1000,  # Subsample for speed
        top_k=10,
        random_state=42,
        verbose=1,
    )
    
    results = holdout_stability.fit(
        X_train, y_train, X_holdout, y_holdout,
        feature_names=feature_cols
    )
    
    return results


def compare_stability_metrics(
    marginal_results: dict,
    shap_results: dict,
    feature_cols: list
) -> pd.DataFrame:
    """
    Create a comparison table of marginal vs SHAP stability.
    
    Returns a DataFrame with per-feature comparison.
    """
    comparison_data = []
    
    # Get per-feature drift from SHAP results
    feature_drift = shap_results.get('feature_drift', {})
    
    for feature in feature_cols:
        # Marginal metrics
        marginal = marginal_results.get(feature, {})
        marginal_complexity = marginal.get('complexity_score', np.nan)
        
        # SHAP metrics
        shap_metrics = feature_drift.get(feature, {})
        shap_drift_score = shap_metrics.get('drift_score', np.nan)
        shap_direction_flip = shap_metrics.get('direction_flip', np.nan)
        shap_magnitude_change = shap_metrics.get('magnitude_change', np.nan)
        
        # Determine divergence
        # High marginal (>0.3) + Low SHAP (<0.15) = False alarm
        # Low marginal (<0.15) + High SHAP (>0.3) = Missed risk
        marginal_high = marginal_complexity > 0.3
        marginal_low = marginal_complexity < 0.15
        shap_high = shap_drift_score > 0.25
        shap_low = shap_drift_score < 0.15
        
        if marginal_high and shap_low:
            divergence = "FALSE ALARM"
        elif marginal_low and shap_high:
            divergence = "MISSED RISK"
        elif marginal_low and shap_low:
            divergence = "Agree (Stable)"
        elif marginal_high and shap_high:
            divergence = "Agree (Unstable)"
        else:
            divergence = "Mixed"
        
        comparison_data.append({
            'Feature': feature,
            'Marginal_Complexity': marginal_complexity,
            'SHAP_Drift_Score': shap_drift_score,
            'SHAP_Direction_Flip': shap_direction_flip,
            'SHAP_Magnitude_Change': shap_magnitude_change,
            'Divergence': divergence,
        })
    
    df = pd.DataFrame(comparison_data)
    return df


def print_comparison_report(comparison_df: pd.DataFrame):
    """Print a detailed comparison report."""
    print("\n" + "="*70)
    print("MARGINAL vs SHAP STABILITY COMPARISON REPORT")
    print("="*70)
    
    # False alarms
    false_alarms = comparison_df[comparison_df['Divergence'] == 'FALSE ALARM']
    if len(false_alarms) > 0:
        print("\n🚨 FALSE ALARMS (Marginal over-estimates risk):")
        print("   Features with unstable marginal distribution but stable model contribution")
        print("   " + "-"*60)
        for _, row in false_alarms.iterrows():
            print(f"   • {row['Feature']}:")
            print(f"     Marginal={row['Marginal_Complexity']:.3f}, SHAP={row['SHAP_Drift_Score']:.3f}")
            print(f"     → Marginal analysis flags this as risky, but model contribution is stable")
    
    # Missed risks
    missed_risks = comparison_df[comparison_df['Divergence'] == 'MISSED RISK']
    if len(missed_risks) > 0:
        print("\n⚠️  MISSED RISKS (Marginal under-estimates risk):")
        print("   Features with stable marginal distribution but unstable model contribution")
        print("   " + "-"*60)
        for _, row in missed_risks.iterrows():
            print(f"   • {row['Feature']}:")
            print(f"     Marginal={row['Marginal_Complexity']:.3f}, SHAP={row['SHAP_Drift_Score']:.3f}")
            print(f"     → Marginal analysis misses this risk, but model contribution is unstable!")
    
    # Agreement - Stable
    agree_stable = comparison_df[comparison_df['Divergence'] == 'Agree (Stable)']
    if len(agree_stable) > 0:
        print(f"\n✅ AGREEMENT (Both methods see stability): {len(agree_stable)} features")
        print("   " + "-"*60)
        for _, row in agree_stable.head(5).iterrows():
            print(f"   • {row['Feature']}: Marginal={row['Marginal_Complexity']:.3f}, SHAP={row['SHAP_Drift_Score']:.3f}")
        if len(agree_stable) > 5:
            print(f"   ... and {len(agree_stable) - 5} more")
    
    # Agreement - Unstable
    agree_unstable = comparison_df[comparison_df['Divergence'] == 'Agree (Unstable)']
    if len(agree_unstable) > 0:
        print(f"\n⚡ AGREEMENT (Both methods see instability): {len(agree_unstable)} features")
        print("   " + "-"*60)
        for _, row in agree_unstable.iterrows():
            print(f"   • {row['Feature']}: Marginal={row['Marginal_Complexity']:.3f}, SHAP={row['SHAP_Drift_Score']:.3f}")
    
    # Mixed
    mixed = comparison_df[comparison_df['Divergence'] == 'Mixed']
    if len(mixed) > 0:
        print(f"\n❓ MIXED RESULTS: {len(mixed)} features")
        print("   " + "-"*60)
        for _, row in mixed.iterrows():
            print(f"   • {row['Feature']}: Marginal={row['Marginal_Complexity']:.3f}, SHAP={row['SHAP_Drift_Score']:.3f}")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    total = len(comparison_df)
    print(f"\nTotal features analyzed: {total}")
    print(f"  False Alarms:  {len(false_alarms):3d} ({100*len(false_alarms)/total:.1f}%)")
    print(f"  Missed Risks:  {len(missed_risks):3d} ({100*len(missed_risks)/total:.1f}%)")
    print(f"  Agree Stable:  {len(agree_stable):3d} ({100*len(agree_stable)/total:.1f}%)")
    print(f"  Agree Unstable:{len(agree_unstable):3d} ({100*len(agree_unstable)/total:.1f}%)")
    print(f"  Mixed:         {len(mixed):3d} ({100*len(mixed)/total:.1f}%)")
    
    # Key insight
    disagreement_rate = (len(false_alarms) + len(missed_risks)) / total
    print(f"\n📊 Disagreement Rate: {100*disagreement_rate:.1f}%")
    print(f"   ({len(false_alarms) + len(missed_risks)} features where marginal and SHAP disagree)")
    
    # Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)
    print("""
For production monitoring, prioritize features with high SHAP instability,
as these directly impact model decisions. Marginal distribution stability
is a proxy that may generate false alarms or miss genuine risks.

Key findings:
- FALSE ALARMS represent wasted monitoring effort
- MISSED RISKS represent undetected model degradation
- SHAP-based stability directly measures what matters for predictions
""")


def create_visualization(comparison_df: pd.DataFrame, output_path: str = None):
    """Create a scatter plot comparing marginal vs SHAP stability."""
    if not HAS_MATPLOTLIB:
        print("\nMatplotlib not available. Skipping visualization.")
        return
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # Color by divergence type
    colors = {
        'FALSE ALARM': '#ff7f0e',      # Orange
        'MISSED RISK': '#d62728',       # Red
        'Agree (Stable)': '#2ca02c',    # Green
        'Agree (Unstable)': '#9467bd',  # Purple
        'Mixed': '#7f7f7f',             # Gray
    }
    
    for divergence, color in colors.items():
        mask = comparison_df['Divergence'] == divergence
        if mask.sum() > 0:
            subset = comparison_df[mask]
            ax.scatter(
                subset['Marginal_Complexity'],
                subset['SHAP_Drift_Score'],
                c=color,
                label=f'{divergence} ({len(subset)})',
                alpha=0.7,
                s=100,
                edgecolors='white',
                linewidth=0.5,
            )
    
    # Add diagonal reference line
    lims = [0, max(comparison_df['Marginal_Complexity'].max(), 
                   comparison_df['SHAP_Drift_Score'].max()) + 0.1]
    ax.plot(lims, lims, 'k--', alpha=0.5, label='Perfect agreement')
    
    # Add threshold lines
    ax.axhline(y=0.25, color='red', linestyle=':', alpha=0.3, label='SHAP threshold')
    ax.axvline(x=0.3, color='orange', linestyle=':', alpha=0.3, label='Marginal threshold')
    
    # Annotate divergent features
    for _, row in comparison_df.iterrows():
        if row['Divergence'] in ['FALSE ALARM', 'MISSED RISK']:
            ax.annotate(
                row['Feature'][:15],
                (row['Marginal_Complexity'], row['SHAP_Drift_Score']),
                xytext=(5, 5),
                textcoords='offset points',
                fontsize=8,
                alpha=0.8,
            )
    
    ax.set_xlabel('Marginal Complexity Score', fontsize=12)
    ax.set_ylabel('SHAP Drift Score', fontsize=12)
    ax.set_title('Marginal vs SHAP Stability: Credit Card Default Model\n'
                 '(Lower = More Stable)', fontsize=14)
    ax.legend(loc='upper left', framealpha=0.9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved to: {output_path}")
    
    plt.show()


def print_full_table(comparison_df: pd.DataFrame):
    """Print the full comparison table."""
    print("\n" + "="*100)
    print("FULL COMPARISON TABLE")
    print("="*100)
    
    # Format for display
    display_df = comparison_df.copy()
    display_df['Marginal_Complexity'] = display_df['Marginal_Complexity'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display_df['SHAP_Drift_Score'] = display_df['SHAP_Drift_Score'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display_df['SHAP_Direction_Flip'] = display_df['SHAP_Direction_Flip'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    display_df['SHAP_Magnitude_Change'] = display_df['SHAP_Magnitude_Change'].apply(
        lambda x: f"{x:.3f}" if pd.notna(x) else "N/A"
    )
    
    print(display_df.to_string(index=False))


def main():
    """Main entry point for the comparison demo."""
    print("="*70)
    print("MARGINAL vs SHAP STABILITY COMPARISON DEMO")
    print("="*70)
    print("""
This demo validates the critique that marginal distribution stability
does not necessarily predict model contribution stability.

The key insight:
> "train stability in marginal space has no reason to predict holdout stability
> in the space where the model actually makes decisions"
""")
    
    # Check dependencies
    if not HAS_LGB:
        print("ERROR: LightGBM is required. Install with: pip install lightgbm")
        return
    
    # Load data
    data_path = os.path.join(os.path.dirname(__file__), "..", "..", "default+of+credit+card+clients", "default of credit card clients.xls")
    df = load_credit_card_data(data_path)
    
    # Create temporal split
    train_df, holdout_df = create_temporal_split(df, train_frac=0.8)
    
    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"\nFeatures to analyze: {len(feature_cols)}")
    
    # Run marginal stability analysis (on train data)
    marginal_results = run_marginal_stability_analysis(train_df, feature_cols)
    
    # Run SHAP stability analysis (train vs holdout)
    shap_results = run_shap_stability_analysis(train_df, holdout_df, feature_cols)
    
    # Compare results
    comparison_df = compare_stability_metrics(marginal_results, shap_results, feature_cols)
    
    # Print full table
    print_full_table(comparison_df)
    
    # Print detailed report
    print_comparison_report(comparison_df)
    
    # Create visualization
    if HAS_MATPLOTLIB:
        create_visualization(
            comparison_df, 
            output_path="marginal_vs_shap_comparison.png"
        )
    
    # Print SHAP drift summary
    print("\n" + "="*70)
    print("SHAP DRIFT SUMMARY")
    print("="*70)
    print(f"Overall Drift Score: {shap_results.get('overall_drift_score', 'N/A')}")
    print(f"Drift Grade: {shap_results.get('drift_grade', 'N/A')}")
    
    drift_metrics = shap_results.get('drift_metrics', {})
    for name, metric in drift_metrics.items():
        if hasattr(metric, 'flagged'):
            status = "🚨 FLAGGED" if metric.flagged else "✓ OK"
            print(f"  {name}: {metric.drift:.3f} {status}")
    
    print("\n" + "="*70)
    print("DEMO COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
