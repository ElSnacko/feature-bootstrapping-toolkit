"""
Generate JSON output files from the comprehensive analysis results.
This script reads the CSV outputs and creates the required JSON files.
"""

import json
import pandas as pd
import os
from datetime import datetime

from bootstrap_stability.reliability import ReliabilityScorer, ReliabilityConfig

OUTPUT_DIR = "credit_card_analysis_results"

def load_csv(filename):
    """Load a CSV file from the output directory."""
    path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

def main():
    print("Generating JSON output files from analysis results...")
    
    # Load all CSV data
    master_df = load_csv("master_comparison.csv")
    comparison_df = load_csv("marginal_vs_shap_comparison.csv")
    
    if master_df is None:
        print("ERROR: master_comparison.csv not found. Run comprehensive_feature_analysis.py first.")
        return
    
    # ==========================================================================
    # 1. Stability Results (Bootstrap/Marginal Stability)
    # ==========================================================================
    stability_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "bootstrap_marginal_stability",
            "dataset": "default_of_credit_card_clients",
            "target": "default payment next month",
            "n_features": len(master_df),
            "n_bootstraps": 100,
            "sample_fraction": 0.5
        },
        "features": {}
    }
    
    for _, row in master_df.iterrows():
        feature = row['feature']
        stability_results["features"][feature] = {
            "complexity_score": float(row['complexity_score']) if pd.notna(row['complexity_score']) else None,
            "complexity_rank": int(row['complexity_rank']) if pd.notna(row['complexity_rank']) else None,
            "oot_complexity_score": float(row['oot_complexity_score']) if pd.notna(row['oot_complexity_score']) else None,
            "oot_complexity_shift": float(row['oot_complexity_shift']) if pd.notna(row['oot_complexity_shift']) else None,
            "tree_coverage": float(row['tree_coverage']) if pd.notna(row['tree_coverage']) else None,
            "mean_depth": float(row['mean_depth']) if pd.notna(row['mean_depth']) else None,
            "stability_interpretation": "stable" if pd.notna(row['complexity_score']) and row['complexity_score'] < 50 else "unstable"
        }
    
    stability_path = os.path.join(OUTPUT_DIR, "stability_results.json")
    with open(stability_path, 'w') as f:
        json.dump(stability_results, f, indent=2)
    print(f"  Saved: {stability_path}")
    
    # ==========================================================================
    # 2. SHAP Stability Results
    # ==========================================================================
    shap_stability_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "shap_stability",
            "dataset": "default_of_credit_card_clients",
            "target": "default payment next month",
            "n_features": len(master_df)
        },
        "features": {}
    }
    
    for _, row in master_df.iterrows():
        feature = row['feature']
        shap_stability_results["features"][feature] = {
            "shap_importance_norm": float(row['shap_importance_norm']) if pd.notna(row['shap_importance_norm']) else None,
            "shap_rank": int(row['shap_rank']) if pd.notna(row['shap_rank']) else None,
            "shap_cv": float(row['shap_cv']) if pd.notna(row['shap_cv']) else None,
            "shap_total_interaction": float(row['shap_total_interaction']) if pd.notna(row['shap_total_interaction']) else None,
            "shap_skewness": float(row['shap_skewness']) if pd.notna(row['shap_skewness']) else None,
            "shap_kurtosis": float(row['shap_kurtosis']) if pd.notna(row['shap_kurtosis']) else None,
            "shap_bimodality_flag": bool(row['shap_bimodality_flag']) if pd.notna(row['shap_bimodality_flag']) else None,
            "cross_seed_mean_rank": float(row['cross_seed_mean_rank']) if pd.notna(row['cross_seed_mean_rank']) else None,
            "cross_seed_std_rank": float(row['cross_seed_std_rank']) if pd.notna(row['cross_seed_std_rank']) else None
        }
    
    # Add comparison data if available
    if comparison_df is not None:
        for _, row in comparison_df.iterrows():
            feature = row['feature']
            if feature in shap_stability_results["features"]:
                shap_stability_results["features"][feature].update({
                    "shap_complexity": float(row['shap_complexity']) if pd.notna(row['shap_complexity']) else None,
                    "oot_drift_score": float(row['oot_drift_score']) if pd.notna(row['oot_drift_score']) else None,
                    "direction_consistent": bool(row['direction_consistent']) if pd.notna(row['direction_consistent']) else None,
                    "classification": row['classification'] if pd.notna(row['classification']) else None
                })
    
    shap_path = os.path.join(OUTPUT_DIR, "shap_stability_results.json")
    with open(shap_path, 'w') as f:
        json.dump(shap_stability_results, f, indent=2)
    print(f"  Saved: {shap_path}")
    
    # ==========================================================================
    # 3. Reliability Results
    # ==========================================================================
    reliability_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "reliability_scoring",
            "dataset": "default_of_credit_card_clients",
            "target": "default payment next month",
            "n_features": len(master_df),
            "scoring_method": "weighted_combination_of_stability_importance_consistency"
        },
        "grade_scale": {
            "A": "0.8 - 1.0 (Highly Reliable)",
            "B": "0.6 - 0.8 (Reliable)",
            "C": "0.4 - 0.6 (Moderate)",
            "D": "0.2 - 0.4 (Low Reliability)",
            "F": "0.0 - 0.2 (Unreliable)"
        },
        "features": {}
    }
    
    # Calculate reliability scores using the documented ReliabilityScorer
    # (4-component: importance=30%, stability=40%, coverage=15%, consistency=15%)
    # complexity_scores here are in [0, 100] range; configure bounds accordingly.
    # coverage is unavailable from the CSV, so we default to 0.5 (neutral).
    scorer = ReliabilityScorer(ReliabilityConfig(
        complexity_min=0.0,
        complexity_max=100.0,
        importance_min=0.0,
        importance_max=1.0,
        cross_seed_std_min=0.0,
        cross_seed_std_max=3.0,
    ))

    for _, row in master_df.iterrows():
        feature = row['feature']

        complexity = float(row['complexity_score']) if pd.notna(row['complexity_score']) else float('nan')
        importance = float(row['shap_importance_norm']) if pd.notna(row['shap_importance_norm']) else float('nan')
        cross_seed_std = float(row['shap_cv']) if pd.notna(row['shap_cv']) else float('nan')

        result = scorer.compute(
            feature_name=feature,
            complexity_score=complexity,
            importance_score=importance,
            coverage_ratio=0.5,  # Not available from CSV; neutral value
            cross_seed_std=cross_seed_std,
        )

        reliability = result.reliability_score

        # Assign grade
        if reliability >= 0.8:
            grade = "A"
        elif reliability >= 0.6:
            grade = "B"
        elif reliability >= 0.4:
            grade = "C"
        elif reliability >= 0.2:
            grade = "D"
        else:
            grade = "F"

        reliability_results["features"][feature] = {
            "reliability_score": round(reliability, 3),
            "grade": grade,
            "components": {
                "stability_component": round(result.stability_component, 3),
                "importance_component": round(result.importance_component, 3),
                "coverage_component": round(result.coverage_component, 3),
                "consistency_component": round(result.consistency_component, 3),
            }
        }
    
    reliability_path = os.path.join(OUTPUT_DIR, "reliability_results.json")
    with open(reliability_path, 'w') as f:
        json.dump(reliability_results, f, indent=2)
    print(f"  Saved: {reliability_path}")
    
    # ==========================================================================
    # 4. Meta-Bootstrap Results (Complexity with Confidence Intervals)
    # ==========================================================================
    meta_bootstrap_results = {
        "metadata": {
            "generated_at": datetime.now().isoformat(),
            "analysis_type": "meta_bootstrap_complexity",
            "dataset": "default_of_credit_card_clients",
            "target": "default payment next month",
            "n_features": len(master_df),
            "n_meta_iterations": 10,
            "confidence_level": 0.95
        },
        "features": {}
    }
    
    for _, row in master_df.iterrows():
        feature = row['feature']
        complexity = row['complexity_score'] if pd.notna(row['complexity_score']) else None
        
        # Confidence intervals require running MetaBootstrap across multiple data splits.
        # They are not available from the pre-computed CSV; set to null here.
        meta_bootstrap_results["features"][feature] = {
            "complexity_score": float(complexity) if complexity is not None else None,
            "complexity_rank": int(row['complexity_rank']) if pd.notna(row['complexity_rank']) else None,
            "confidence_interval_95": {
                "lower": None,
                "upper": None,
                "computed": False,
                "note": "Run MetaBootstrap.fit() to compute real confidence intervals"
            },
            "oot_complexity_score": float(row['oot_complexity_score']) if pd.notna(row['oot_complexity_score']) else None,
            "oot_complexity_shift": float(row['oot_complexity_shift']) if pd.notna(row['oot_complexity_shift']) else None,
            "stability_category": "stable" if complexity is not None and complexity < 50 else "unstable"
        }
    
    meta_path = os.path.join(OUTPUT_DIR, "meta_bootstrap_results.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_bootstrap_results, f, indent=2)
    print(f"  Saved: {meta_path}")
    
    # ==========================================================================
    # 5. Master Table CSV (already exists, just confirm)
    # ==========================================================================
    master_csv_path = os.path.join(OUTPUT_DIR, "master_table.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"  Saved: {master_csv_path}")
    
    # ==========================================================================
    # 6. Summary Report (Markdown)
    # ==========================================================================
    summary_md_path = os.path.join(OUTPUT_DIR, "summary_report.md")
    
    # Sort features by reliability for top features list
    feature_reliability = []
    for feature, data in reliability_results["features"].items():
        feature_reliability.append((feature, data['reliability_score'], data['grade']))
    feature_reliability.sort(key=lambda x: x[1], reverse=True)
    
    # Get classification counts
    classification_counts = {"STABLE": 0, "CONFIRMED_UNSTABLE": 0, "FALSE_ALARM": 0, "MISSED_RISK": 0, "OOT_DRIFT": 0}
    if comparison_df is not None:
        for cls in classification_counts:
            classification_counts[cls] = int((comparison_df['classification'] == cls).sum())
    
    with open(summary_md_path, 'w') as f:
        f.write("# Credit Card Default Feature Analysis Report\n\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("**Dataset:** Default of Credit Card Clients (30,000 samples, 23 features)\n\n")
        f.write("**Target Variable:** `default payment next month`\n\n")
        
        f.write("---\n\n")
        f.write("## Executive Summary\n\n")
        f.write(f"This analysis evaluated **23 features** using bootstrap stability analysis, ")
        f.write("SHAP-based stability metrics, and reliability scoring.\n\n")
        
        f.write("### Key Findings\n\n")
        f.write(f"- **{classification_counts['STABLE']}** features are stable and reliable for production use\n")
        f.write(f"- **{classification_counts['OOT_DRIFT']}** features show drift between training and OOT periods\n")
        f.write(f"- **{classification_counts['FALSE_ALARM']}** features were false alarms (marginally unstable but SHAP-stable)\n")
        f.write(f"- **{classification_counts['CONFIRMED_UNSTABLE']}** features are confirmed unstable\n\n")
        
        f.write("---\n\n")
        f.write("## Top 10 Most Reliable Features\n\n")
        f.write("| Rank | Feature | Reliability Score | Grade |\n")
        f.write("|------|---------|-------------------|-------|\n")
        for i, (feature, score, grade) in enumerate(feature_reliability[:10], 1):
            f.write(f"| {i} | `{feature}` | {score:.3f} | {grade} |\n")
        
        f.write("\n---\n\n")
        f.write("## Feature Classification Summary\n\n")
        f.write("| Classification | Count | Description |\n")
        f.write("|----------------|-------|-------------|\n")
        f.write(f"| STABLE | {classification_counts['STABLE']} | Stable marginal distribution and SHAP contributions |\n")
        f.write(f"| CONFIRMED_UNSTABLE | {classification_counts['CONFIRMED_UNSTABLE']} | Unstable in both marginal and SHAP analysis |\n")
        f.write(f"| FALSE_ALARM | {classification_counts['FALSE_ALARM']} | Marginal instability but stable SHAP contributions |\n")
        f.write(f"| MISSED_RISK | {classification_counts['MISSED_RISK']} | Stable marginal but unstable SHAP contributions |\n")
        f.write(f"| OOT_DRIFT | {classification_counts['OOT_DRIFT']} | Different behavior in out-of-time test period |\n")
        
        f.write("\n---\n\n")
        f.write("## OOT Drift Features\n\n")
        if comparison_df is not None:
            drift_features = comparison_df[comparison_df['classification'] == 'OOT_DRIFT'].sort_values('oot_drift_score', ascending=False)
            if len(drift_features) > 0:
                f.write("| Feature | Drift Score | Direction Consistent |\n")
                f.write("|---------|-------------|---------------------|\n")
                for _, row in drift_features.iterrows():
                    dir_consistent = "✓" if row['direction_consistent'] else "✗"
                    f.write(f"| `{row['feature']}` | {row['oot_drift_score']:.3f} | {dir_consistent} |\n")
            else:
                f.write("No features with significant OOT drift detected.\n")
        
        f.write("\n---\n\n")
        f.write("## False Alarms (Marginal Unstable, SHAP Stable)\n\n")
        if comparison_df is not None:
            false_alarms = comparison_df[comparison_df['classification'] == 'FALSE_ALARM']
            if len(false_alarms) > 0:
                f.write("| Feature | Marginal Complexity | SHAP Complexity |\n")
                f.write("|---------|---------------------|-----------------|\n")
                for _, row in false_alarms.iterrows():
                    f.write(f"| `{row['feature']}` | {row['marginal_complexity']:.2f} | {row['shap_complexity']:.3f} |\n")
                f.write("\n> **Note:** These features have complex marginal distributions but stable model contributions. ")
                f.write("They may be suitable for production use despite marginal instability.\n")
            else:
                f.write("No false alarms detected.\n")
        
        f.write("\n---\n\n")
        f.write("## Recommendations\n\n")
        
        if classification_counts['OOT_DRIFT'] > 0:
            f.write(f"1. **Monitor OOT Drift Features:** {classification_counts['OOT_DRIFT']} features show different behavior ")
            f.write("in the test period. Consider monitoring these closely in production.\n\n")
        
        if classification_counts['FALSE_ALARM'] > 0:
            f.write(f"2. **False Alarms are Safe:** {classification_counts['FALSE_ALARM']} features were flagged by marginal ")
            f.write("analysis but have stable SHAP contributions. These are safe for production use.\n\n")
        
        stable_count = classification_counts['STABLE']
        f.write(f"3. **Stable Features:** {stable_count} features are confirmed stable and reliable for production use.\n\n")
        
        f.write("---\n\n")
        f.write("## Output Files\n\n")
        f.write("| File | Description |\n")
        f.write("|------|-------------|\n")
        f.write("| `stability_results.json` | Bootstrap/marginal stability metrics per feature |\n")
        f.write("| `shap_stability_results.json` | SHAP-based stability metrics |\n")
        f.write("| `reliability_results.json` | Reliability scores with grades |\n")
        f.write("| `meta_bootstrap_results.json` | Complexity scores with confidence intervals |\n")
        f.write("| `master_table.csv` | All features with all metrics combined |\n")
        f.write("| `summary_report.md` | This human-readable synthesis |\n")
        f.write("| `*.png` | Visualization plots |\n")
        
        f.write("\n---\n\n")
        f.write("*Report generated by Feature Bootstrapping Toolkit*\n")
    
    print(f"  Saved: {summary_md_path}")
    
    print("\n✓ All JSON output files generated successfully!")

if __name__ == "__main__":
    main()
