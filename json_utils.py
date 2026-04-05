#!/usr/bin/env python3
"""
JSON Utilities for Feature Bootstrapping Toolkit.

This module provides utilities for generating JSON output files from analysis results.
It combines simple CSV-to-JSON conversion (standalone, no dependencies) with full-featured
JSON generation using the bootstrap_stability module.

The module is organized into two main sections:
1. Simple Export Functions - Standalone functions that don't require bootstrap_stability
2. Full Export Functions - Functions that use bootstrap_stability.reliability module

Usage:
    # Simple export (no bootstrap_stability dependency):
    from json_utils import simple_export_main, generate_stability_results
    
    # Full export (with bootstrap_stability):
    from json_utils import full_export_main, generate_reliability_results_full
"""

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

# Optional import for full export functionality
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False

try:
    from bootstrap_stability.reliability import ReliabilityScorer, ReliabilityConfig
    RELIABILITY_SCORER_AVAILABLE = True
except ImportError:
    RELIABILITY_SCORER_AVAILABLE = False


# =============================================================================
# SHARED UTILITIES
# =============================================================================

def get_default_output_dir() -> Path:
    """Return the default output directory path."""
    return Path("credit_card_analysis_results")


# =============================================================================
# SECTION 1: SIMPLE EXPORT FUNCTIONS (Originally from simple_json_export.py)
# =============================================================================
# These functions are standalone and do not require pandas or bootstrap_stability.
# They provide basic CSV-to-JSON conversion functionality.
# =============================================================================

def load_csv_simple(filename: str, output_dir: Optional[Path] = None) -> List[Dict[str, str]]:
    """
    Load CSV file and return list of dictionaries.
    
    Simple standalone function that doesn't require pandas.
    
    Args:
        filename: Name of the CSV file to load
        output_dir: Directory containing the file (defaults to credit_card_analysis_results)
    
    Returns:
        List of dictionaries with string values, or empty list if file not found
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    filepath = output_dir / filename
    if not filepath.exists():
        print(f"Warning: {filepath} not found")
        return []
    
    with open(filepath, 'r') as f:
        reader = csv.DictReader(f)
        return list(reader)


def convert_numeric(value: str) -> Optional[Any]:
    """
    Convert string value to numeric if possible.
    
    Args:
        value: String value to convert
    
    Returns:
        int, float, None, or the original string if conversion fails
    """
    if value == '' or value is None:
        return None
    try:
        if '.' in value:
            return float(value)
        return int(value)
    except (ValueError, TypeError):
        return value


def generate_stability_results(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Generate stability_results.json from master_comparison.csv.
    
    Simple standalone version that doesn't require pandas.
    
    Args:
        output_dir: Directory containing input CSV and where to save output
    
    Returns:
        Dictionary mapping feature names to stability metrics
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    data = load_csv_simple("master_comparison.csv", output_dir)
    
    results = {}
    for row in data:
        feature = row.get('feature', '')
        if not feature:
            continue
        
        results[feature] = {
            "complexity_score": convert_numeric(row.get('complexity_score')),
            "complexity_rank": convert_numeric(row.get('complexity_rank')),
            "holdout_complexity_score": convert_numeric(row.get('holdout_complexity_score')),
            "holdout_complexity_shift": convert_numeric(row.get('holdout_complexity_shift')),
            "lgbm_gain_norm": convert_numeric(row.get('lgbm_gain_norm')),
            "lgbm_gain_rank": convert_numeric(row.get('lgbm_gain_rank')),
            "tree_coverage": convert_numeric(row.get('tree_coverage')),
            "mean_depth": convert_numeric(row.get('mean_depth')),
        }
    
    return results


def generate_shap_stability_results(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Generate shap_stability_results.json from master_comparison.csv.
    
    Simple standalone version that doesn't require pandas.
    
    Args:
        output_dir: Directory containing input CSV and where to save output
    
    Returns:
        Dictionary mapping feature names to SHAP stability metrics
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    data = load_csv_simple("master_comparison.csv", output_dir)
    
    results = {}
    for row in data:
        feature = row.get('feature', '')
        if not feature:
            continue
        
        results[feature] = {
            "shap_importance_norm": convert_numeric(row.get('shap_importance_norm')),
            "shap_rank": convert_numeric(row.get('shap_rank')),
            "shap_cv": convert_numeric(row.get('shap_cv')),
            "shap_total_interaction": convert_numeric(row.get('shap_total_interaction')),
            "shap_skewness": convert_numeric(row.get('shap_skewness')),
            "shap_kurtosis": convert_numeric(row.get('shap_kurtosis')),
            "shap_bimodality_flag": row.get('shap_bimodality_flag') == 'True',
            "cross_seed_mean_rank": convert_numeric(row.get('cross_seed_mean_rank')),
            "cross_seed_std_rank": convert_numeric(row.get('cross_seed_std_rank')),
        }
    
    return results


def generate_reliability_results(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Generate reliability_results.json combining stability and importance.
    
    Simple standalone version with basic reliability scoring formula.
    Uses a simple formula: importance * (1 - normalized_complexity)
    
    Args:
        output_dir: Directory containing input CSV and where to save output
    
    Returns:
        Dictionary mapping feature names to reliability metrics and grades
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    data = load_csv_simple("master_comparison.csv", output_dir)
    
    results = {}
    for row in data:
        feature = row.get('feature', '')
        if not feature:
            continue
        
        complexity = convert_numeric(row.get('complexity_score'))
        importance = convert_numeric(row.get('shap_importance_norm'))
        
        # Calculate reliability score (higher is better)
        # Normalize complexity (lower is better) and combine with importance
        if complexity is not None and importance is not None:
            # Simple reliability formula: importance * (1 - normalized_complexity)
            # Assuming complexity ranges from -300 to 200 based on data
            normalized_complexity = max(0, min(1, (complexity + 300) / 500))
            reliability_score = importance * (1 - normalized_complexity)
            
            # Assign grade
            if reliability_score >= 0.1:
                grade = "A"
            elif reliability_score >= 0.05:
                grade = "B"
            elif reliability_score >= 0.02:
                grade = "C"
            elif reliability_score >= 0.01:
                grade = "D"
            else:
                grade = "F"
        else:
            reliability_score = None
            grade = "N/A"
        
        results[feature] = {
            "reliability_score": reliability_score,
            "grade": grade,
            "components": {
                "stability": {
                    "complexity_score": complexity,
                    "complexity_rank": convert_numeric(row.get('complexity_rank')),
                },
                "importance": {
                    "shap_importance_norm": importance,
                    "shap_rank": convert_numeric(row.get('shap_rank')),
                    "lgbm_gain_norm": convert_numeric(row.get('lgbm_gain_norm')),
                },
                "consistency": {
                    "cross_seed_std_rank": convert_numeric(row.get('cross_seed_std_rank')),
                    "shap_cv": convert_numeric(row.get('shap_cv')),
                }
            }
        }
    
    return results


def generate_meta_bootstrap_results(output_dir: Optional[Path] = None) -> Dict[str, Dict]:
    """
    Generate meta_bootstrap_results.json with confidence intervals.
    
    Simple standalone version that estimates confidence intervals from cross-seed variability.
    
    Args:
        output_dir: Directory containing input CSV and where to save output
    
    Returns:
        Dictionary mapping feature names to meta-bootstrap metrics
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    data = load_csv_simple("master_comparison.csv", output_dir)
    
    results = {}
    for row in data:
        feature = row.get('feature', '')
        if not feature:
            continue
        
        complexity = convert_numeric(row.get('complexity_score'))
        holdout_complexity = convert_numeric(row.get('holdout_complexity_score'))
        holdout_shift = convert_numeric(row.get('holdout_complexity_shift'))
        
        # Use cross_seed_std as a proxy for confidence interval width
        cross_seed_std = convert_numeric(row.get('cross_seed_std_rank'))
        
        # Estimate CI based on cross-seed variability
        if complexity is not None:
            ci_width = (cross_seed_std * 10) if cross_seed_std else 0
            ci_lower = complexity - ci_width
            ci_upper = complexity + ci_width
        else:
            ci_lower = ci_upper = None
        
        results[feature] = {
            "mean_complexity": complexity,
            "std_complexity": cross_seed_std,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "holdout_complexity": holdout_complexity,
            "holdout_shift": holdout_shift,
            "shift_flag": abs(holdout_shift) > 50 if holdout_shift else False,
        }
    
    return results


def generate_summary_report(output_dir: Optional[Path] = None) -> str:
    """
    Generate a summary report in markdown format.
    
    Simple standalone version that creates a basic summary report.
    
    Args:
        output_dir: Directory containing input CSVs and where to save output
    
    Returns:
        Markdown-formatted summary report string
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    data = load_csv_simple("master_comparison.csv", output_dir)
    
    # Sort by complexity score
    sorted_by_complexity = sorted(
        [d for d in data if d.get('complexity_score') and d['complexity_score']],
        key=lambda x: float(x.get('complexity_score', 0))
    )
    
    # Sort by importance
    sorted_by_importance = sorted(
        [d for d in data if d.get('shap_importance_norm')],
        key=lambda x: float(x.get('shap_importance_norm', 0)),
        reverse=True
    )
    
    report = """# Credit Card Default Analysis Summary

## Executive Summary

Analysis performed on the Default of Credit Card Clients dataset (30,000 samples, 23 features).

### Top 5 Most Stable Features (Lowest Complexity)
"""
    
    for i, row in enumerate(sorted_by_complexity[:5], 1):
        complexity = float(row['complexity_score']) if row['complexity_score'] else 0
        report += f"{i}. **{row['feature']}**: complexity = {complexity:.2f}\n"
    
    report += "\n### Top 5 Most Important Features (SHAP)\n"
    
    for i, row in enumerate(sorted_by_importance[:5], 1):
        importance = float(row['shap_importance_norm']) if row['shap_importance_norm'] else 0
        report += f"{i}. **{row['feature']}**: importance = {importance:.4f}\n"
    
    # Load marginal vs SHAP comparison
    comparison = load_csv_simple("marginal_vs_shap_comparison.csv", output_dir)
    
    stable = [c for c in comparison if c.get('classification') == 'STABLE']
    false_alarms = [c for c in comparison if c.get('classification') == 'FALSE_ALARM']
    holdout_drift = [c for c in comparison if c.get('classification') == 'HOLDOUT_DRIFT']
    
    report += f"""
## Classification Summary

| Classification | Count |
|---------------|-------|
| STABLE | {len(stable)} |
| FALSE_ALARM | {len(false_alarms)} |
| HOLDOUT_DRIFT | {len(holdout_drift)} |

"""
    
    if false_alarms:
        report += "### False Alarms (Marginal unstable, SHAP stable)\n"
        for fa in false_alarms:
            report += f"- {fa['feature']}: marginal={fa.get('marginal_complexity', 'N/A')}, shap={fa.get('shap_complexity', 'N/A')}\n"
    
    if holdout_drift:
        report += "\n### Holdout Drift Features\n"
        for od in holdout_drift:
            report += f"- {od['feature']}: drift_score={od.get('drift_score', 'N/A')}\n"
    
    report += """
## Recommendations

### Tier 1 - Production Ready
Features with low complexity and high importance:
"""
    
    # Features with low complexity and high importance
    tier1 = []
    for row in data:
        complexity = convert_numeric(row.get('complexity_score'))
        importance = convert_numeric(row.get('shap_importance_norm'))
        if complexity is not None and importance is not None:
            if complexity < 10 and importance > 0.03:
                tier1.append(row['feature'])
    
    for f in tier1[:5]:
        report += f"- {f}\n"
    
    report += """
### Tier 2 - Monitor
Features with moderate complexity:
"""
    
    tier2 = []
    for row in data:
        complexity = convert_numeric(row.get('complexity_score'))
        if complexity is not None and 10 <= complexity < 100:
            tier2.append(row['feature'])
    
    for f in tier2[:5]:
        report += f"- {f}\n"
    
    report += """
### Tier 3 - Investigate
Features with OOT drift or high complexity:
"""
    
    for od in holdout_drift[:5]:
        report += f"- {od['feature']} (OOT drift)\n"
    
    return report


def simple_export_main(output_dir: Optional[Path] = None) -> None:
    """
    Main function for simple JSON export.
    
    Generates all JSON output files from CSV data without requiring
    pandas or bootstrap_stability module.
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    """
    if output_dir is None:
        output_dir = get_default_output_dir()
    
    print("Generating JSON output files (simple mode)...")
    
    # Generate stability results
    stability = generate_stability_results(output_dir)
    with open(output_dir / "stability_results.json", 'w') as f:
        json.dump(stability, f, indent=2)
    print(f"Generated stability_results.json ({len(stability)} features)")
    
    # Generate SHAP stability results
    shap_stability = generate_shap_stability_results(output_dir)
    with open(output_dir / "shap_stability_results.json", 'w') as f:
        json.dump(shap_stability, f, indent=2)
    print(f"Generated shap_stability_results.json ({len(shap_stability)} features)")
    
    # Generate reliability results
    reliability = generate_reliability_results(output_dir)
    with open(output_dir / "reliability_results.json", 'w') as f:
        json.dump(reliability, f, indent=2)
    print(f"Generated reliability_results.json ({len(reliability)} features)")
    
    # Generate meta-bootstrap results
    meta_bootstrap = generate_meta_bootstrap_results(output_dir)
    with open(output_dir / "meta_bootstrap_results.json", 'w') as f:
        json.dump(meta_bootstrap, f, indent=2)
    print(f"Generated meta_bootstrap_results.json ({len(meta_bootstrap)} features)")
    
    # Generate summary report
    summary = generate_summary_report(output_dir)
    with open(output_dir / "summary_report.md", 'w') as f:
        f.write(summary)
    print("Generated summary_report.md")
    
    print("\nAll JSON files generated successfully!")


# =============================================================================
# SECTION 2: FULL EXPORT FUNCTIONS (Originally from generate_json_outputs.py)
# =============================================================================
# These functions require pandas and bootstrap_stability.reliability module.
# They provide enhanced JSON generation with proper reliability scoring.
# =============================================================================

def load_csv_dataframe(filename: str, output_dir: Optional[str] = None) -> Optional[Any]:
    """
    Load a CSV file from the output directory as a pandas DataFrame.
    
    Args:
        filename: Name of the CSV file to load
        output_dir: Directory containing the file
    
    Returns:
        pandas DataFrame or None if file not found or pandas not available
    """
    if not PANDAS_AVAILABLE:
        print("Warning: pandas not available, cannot load DataFrame")
        return None
    
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    path = os.path.join(output_dir, filename)
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def generate_stability_results_full(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate stability_results.json with full metadata.
    
    Enhanced version that includes metadata and uses pandas for data processing.
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    
    Returns:
        Dictionary with metadata and feature stability metrics
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    if master_df is None:
        return {}
    
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
            "holdout_complexity_score": float(row['holdout_complexity_score']) if pd.notna(row['holdout_complexity_score']) else None,
            "holdout_complexity_shift": float(row['holdout_complexity_shift']) if pd.notna(row['holdout_complexity_shift']) else None,
            "tree_coverage": float(row['tree_coverage']) if pd.notna(row['tree_coverage']) else None,
            "mean_depth": float(row['mean_depth']) if pd.notna(row['mean_depth']) else None,
            "stability_interpretation": "stable" if pd.notna(row['complexity_score']) and row['complexity_score'] < 50 else "unstable"
        }
    
    return stability_results


def generate_shap_stability_results_full(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate shap_stability_results.json with full metadata and comparison data.
    
    Enhanced version that includes comparison data from marginal_vs_shap_comparison.csv.
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    
    Returns:
        Dictionary with metadata and feature SHAP stability metrics
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    comparison_df = load_csv_dataframe("marginal_vs_shap_comparison.csv", output_dir)
    
    if master_df is None:
        return {}
    
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
                    "holdout_drift_score": float(row['holdout_drift_score']) if pd.notna(row['holdout_drift_score']) else None,
                    "direction_consistent": bool(row['direction_consistent']) if pd.notna(row['direction_consistent']) else None,
                    "classification": row['classification'] if pd.notna(row['classification']) else None
                })
    
    return shap_stability_results


def generate_reliability_results_full(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate reliability_results.json using the ReliabilityScorer module.
    
    Enhanced version that uses the 4-component reliability scoring:
    importance=30%, stability=40%, coverage=15%, consistency=15%
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    
    Returns:
        Dictionary with metadata, grade scale, and feature reliability metrics
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    if master_df is None:
        return {}
    
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
    # cross_seed_std_rank: std dev of feature rank across seeds (rank-position units, range ≈ 0–8)
    if RELIABILITY_SCORER_AVAILABLE:
        scorer = ReliabilityScorer(ReliabilityConfig(
            complexity_min=0.0,
            complexity_max=100.0,
            importance_min=0.0,
            importance_max=1.0,
            cross_seed_std_min=0.0,
            cross_seed_std_max=5.0,
        ))
        use_scorer = True
    else:
        use_scorer = False

    for _, row in master_df.iterrows():
        feature = row['feature']

        complexity = float(row['complexity_score']) if pd.notna(row['complexity_score']) else float('nan')
        importance = float(row['shap_importance_norm']) if pd.notna(row['shap_importance_norm']) else float('nan')
        # cross_seed_std_rank: std dev of rank across seeds (rank units, not shap_cv)
        cross_seed_std = float(row['cross_seed_std_rank']) if pd.notna(row['cross_seed_std_rank']) else float('nan')
        # tree_coverage: fraction of trees using the feature (already in [0, 1])
        coverage = float(row['tree_coverage']) if pd.notna(row['tree_coverage']) else float('nan')

        if use_scorer:
            result = scorer.compute(
                feature_name=feature,
                complexity_score=complexity,
                importance_score=importance,
                coverage_ratio=coverage,
                cross_seed_std=cross_seed_std,
            )
            reliability = result.reliability_score
            
            reliability_results["features"][feature] = {
                "reliability_score": round(reliability, 3),
                "grade": _get_grade(reliability),
                "components": {
                    "stability_component": round(result.stability_component, 3),
                    "importance_component": round(result.importance_component, 3),
                    "coverage_component": round(result.coverage_component, 3),
                    "consistency_component": round(result.consistency_component, 3),
                }
            }
        else:
            # Fallback to simple calculation
            reliability = _simple_reliability_score(complexity, importance)
            reliability_results["features"][feature] = {
                "reliability_score": round(reliability, 3) if reliability is not None else None,
                "grade": _get_grade(reliability) if reliability is not None else "N/A",
                "components": {
                    "complexity_score": complexity if not pd.isna(complexity) else None,
                    "importance_score": importance if not pd.isna(importance) else None,
                }
            }
    
    return reliability_results


def _get_grade(reliability: float) -> str:
    """Convert reliability score to letter grade."""
    if reliability >= 0.8:
        return "A"
    elif reliability >= 0.6:
        return "B"
    elif reliability >= 0.4:
        return "C"
    elif reliability >= 0.2:
        return "D"
    else:
        return "F"


def _simple_reliability_score(complexity: float, importance: float) -> Optional[float]:
    """Calculate simple reliability score as fallback."""
    if complexity is None or importance is None or pd.isna(complexity) or pd.isna(importance):
        return None
    normalized_complexity = max(0, min(1, (complexity + 300) / 500))
    return importance * (1 - normalized_complexity)


def generate_meta_bootstrap_results_full(output_dir: Optional[str] = None) -> Dict[str, Any]:
    """
    Generate meta_bootstrap_results.json with full metadata.
    
    Enhanced version that includes proper structure for confidence intervals
    (though actual CIs require running MetaBootstrap.fit()).
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    
    Returns:
        Dictionary with metadata and feature meta-bootstrap metrics
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    if master_df is None:
        return {}
    
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
        
        meta_bootstrap_results["features"][feature] = {
            "complexity_score": float(complexity) if complexity is not None else None,
            "complexity_rank": int(row['complexity_rank']) if pd.notna(row['complexity_rank']) else None,
            "confidence_interval_95": {
                "lower": None,
                "upper": None,
                "computed": False,
                "note": "Run MetaBootstrap.fit() to compute real confidence intervals"
            },
            "holdout_complexity_score": float(row['holdout_complexity_score']) if pd.notna(row['holdout_complexity_score']) else None,
            "holdout_complexity_shift": float(row['holdout_complexity_shift']) if pd.notna(row['holdout_complexity_shift']) else None,
            "stability_category": "stable" if complexity is not None and complexity < 50 else "unstable"
        }
    
    return meta_bootstrap_results


def generate_summary_report_full(output_dir: Optional[str] = None) -> str:
    """
    Generate a comprehensive summary report in markdown format.
    
    Enhanced version with more detailed analysis and better formatting.
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    
    Returns:
        Markdown-formatted summary report string
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    comparison_df = load_csv_dataframe("marginal_vs_shap_comparison.csv", output_dir)
    
    if master_df is None:
        return "# Error: No data available\n\nCould not load master_comparison.csv"
    
    # Get reliability results for ranking
    reliability_results = generate_reliability_results_full(output_dir)
    
    # Sort features by reliability for top features list
    feature_reliability = []
    for feature, data in reliability_results.get("features", {}).items():
        score = data.get('reliability_score')
        if score is not None:
            feature_reliability.append((feature, score, data.get('grade', 'N/A')))
    feature_reliability.sort(key=lambda x: x[1], reverse=True)
    
    # Get classification counts
    classification_counts = {"STABLE": 0, "CONFIRMED_UNSTABLE": 0, "FALSE_ALARM": 0, "MISSED_RISK": 0, "HOLDOUT_DRIFT": 0}
    if comparison_df is not None:
        for cls in classification_counts:
            classification_counts[cls] = int((comparison_df['classification'] == cls).sum())
    
    report_lines = [
        "# Credit Card Default Feature Analysis Report",
        "",
        f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "**Dataset:** Default of Credit Card Clients (30,000 samples, 23 features)",
        "",
        "**Target Variable:** `default payment next month`",
        "",
        "---",
        "",
        "## Executive Summary",
        "",
        "This analysis evaluated **23 features** using bootstrap stability analysis, ",
        "SHAP-based stability metrics, and reliability scoring.",
        "",
        "### Key Findings",
        "",
        f"- **{classification_counts['STABLE']}** features are stable and reliable for production use",
        f"- **{classification_counts['HOLDOUT_DRIFT']}** features show drift between training and holdout sets",
        f"- **{classification_counts['FALSE_ALARM']}** features were false alarms (marginally unstable but SHAP-stable)",
        f"- **{classification_counts['CONFIRMED_UNSTABLE']}** features are confirmed unstable",
        "",
        "---",
        "",
        "## Top 10 Most Reliable Features",
        "",
        "| Rank | Feature | Reliability Score | Grade |",
        "|------|---------|-------------------|-------|",
    ]
    
    for i, (feature, score, grade) in enumerate(feature_reliability[:10], 1):
        report_lines.append(f"| {i} | `{feature}` | {score:.3f} | {grade} |")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Feature Classification Summary",
        "",
        "| Classification | Count | Description |",
        "|----------------|-------|-------------|",
        f"| STABLE | {classification_counts['STABLE']} | Stable marginal distribution and SHAP contributions |",
        f"| CONFIRMED_UNSTABLE | {classification_counts['CONFIRMED_UNSTABLE']} | Unstable in both marginal and SHAP analysis |",
        f"| FALSE_ALARM | {classification_counts['FALSE_ALARM']} | Marginal instability but stable SHAP contributions |",
        f"| MISSED_RISK | {classification_counts['MISSED_RISK']} | Stable marginal but unstable SHAP contributions |",
        f"| HOLDOUT_DRIFT | {classification_counts['HOLDOUT_DRIFT']} | Different behavior in holdout set |",
        "",
        "---",
        "",
        "## Holdout Drift Features",
        "",
    ])
    
    if comparison_df is not None:
        drift_features = comparison_df[comparison_df['classification'] == 'HOLDOUT_DRIFT'].sort_values('holdout_drift_score', ascending=False)
        if len(drift_features) > 0:
            report_lines.extend([
                "| Feature | Drift Score | Direction Consistent |",
                "|---------|-------------|---------------------|",
            ])
            for _, row in drift_features.iterrows():
                dir_consistent = "✓" if row['direction_consistent'] else "✗"
                report_lines.append(f"| `{row['feature']}` | {row['holdout_drift_score']:.3f} | {dir_consistent} |")
        else:
            report_lines.append("No features with significant OOT drift detected.")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## False Alarms (Marginal Unstable, SHAP Stable)",
        "",
    ])
    
    if comparison_df is not None:
        false_alarms = comparison_df[comparison_df['classification'] == 'FALSE_ALARM']
        if len(false_alarms) > 0:
            report_lines.extend([
                "| Feature | Marginal Complexity | SHAP Complexity |",
                "|---------|---------------------|-----------------|",
            ])
            for _, row in false_alarms.iterrows():
                report_lines.append(f"| `{row['feature']}` | {row['marginal_complexity']:.2f} | {row['shap_complexity']:.3f} |")
            report_lines.extend([
                "",
                "> **Note:** These features have complex marginal distributions but stable model contributions. ",
                "They may be suitable for production use despite marginal instability.",
            ])
        else:
            report_lines.append("No false alarms detected.")
    
    report_lines.extend([
        "",
        "---",
        "",
        "## Recommendations",
        "",
    ])
    
    if classification_counts['HOLDOUT_DRIFT'] > 0:
        report_lines.extend([
            f"1. **Monitor Holdout Drift Features:** {classification_counts['HOLDOUT_DRIFT']} features show different behavior ",
            "in the test period. Consider monitoring these closely in production.",
            "",
        ])
    
    if classification_counts['FALSE_ALARM'] > 0:
        report_lines.extend([
            f"2. **False Alarms are Safe:** {classification_counts['FALSE_ALARM']} features were flagged by marginal ",
            "analysis but have stable SHAP contributions. These are safe for production use.",
            "",
        ])
    
    stable_count = classification_counts['STABLE']
    report_lines.extend([
        f"3. **Stable Features:** {stable_count} features are confirmed stable and reliable for production use.",
        "",
        "---",
        "",
        "## Output Files",
        "",
        "| File | Description |",
        "|------|-------------|",
        "| `stability_results.json` | Bootstrap/marginal stability metrics per feature |",
        "| `shap_stability_results.json` | SHAP-based stability metrics |",
        "| `reliability_results.json` | Reliability scores with grades |",
        "| `meta_bootstrap_results.json` | Complexity scores with confidence intervals |",
        "| `master_table.csv` | All features with all metrics combined |",
        "| `summary_report.md` | This human-readable synthesis |",
        "| `*.png` | Visualization plots |",
        "",
        "---",
        "",
        "*Report generated by Feature Bootstrapping Toolkit*",
    ])
    
    return "\n".join(report_lines)


def full_export_main(output_dir: Optional[str] = None) -> None:
    """
    Main function for full JSON export with pandas and bootstrap_stability.
    
    Generates all JSON output files with enhanced metadata and proper
    reliability scoring using the bootstrap_stability module.
    
    Args:
        output_dir: Directory containing input CSVs and where to save outputs
    """
    if output_dir is None:
        output_dir = "credit_card_analysis_results"
    
    print("Generating JSON output files from analysis results...")
    
    # Load master data to verify it exists
    master_df = load_csv_dataframe("master_comparison.csv", output_dir)
    
    if master_df is None:
        print("ERROR: master_comparison.csv not found. Run comprehensive_feature_analysis.py first.")
        return
    
    # ==========================================================================
    # 1. Stability Results
    # ==========================================================================
    stability_results = generate_stability_results_full(output_dir)
    stability_path = os.path.join(output_dir, "stability_results.json")
    with open(stability_path, 'w') as f:
        json.dump(stability_results, f, indent=2)
    print(f"  Saved: {stability_path}")
    
    # ==========================================================================
    # 2. SHAP Stability Results
    # ==========================================================================
    shap_stability_results = generate_shap_stability_results_full(output_dir)
    shap_path = os.path.join(output_dir, "shap_stability_results.json")
    with open(shap_path, 'w') as f:
        json.dump(shap_stability_results, f, indent=2)
    print(f"  Saved: {shap_path}")
    
    # ==========================================================================
    # 3. Reliability Results
    # ==========================================================================
    reliability_results = generate_reliability_results_full(output_dir)
    reliability_path = os.path.join(output_dir, "reliability_results.json")
    with open(reliability_path, 'w') as f:
        json.dump(reliability_results, f, indent=2)
    print(f"  Saved: {reliability_path}")
    
    # ==========================================================================
    # 4. Meta-Bootstrap Results
    # ==========================================================================
    meta_bootstrap_results = generate_meta_bootstrap_results_full(output_dir)
    meta_path = os.path.join(output_dir, "meta_bootstrap_results.json")
    with open(meta_path, 'w') as f:
        json.dump(meta_bootstrap_results, f, indent=2)
    print(f"  Saved: {meta_path}")
    
    # ==========================================================================
    # 5. Master Table CSV
    # ==========================================================================
    master_csv_path = os.path.join(output_dir, "master_table.csv")
    master_df.to_csv(master_csv_path, index=False)
    print(f"  Saved: {master_csv_path}")
    
    # ==========================================================================
    # 6. Summary Report
    # ==========================================================================
    summary = generate_summary_report_full(output_dir)
    summary_md_path = os.path.join(output_dir, "summary_report.md")
    with open(summary_md_path, 'w') as f:
        f.write(summary)
    print(f"  Saved: {summary_md_path}")
    
    print("\n✓ All JSON output files generated successfully!")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main():
    """
    Main entry point for JSON export.
    
    Automatically selects full export mode if pandas and bootstrap_stability
    are available, otherwise falls back to simple export mode.
    """
    if PANDAS_AVAILABLE:
        full_export_main()
    else:
        print("Note: pandas not available, using simple export mode")
        simple_export_main()


if __name__ == "__main__":
    main()
