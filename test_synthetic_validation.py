#!/usr/bin/env python3
"""
Test Script for Synthetic Validation Suite

This script runs the full SyntheticValidation suite to test the detection
capabilities of the bootstrap stability analysis.

Results are saved to test_outputs/synthetic_validation/

Tests all 4 instability types:
- Heteroscedastic noise
- Distribution shift
- Interaction-dependent instability
- Missing not at random (MNAR)
"""

import os
import sys
import json
import warnings
from datetime import datetime
from pathlib import Path

# Set matplotlib backend before any imports
import matplotlib
matplotlib.use('Agg')

import numpy as np
import pandas as pd

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Output directory
OUTPUT_DIR = Path("test_outputs/synthetic_validation")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def print_progress(msg: str):
    """Print progress message with timestamp."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def save_results(results: dict, filename: str):
    """Save results to JSON file."""
    filepath = OUTPUT_DIR / filename
    
    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif pd.isna(obj):
            return None
        return obj
    
    serializable = convert_to_serializable(results)
    
    with open(filepath, 'w') as f:
        json.dump(serializable, f, indent=2, default=str)
    
    print_progress(f"Saved: {filepath}")


def save_dataframe(df: pd.DataFrame, filename: str):
    """Save DataFrame to CSV file."""
    filepath = OUTPUT_DIR / filename
    df.to_csv(filepath, index=False)
    print_progress(f"Saved: {filepath}")


def run_synthetic_validation_suite():
    """
    Run the full synthetic validation suite.
    
    Returns
    -------
    dict
        Complete validation results.
    """
    from bootstrap_stability import (
        SyntheticValidation,
        InstabilityType,
        BootstrapStability,
    )
    
    print_progress("=" * 60)
    print_progress("SYNTHETIC VALIDATION SUITE")
    print_progress("=" * 60)
    print_progress("")
    
    # Create validator
    validator = SyntheticValidation(random_state=42)
    
    # Configuration
    n_samples = 1000
    n_features = 10
    n_corrupted = 3
    # Use a threshold that flags features with higher complexity than average
    # Since complexity scores can be negative, we use a small negative threshold
    threshold = -0.01  # Flag features with complexity > -0.01
    
    print_progress(f"Configuration:")
    print_progress(f"  Samples per test: {n_samples}")
    print_progress(f"  Features per test: {n_features}")
    print_progress(f"  Corrupted features per test: {n_corrupted}")
    print_progress(f"  Detection threshold: {threshold}")
    print_progress("")
    
    # Store all results
    all_results = {
        "analysis_timestamp": datetime.now().isoformat(),
        "configuration": {
            "n_samples": n_samples,
            "n_features": n_features,
            "n_corrupted": n_corrupted,
            "threshold": threshold,
        },
        "test_results": {},
        "summary": {},
    }
    
    # Test each instability type
    instability_types = [
        (InstabilityType.HETEROSCEDASTIC, "Heteroscedastic Noise"),
        (InstabilityType.DISTRIBUTION_SHIFT, "Distribution Shift"),
        (InstabilityType.INTERACTION, "Interaction-Dependent"),
        (InstabilityType.MISSING_NOT_AT_RANDOM, "Missing Not At Random"),
    ]
    
    detection_metrics = []
    
    for instability_type, type_name in instability_types:
        print_progress("-" * 60)
        print_progress(f"Testing: {type_name}")
        print_progress("-" * 60)
        
        try:
            # Generate test data with more extreme perturbations
            print_progress(f"  Generating test data...")
            X, y, metadata = validator.generate_test_data(
                n_samples=n_samples,
                n_features=n_features,
                instability_type=instability_type,
                n_corrupted=n_corrupted,
                noise_scale=2.0,           # Increased from 0.5
                shift_magnitude=3.0,       # Increased from 1.0
                shift_fraction=0.5,        # Increased from 0.3
                interaction_strength=2.0,  # Increased from 0.5
                missing_fraction=0.3,      # Increased from 0.1
            )
            
            print_progress(f"  Corrupted features: {metadata['corrupted_features']}")
            print_progress(f"  Clean features: {metadata['clean_features']}")
            
            # Run the test
            print_progress(f"  Running stability analysis...")
            result = validator.run_test(
                X, y, metadata,
                threshold=threshold,
                use_permutation=False,
                n_resamples=20,
                resample_frac=0.8,
                random_state=42,
            )
            
            # Store results
            all_results["test_results"][type_name] = {
                "instability_type": instability_type.value,
                "detection_rate": result.detection_rate,
                "false_positive_rate": result.false_positive_rate,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
                "threshold_used": result.threshold_used,
                "injected_features": result.injected_features,
                "clean_features": result.clean_features,
                "feature_scores": result.feature_scores,
            }
            
            # Print results
            print_progress(f"  Results:")
            print_progress(f"    Detection Rate: {result.detection_rate:.1%}")
            print_progress(f"    False Positive Rate: {result.false_positive_rate:.1%}")
            print_progress(f"    Precision: {result.precision:.1%}")
            print_progress(f"    Recall: {result.recall:.1%}")
            print_progress(f"    F1 Score: {result.f1_score:.3f}")
            
            # Store for metrics aggregation
            detection_metrics.append({
                "instability_type": type_name,
                "detection_rate": result.detection_rate,
                "false_positive_rate": result.false_positive_rate,
                "precision": result.precision,
                "recall": result.recall,
                "f1_score": result.f1_score,
            })
            
        except Exception as e:
            print_progress(f"  ERROR: {e}")
            import traceback
            traceback.print_exc()
            all_results["test_results"][type_name] = {
                "instability_type": instability_type.value,
                "error": str(e),
            }
        
        print_progress("")
    
    # Compute summary statistics
    if detection_metrics:
        metrics_df = pd.DataFrame(detection_metrics)
        
        all_results["summary"] = {
            "mean_detection_rate": float(metrics_df["detection_rate"].mean()),
            "std_detection_rate": float(metrics_df["detection_rate"].std()),
            "mean_false_positive_rate": float(metrics_df["false_positive_rate"].mean()),
            "mean_precision": float(metrics_df["precision"].mean()),
            "mean_recall": float(metrics_df["recall"].mean()),
            "mean_f1_score": float(metrics_df["f1_score"].mean()),
            "n_tests": len(detection_metrics),
        }
        
        print_progress("=" * 60)
        print_progress("SUMMARY STATISTICS")
        print_progress("=" * 60)
        print_progress(f"  Mean Detection Rate: {all_results['summary']['mean_detection_rate']:.1%}")
        print_progress(f"  Std Detection Rate: {all_results['summary']['std_detection_rate']:.1%}")
        print_progress(f"  Mean False Positive Rate: {all_results['summary']['mean_false_positive_rate']:.1%}")
        print_progress(f"  Mean Precision: {all_results['summary']['mean_precision']:.1%}")
        print_progress(f"  Mean F1 Score: {all_results['summary']['mean_f1_score']:.3f}")
        print_progress("")
    
    return all_results, detection_metrics


def run_detailed_test(instability_type, type_name: str):
    """
    Run a more detailed test for a specific instability type.
    
    Parameters
    ----------
    instability_type : InstabilityType
        Type of instability to test.
    type_name : str
        Human-readable name.
    
    Returns
    -------
    dict
        Detailed test results.
    """
    from bootstrap_stability import (
        SyntheticValidation,
        InstabilityType,
        BootstrapStability,
    )
    
    print_progress(f"Running detailed test for {type_name}...")
    
    validator = SyntheticValidation(random_state=42)
    
    # Generate larger dataset for detailed analysis
    X, y, metadata = validator.generate_test_data(
        n_samples=2000,
        n_features=15,
        instability_type=instability_type,
        n_corrupted=5,
    )
    
    # Run analysis with more resamples
    result = validator.run_test(
        X, y, metadata,
        threshold=0.5,
        use_permutation=False,
        n_resamples=30,
    )
    
    return {
        "type_name": type_name,
        "n_samples": 2000,
        "n_features": 15,
        "n_corrupted": 5,
        "detection_rate": result.detection_rate,
        "false_positive_rate": result.false_positive_rate,
        "precision": result.precision,
        "recall": result.recall,
        "f1_score": result.f1_score,
        "feature_scores": result.feature_scores,
        "injected_features": result.injected_features,
        "detected_features": [
            f for f in result.injected_features 
            if result.feature_scores.get(f, 0) >= result.threshold_used
        ],
        "missed_features": [
            f for f in result.injected_features 
            if result.feature_scores.get(f, 0) < result.threshold_used
        ],
    }


def generate_validation_report(all_results: dict, detection_metrics: list):
    """
    Generate a comprehensive validation report.
    
    Parameters
    ----------
    all_results : dict
        Complete validation results.
    detection_metrics : list
        List of detection metric dictionaries.
    
    Returns
    -------
    str
        Report text.
    """
    print_progress("=" * 60)
    print_progress("GENERATING VALIDATION REPORT")
    print_progress("=" * 60)
    
    lines = [
        "=" * 70,
        "SYNTHETIC VALIDATION SUITE - REPORT",
        "=" * 70,
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "-" * 70,
        "CONFIGURATION",
        "-" * 70,
        "",
    ]
    
    config = all_results.get("configuration", {})
    lines.extend([
        f"  Samples per test:     {config.get('n_samples', 'N/A')}",
        f"  Features per test:    {config.get('n_features', 'N/A')}",
        f"  Corrupted features:   {config.get('n_corrupted', 'N/A')}",
        f"  Detection threshold:  {config.get('threshold', 'N/A')}",
        "",
        "-" * 70,
        "DETECTION RESULTS BY INSTABILITY TYPE",
        "-" * 70,
        "",
    ])
    
    # Results by type
    test_results = all_results.get("test_results", {})
    
    for type_name, result in test_results.items():
        if "error" in result:
            lines.extend([
                f"{type_name}:",
                f"  ERROR: {result['error']}",
                "",
            ])
            continue
        
        lines.extend([
            f"{type_name}:",
            f"  Detection Rate:       {result['detection_rate']:.1%}",
            f"  False Positive Rate:  {result['false_positive_rate']:.1%}",
            f"  Precision:            {result['precision']:.1%}",
            f"  Recall:               {result['recall']:.1%}",
            f"  F1 Score:             {result['f1_score']:.3f}",
            "",
            f"  Injected Features:    {result['injected_features']}",
            f"  Clean Features:       {result['clean_features'][:3]}...",
            "",
            f"  Feature Scores:",
        ])
        
        # Show feature scores
        scores = result.get("feature_scores", {})
        sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        
        for feat, score in sorted_scores[:5]:  # Top 5
            flag = "FLAGGED" if score >= result['threshold_used'] else "clean"
            lines.append(f"    {feat:<20}: {score:.4f} [{flag}]")
        
        lines.append("")
    
    # Summary statistics
    lines.extend([
        "-" * 70,
        "SUMMARY STATISTICS",
        "-" * 70,
        "",
    ])
    
    summary = all_results.get("summary", {})
    if summary:
        lines.extend([
            f"  Mean Detection Rate:       {summary.get('mean_detection_rate', 0):.1%}",
            f"  Std Detection Rate:        {summary.get('std_detection_rate', 0):.1%}",
            f"  Mean False Positive Rate:  {summary.get('mean_false_positive_rate', 0):.1%}",
            f"  Mean Precision:            {summary.get('mean_precision', 0):.1%}",
            f"  Mean Recall:               {summary.get('mean_recall', 0):.1%}",
            f"  Mean F1 Score:             {summary.get('mean_f1_score', 0):.3f}",
            f"  Total Tests:               {summary.get('n_tests', 0)}",
            "",
        ])
    
    # Detection metrics table
    if detection_metrics:
        lines.extend([
            "-" * 70,
            "DETECTION METRICS TABLE",
            "-" * 70,
            "",
            f"{'Instability Type':<25} {'Det. Rate':>12} {'FPR':>10} {'F1':>8}",
            "-" * 55,
        ])
        
        for m in detection_metrics:
            lines.append(
                f"{m['instability_type']:<25} "
                f"{m['detection_rate']:>11.1%} "
                f"{m['false_positive_rate']:>9.1%} "
                f"{m['f1_score']:>8.3f}"
            )
        
        lines.append("")
    
    # Interpretation
    lines.extend([
        "-" * 70,
        "INTERPRETATION GUIDE",
        "-" * 70,
        "",
        "Detection Rate: Proportion of corrupted features correctly identified.",
        "  - >80%: Excellent detection",
        "  - 60-80%: Good detection",
        "  - 40-60%: Moderate detection",
        "  - <40%: Poor detection (may need parameter tuning)",
        "",
        "False Positive Rate: Proportion of clean features incorrectly flagged.",
        "  - <10%: Excellent specificity",
        "  - 10-20%: Good specificity",
        "  - 20-30%: Moderate specificity",
        "  - >30%: High false alarm rate",
        "",
        "F1 Score: Harmonic mean of precision and recall.",
        "  - >0.8: Excellent",
        "  - 0.6-0.8: Good",
        "  - 0.4-0.6: Moderate",
        "  - <0.4: Poor",
        "",
        "=" * 70,
        "END OF REPORT",
        "=" * 70,
    ])
    
    return "\n".join(lines)


def main():
    """Main entry point for the synthetic validation test script."""
    print_progress("=" * 60)
    print_progress("SYNTHETIC VALIDATION TEST")
    print_progress("=" * 60)
    print_progress(f"Output directory: {OUTPUT_DIR.absolute()}")
    print_progress("")
    
    # Step 1: Run the full validation suite
    print_progress("Step 1: Running full validation suite...")
    all_results, detection_metrics = run_synthetic_validation_suite()
    
    # Step 2: Save complete results
    print_progress("Step 2: Saving results...")
    save_results(all_results, "validation_results.json")
    
    # Step 3: Save detection metrics as CSV
    if detection_metrics:
        metrics_df = pd.DataFrame(detection_metrics)
        save_dataframe(metrics_df, "detection_metrics.csv")
    
    # Step 4: Generate and save report
    print_progress("Step 3: Generating validation report...")
    report = generate_validation_report(all_results, detection_metrics)
    
    report_path = OUTPUT_DIR / "validation_report.txt"
    with open(report_path, 'w') as f:
        f.write(report)
    print_progress(f"Saved: {report_path}")
    
    # Print report to console
    print_progress("")
    print(report)
    
    print_progress("")
    print_progress("=" * 60)
    print_progress("TEST COMPLETE")
    print_progress(f"All results saved to: {OUTPUT_DIR.absolute()}")
    print_progress("=" * 60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
