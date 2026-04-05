"""
Synthetic Validation Suite for Ground Truth Testing.

This module provides tools to generate synthetic data with known instabilities
and test the detection capabilities of the bootstrap stability analysis.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Type, Any

import numpy as np
import pandas as pd

from .analyzer import BootstrapStability


class InstabilityType(Enum):
    """Types of instabilities that can be injected into synthetic data."""
    
    HETEROSCEDASTIC = "heteroscedastic"  # Noise increases with feature value
    DISTRIBUTION_SHIFT = "distribution_shift"  # Mean/variance shift across samples
    INTERACTION = "interaction"  # Instability depends on another feature
    MISSING_NOT_AT_RANDOM = "mnar"  # Missingness correlated with value


@dataclass
class TestResult:
    """
    Results from a synthetic validation test.
    
    Attributes
    ----------
    test_name : str
        Name of the test performed.
    instability_type : str
        Type of instability tested (from InstabilityType).
    injected_features : List[str]
        Features with known instability (ground truth positives).
    clean_features : List[str]
        Features that should be stable (ground truth negatives).
    detection_rate : float
        True positive rate (proportion of corrupted features detected).
    false_positive_rate : float
        Proportion of clean features incorrectly flagged as unstable.
    precision : float
        Precision of detection (TP / (TP + FP)).
    recall : float
        Recall of detection (TP / (TP + FN)).
    f1_score : float
        F1 score combining precision and recall.
    feature_scores : Dict[str, float]
        Complexity scores per feature.
    threshold_used : float
        Threshold used to flag features as unstable.
    """
    
    test_name: str
    instability_type: str
    injected_features: List[str]
    clean_features: List[str]
    detection_rate: float
    false_positive_rate: float
    precision: float
    recall: float
    f1_score: float
    feature_scores: Dict[str, float]
    threshold_used: float
    
    def summary(self) -> Dict[str, Any]:
        """Return a summary dictionary of the test results."""
        return {
            "test_name": self.test_name,
            "instability_type": self.instability_type,
            "n_injected": len(self.injected_features),
            "n_clean": len(self.clean_features),
            "detection_rate": self.detection_rate,
            "false_positive_rate": self.false_positive_rate,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "threshold_used": self.threshold_used,
        }


class SyntheticValidation:
    """
    Validation suite for testing instability detection capabilities.
    
    Generates synthetic data with known instabilities and evaluates how well
    the bootstrap stability analysis detects them.
    
    Parameters
    ----------
    random_state : int, default=42
        Random seed for reproducibility.
    
    Examples
    --------
    >>> validator = SyntheticValidation(random_state=42)
    >>> X, y, metadata = validator.generate_test_data(
    ...     n_samples=1000,
    ...     n_features=10,
    ...     instability_type=InstabilityType.HETEROSCEDASTIC,
    ...     n_corrupted=3
    ... )
    >>> result = validator.run_test(X, y, metadata)
    >>> print(f"Detection rate: {result.detection_rate:.2%}")
    """
    
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self._rng = np.random.RandomState(random_state)
    
    def _reset_rng(self):
        """Reset the random number generator to initial state."""
        self._rng = np.random.RandomState(self.random_state)
    
    def generate_test_data(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        instability_type: InstabilityType = InstabilityType.HETEROSCEDASTIC,
        n_corrupted: int = 3,
        noise_scale: float = 0.5,
        shift_magnitude: float = 1.0,
        shift_fraction: float = 0.3,
        interaction_strength: float = 0.5,
        missing_fraction: float = 0.1,
        **kwargs
    ) -> Tuple[pd.DataFrame, pd.Series, Dict]:
        """
        Generate synthetic data with known instabilities.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples to generate.
        n_features : int, default=10
            Total number of features.
        instability_type : InstabilityType, default=HETEROSCEDASTIC
            Type of instability to inject.
        n_corrupted : int, default=3
            Number of features to corrupt with instability.
        noise_scale : float, default=0.5
            Scale of heteroscedastic noise.
        shift_magnitude : float, default=1.0
            Magnitude of distribution shift.
        shift_fraction : float, default=0.3
            Fraction of samples affected by distribution shift.
        interaction_strength : float, default=0.5
            Strength of interaction-dependent instability.
        missing_fraction : float, default=0.1
            Fraction of values to make missing (for MNAR).
        **kwargs
            Additional parameters passed to specific instability generators.
        
        Returns
        -------
        X : pd.DataFrame
            Feature matrix with shape (n_samples, n_features).
        y : pd.Series
            Target variable (binary classification).
        metadata : Dict
            Dictionary containing:
            - 'corrupted_features': List of feature names with instability
            - 'clean_features': List of stable feature names
            - 'instability_type': Type of instability injected
            - 'instability_params': Parameters used for injection
        """
        self._reset_rng()
        
        # Generate base features from standard normal
        X = pd.DataFrame(
            self._rng.randn(n_samples, n_features),
            columns=[f"feature_{i}" for i in range(n_features)]
        )
        
        # Generate binary target with some signal from features
        base_signal = X.iloc[:, :min(3, n_features)].sum(axis=1)
        y_prob = 1 / (1 + np.exp(-base_signal))  # Sigmoid
        y = pd.Series((y_prob > 0.5).astype(int), name="target")
        
        # Select features to corrupt
        corrupted_indices = self._rng.choice(
            n_features, size=min(n_corrupted, n_features), replace=False
        )
        corrupted_features = [f"feature_{i}" for i in corrupted_indices]
        clean_features = [f"feature_{i}" for i in range(n_features) 
                         if i not in corrupted_indices]
        
        # Inject instability based on type
        if instability_type == InstabilityType.HETEROSCEDASTIC:
            X = self._inject_heteroscedastic(
                X, corrupted_features, noise_scale=noise_scale
            )
        elif instability_type == InstabilityType.DISTRIBUTION_SHIFT:
            X = self._inject_distribution_shift(
                X, corrupted_features, 
                shift_magnitude=shift_magnitude,
                shift_fraction=shift_fraction
            )
        elif instability_type == InstabilityType.INTERACTION:
            X = self._inject_interaction(
                X, corrupted_features, 
                interaction_strength=interaction_strength
            )
        elif instability_type == InstabilityType.MISSING_NOT_AT_RANDOM:
            X = self._inject_mnar(
                X, corrupted_features,
                missing_fraction=missing_fraction
            )
        else:
            raise ValueError(f"Unknown instability type: {instability_type}")
        
        metadata = {
            "corrupted_features": corrupted_features,
            "clean_features": clean_features,
            "instability_type": instability_type.value,
            "instability_params": {
                "noise_scale": noise_scale,
                "shift_magnitude": shift_magnitude,
                "shift_fraction": shift_fraction,
                "interaction_strength": interaction_strength,
                "missing_fraction": missing_fraction,
            }
        }
        
        return X, y, metadata
    
    def _inject_heteroscedastic(
        self, 
        X: pd.DataFrame, 
        corrupted_features: List[str],
        noise_scale: float = 0.5
    ) -> pd.DataFrame:
        """
        Inject heteroscedastic noise (noise scales with feature value).
        
        The noise magnitude increases with the absolute value of the feature,
        making higher values less stable across bootstrap samples.
        """
        X = X.copy()
        n_samples = len(X)
        
        for feat in corrupted_features:
            base_values = X[feat].values
            # Noise proportional to absolute feature value
            noise = np.abs(base_values) * self._rng.normal(0, noise_scale, n_samples)
            X[feat] = base_values + noise
        
        return X
    
    def _inject_distribution_shift(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        shift_magnitude: float = 1.0,
        shift_fraction: float = 0.3
    ) -> pd.DataFrame:
        """
        Inject distribution shift (different mean/variance in sample regions).
        
        A fraction of samples have their values shifted, creating instability
        in the distribution across different bootstrap samples.
        """
        X = X.copy()
        n_samples = len(X)
        n_shift = int(n_samples * shift_fraction)
        
        for feat in corrupted_features:
            # Select random samples to shift
            shift_idx = self._rng.choice(n_samples, size=n_shift, replace=False)
            
            # Apply shift with some random variation
            shifts = self._rng.normal(shift_magnitude, shift_magnitude * 0.3, n_shift)
            X.loc[X.index[shift_idx], feat] += shifts
        
        return X
    
    def _inject_interaction(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        interaction_strength: float = 0.5
    ) -> pd.DataFrame:
        """
        Inject interaction-dependent instability.
        
        The feature value depends on another feature, creating instability
        when the relationship varies across bootstrap samples.
        """
        X = X.copy()
        n_samples = len(X)
        feature_names = list(X.columns)
        
        for i, feat in enumerate(corrupted_features):
            # Choose another feature as interaction partner
            other_features = [f for f in feature_names if f != feat]
            partner = self._rng.choice(other_features)
            
            # Add interaction term
            interaction_term = interaction_strength * X[partner].values
            # Add noise that depends on the partner feature
            noise = interaction_term * self._rng.normal(0, 0.5, n_samples)
            X[feat] = X[feat].values + noise
        
        return X
    
    def _inject_mnar(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        missing_fraction: float = 0.1
    ) -> pd.DataFrame:
        """
        Inject missing-not-at-random (MNAR) pattern.
        
        Missing values are correlated with feature magnitude, making
        the feature distribution unstable across samples.
        """
        X = X.copy()
        n_samples = len(X)
        
        for feat in corrupted_features:
            # Probability of missing increases with absolute value
            abs_values = np.abs(X[feat].values)
            # Normalize to [0, 1] range
            normalized = (abs_values - abs_values.min()) / (abs_values.max() - abs_values.min() + 1e-8)
            # Higher values more likely to be missing
            missing_prob = normalized * missing_fraction * 2  # Scale up to reach target fraction
            
            # Generate missing mask
            missing_mask = self._rng.random(n_samples) < missing_prob
            
            # Set to NaN
            X.loc[missing_mask, feat] = np.nan
        
        return X
    
    def run_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Dict,
        analyzer_class: Type[BootstrapStability] = BootstrapStability,
        threshold: float = 0.5,
        **analyzer_kwargs
    ) -> TestResult:
        """
        Run stability analysis and compute detection metrics.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature matrix.
        y : pd.Series
            Target variable.
        metadata : Dict
            Metadata from generate_test_data containing ground truth.
        analyzer_class : Type[BootstrapStability], default=BootstrapStability
            Analyzer class to use for stability analysis.
        threshold : float, default=0.5
            Complexity score threshold for flagging features as unstable.
            Higher scores indicate more instability.
        **analyzer_kwargs
            Additional arguments passed to the analyzer constructor.
        
        Returns
        -------
        TestResult
            Results including detection metrics and feature scores.
        """
        # Create analyzer with defaults
        default_kwargs = {
            "n_resamples": 20,
            "resample_frac": 0.8,
            "random_state": self.random_state,
        }
        default_kwargs.update(analyzer_kwargs)
        
        analyzer = analyzer_class(**default_kwargs)
        
        # Prepare data for analysis
        df = X.copy()
        df["target"] = y.values
        
        # Run stability analysis on each feature
        feature_scores = {}
        feature_cols = metadata["corrupted_features"] + metadata["clean_features"]
        
        for feat in feature_cols:
            try:
                result = analyzer.fit(df, feature_col=feat, target_col="target")
                # Get complexity score - higher means more unstable
                score = result.get("complexity_score", 0.0)
                if score is None or np.isnan(score):
                    score = 0.0
                feature_scores[feat] = float(score)
            except Exception as e:
                # If analysis fails, assign middle score
                feature_scores[feat] = threshold
        
        # Compute detection metrics
        corrupted_features = metadata["corrupted_features"]
        clean_features = metadata["clean_features"]
        
        # True positives: corrupted features flagged as unstable
        detected_corrupted = [
            f for f in corrupted_features 
            if feature_scores.get(f, 0) >= threshold
        ]
        
        # False positives: clean features flagged as unstable
        flagged_clean = [
            f for f in clean_features 
            if feature_scores.get(f, 0) >= threshold
        ]
        
        # False negatives: corrupted features not flagged
        missed_corrupted = [
            f for f in corrupted_features 
            if feature_scores.get(f, 0) < threshold
        ]
        
        # True negatives: clean features correctly identified
        correct_clean = [
            f for f in clean_features 
            if feature_scores.get(f, 0) < threshold
        ]
        
        # Calculate metrics
        n_corrupted = len(corrupted_features)
        n_clean = len(clean_features)
        
        detection_rate = len(detected_corrupted) / n_corrupted if n_corrupted > 0 else 0.0
        false_positive_rate = len(flagged_clean) / n_clean if n_clean > 0 else 0.0
        
        # Precision and recall
        n_flagged = len(detected_corrupted) + len(flagged_clean)
        precision = len(detected_corrupted) / n_flagged if n_flagged > 0 else 0.0
        recall = detection_rate  # Same as detection rate
        
        # F1 score
        f1_score = (
            2 * precision * recall / (precision + recall) 
            if (precision + recall) > 0 else 0.0
        )
        
        # Get test name from instability type
        test_name = metadata["instability_type"].replace("_", " ").title()
        
        return TestResult(
            test_name=test_name,
            instability_type=metadata["instability_type"],
            injected_features=corrupted_features,
            clean_features=clean_features,
            detection_rate=detection_rate,
            false_positive_rate=false_positive_rate,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            feature_scores=feature_scores,
            threshold_used=threshold,
        )
    
    def run_full_suite(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        analyzer_class: Type[BootstrapStability] = BootstrapStability,
        threshold: float = 0.5,
        n_corrupted: int = 3,
        verbose: bool = True,
        **analyzer_kwargs
    ) -> List[TestResult]:
        """
        Run all test types and return results.
        
        Parameters
        ----------
        n_samples : int, default=1000
            Number of samples per test.
        n_features : int, default=10
            Number of features per test.
        analyzer_class : Type[BootstrapStability], default=BootstrapStability
            Analyzer class to use.
        threshold : float, default=0.5
            Complexity threshold for flagging unstable features.
        n_corrupted : int, default=3
            Number of features to corrupt per test.
        verbose : bool, default=True
            Whether to print progress information.
        **analyzer_kwargs
            Additional arguments passed to the analyzer.
        
        Returns
        -------
        List[TestResult]
            List of results from all test types.
        """
        results = []
        
        for instability_type in InstabilityType:
            if verbose:
                print(f"Running test: {instability_type.value}...")
            
            try:
                X, y, metadata = self.generate_test_data(
                    n_samples=n_samples,
                    n_features=n_features,
                    instability_type=instability_type,
                    n_corrupted=n_corrupted
                )
                
                result = self.run_test(
                    X, y, metadata,
                    analyzer_class=analyzer_class,
                    threshold=threshold,
                    **analyzer_kwargs
                )
                results.append(result)
                
                if verbose:
                    print(f"  Detection rate: {result.detection_rate:.1%}")
                    print(f"  False positive rate: {result.false_positive_rate:.1%}")
                    print(f"  F1 score: {result.f1_score:.2f}")
            
            except Exception as e:
                if verbose:
                    print(f"  Error: {e}")
        
        return results
    
    def generate_report(self, results: List[TestResult]) -> str:
        """
        Generate a formatted validation report.
        
        Parameters
        ----------
        results : List[TestResult]
            List of test results to include in report.
        
        Returns
        -------
        str
            Formatted report string.
        """
        lines = []
        lines.append("=" * 50)
        lines.append("Synthetic Validation Report")
        lines.append("=" * 50)
        lines.append("")
        
        # Per-test results
        for result in results:
            lines.append(f"Test: {result.test_name}")
            lines.append("-" * 40)
            
            # Format injected features
            injected = ", ".join(result.injected_features)
            lines.append(f"- Injected instability in: {injected}")
            
            # Detection summary
            n_detected = int(result.detection_rate * len(result.injected_features))
            lines.append(
                f"- Detection rate: {result.detection_rate:.0%} "
                f"({n_detected}/{len(result.injected_features)} detected)"
            )
            
            # False positive summary
            n_fp = int(result.false_positive_rate * len(result.clean_features))
            lines.append(
                f"- False positive rate: {result.false_positive_rate:.0%} "
                f"({n_fp}/{len(result.clean_features)} clean features flagged)"
            )
            
            # Metrics
            lines.append(
                f"- Precision: {result.precision:.2f}, "
                f"Recall: {result.recall:.2f}, "
                f"F1: {result.f1_score:.2f}"
            )
            lines.append("")
        
        # Summary statistics
        if results:
            lines.append("=" * 50)
            lines.append("Summary")
            lines.append("=" * 50)
            
            avg_detection = np.mean([r.detection_rate for r in results])
            avg_fpr = np.mean([r.false_positive_rate for r in results])
            avg_f1 = np.mean([r.f1_score for r in results])
            
            lines.append(f"- Average detection rate: {avg_detection:.0%}")
            lines.append(f"- Average false positive rate: {avg_fpr:.0%}")
            lines.append(f"- Overall F1: {avg_f1:.2f}")
        
        return "\n".join(lines)


def print_synthetic_report(results: List[TestResult]) -> None:
    """
    Print a formatted validation report to console.
    
    Parameters
    ----------
    results : List[TestResult]
        List of test results to report.
    """
    validator = SyntheticValidation()
    print(validator.generate_report(results))
