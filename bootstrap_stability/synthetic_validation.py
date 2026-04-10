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
from .permutation_baseline import PermutationBaseline


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
    feature_p_values: Dict[str, float] = field(default_factory=dict)
    detection_method: str = "raw_threshold"
    
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
            "detection_method": self.detection_method,
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
        noise_scale: float = 3.0,
        shift_magnitude: float = 4.0,
        shift_fraction: float = 0.4,
        interaction_strength: float = 3.0,
        missing_fraction: float = 0.3,
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
        noise_scale : float, default=3.0
            Scale of heteroscedastic noise.
        shift_magnitude : float, default=4.0
            Magnitude of distribution shift.
        shift_fraction : float, default=0.4
            Fraction of samples affected by distribution shift.
        interaction_strength : float, default=3.0
            Strength of interaction-dependent instability.
        missing_fraction : float, default=0.3
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
        # Per-type seed so each instability type gets different but reproducible data
        type_seed = self.random_state + list(InstabilityType).index(instability_type)
        self._rng = np.random.RandomState(type_seed)

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
        
        # Cache target for injection methods that need target-dependent behavior
        self._y_cache = y.values

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
        noise_scale: float = 1.5
    ) -> pd.DataFrame:
        """
        Inject an influential minority with target-aligned extreme values.

        Creates structural instability by making ~10% of observations have
        extreme feature values that are strongly correlated with the target.
        Including/excluding these influential points in bootstrap samples
        swings Spearman/IV at any sample size, producing a positive floor.
        """
        X = X.copy()
        n_samples = len(X)
        y = self._y_cache

        for feat in corrupted_features:
            n_influential = int(0.20 * n_samples)
            idx = self._rng.choice(n_samples, n_influential, replace=False)
            # Extreme values aligned with target: y=1 → positive, y=0 → negative
            y_vals = y[idx]
            direction = 2.0 * y_vals - 1.0  # -1 for y=0, +1 for y=1
            magnitude = self._rng.uniform(4, 8, n_influential) * noise_scale
            X.iloc[idx, X.columns.get_loc(feat)] = direction * magnitude

        return X

    def _inject_distribution_shift(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        shift_magnitude: float = 2.5,
        shift_fraction: float = 0.4
    ) -> pd.DataFrame:
        """
        Inject an influential minority with larger magnitude than heteroscedastic.

        Same core mechanism (target-aligned extreme values) but with a larger
        fraction (15%) and wider magnitude range, simulating a distribution
        with a heavier influential tail.
        """
        X = X.copy()
        n_samples = len(X)
        y = self._y_cache

        for feat in corrupted_features:
            n_influential = int(0.20 * n_samples)
            idx = self._rng.choice(n_samples, n_influential, replace=False)
            direction = 2.0 * y[idx] - 1.0
            magnitude = self._rng.uniform(4, 8, n_influential) * shift_magnitude
            X.iloc[idx, X.columns.get_loc(feat)] = direction * magnitude

        return X

    def _inject_interaction(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        interaction_strength: float = 1.5
    ) -> pd.DataFrame:
        """
        Inject influential minority with partner-modulated magnitude.

        Creates target-aligned extreme values (like heteroscedastic) but
        the magnitude is amplified when a partner feature is extreme.
        This means the influential points cluster in a partner-dependent
        region, making bootstrap instability depend on which region is
        sampled — a genuinely interaction-driven structural instability.
        """
        X = X.copy()
        n_samples = len(X)
        y = self._y_cache
        feature_names = list(X.columns)

        for feat in corrupted_features:
            other_features = [f for f in feature_names if f != feat]
            partner = self._rng.choice(other_features)
            partner_abs = np.abs(X[partner].values)
            # Normalize partner to [0.5, 1.5] range for magnitude modulation
            partner_scale = 0.5 + (partner_abs / (partner_abs.max() + 1e-8))

            n_influential = int(0.20 * n_samples)
            idx = self._rng.choice(n_samples, n_influential, replace=False)

            direction = 2.0 * y[idx] - 1.0
            base_magnitude = self._rng.uniform(4, 8, n_influential)
            # Partner modulates magnitude — extreme partner values amplify
            magnitude = base_magnitude * partner_scale[idx] * interaction_strength
            X.iloc[idx, X.columns.get_loc(feat)] = direction * magnitude

        return X

    def _inject_mnar(
        self,
        X: pd.DataFrame,
        corrupted_features: List[str],
        missing_fraction: float = 0.3
    ) -> pd.DataFrame:
        """
        Inject influential minority plus moderate target-dependent missingness.

        Combines two mechanisms: (1) an influential minority with extreme
        target-aligned values, and (2) moderate MNAR that removes some
        informative observations. The combination creates structural
        instability from the influential points while the missingness
        adds noise that prevents convergence.
        """
        X = X.copy()
        n_samples = len(X)
        y = self._y_cache

        for feat in corrupted_features:
            # First: influential minority (20%)
            n_influential = int(0.20 * n_samples)
            idx = self._rng.choice(n_samples, n_influential, replace=False)
            direction = 2.0 * y[idx] - 1.0
            magnitude = self._rng.uniform(4, 8, n_influential)
            X.iloc[idx, X.columns.get_loc(feat)] = direction * magnitude

            # Second: moderate MNAR (15%) on non-influential observations
            remaining = np.setdiff1d(np.arange(n_samples), idx)
            abs_vals = np.abs(X[feat].values[remaining])
            normalized = (abs_vals - abs_vals.min()) / (abs_vals.max() - abs_vals.min() + 1e-8)
            missing_prob = normalized * missing_fraction
            missing_mask = self._rng.random(len(remaining)) < missing_prob
            X.iloc[remaining[missing_mask], X.columns.get_loc(feat)] = np.nan

        return X
    
    def run_test(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        metadata: Dict,
        analyzer_class: Type[BootstrapStability] = BootstrapStability,
        threshold: float = 0.05,
        use_permutation: bool = True,
        n_permutations: int = 30,
        n_jobs: int = -1,
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
        threshold : float, default=0.05
            When use_permutation=True, this is the significance level (alpha)
            for the permutation test. When use_permutation=False, this is the
            raw complexity score threshold for flagging features.
        use_permutation : bool, default=True
            If True, use PermutationBaseline for calibrated detection.
            If False, use raw complexity score threshold (legacy behavior).
        n_permutations : int, default=20
            Number of permutations for the null distribution. Must be >= 20
            for p < 0.05 to be achievable.
        n_jobs : int, default=-1
            Number of parallel jobs for permutation runs.
        **analyzer_kwargs
            Additional arguments passed to the analyzer constructor.

        Returns
        -------
        TestResult
            Results including detection metrics and feature scores.
        """
        # Create analyzer defaults
        default_kwargs = {
            "n_resamples": 20,
            "resample_frac": 0.8,
            "random_state": self.random_state,
        }
        default_kwargs.update(analyzer_kwargs)

        # Prepare data for analysis
        df = X.copy()
        df["target"] = y.values

        corrupted_features = metadata["corrupted_features"]
        clean_features = metadata["clean_features"]
        feature_cols = corrupted_features + clean_features

        feature_scores = {}
        feature_p_values = {}
        feature_flagged = {}

        if use_permutation:
            # Permutation-calibrated detection
            perm = PermutationBaseline(
                n_permutations=n_permutations,
                analyzer_kwargs={
                    "n_resamples": default_kwargs.get("n_resamples", 10),
                    "resample_frac": default_kwargs.get("resample_frac", 0.8),
                    "random_state": self.random_state,
                    "n_jobs": 1,
                },
                alpha=threshold,
                random_state=self.random_state,
                verbose=0,
                n_jobs=n_jobs,
            )

            for feat in feature_cols:
                try:
                    perm_result = perm.fit(
                        df, feature_col=feat, target_col="target",
                    )
                    score = perm_result.get("observed", 0.0)
                    if score is None or np.isnan(score):
                        score = 0.0
                    feature_scores[feat] = float(score)
                    feature_p_values[feat] = float(perm_result.get("p_value", 1.0))
                    feature_flagged[feat] = bool(perm_result.get("significant", False))
                except Exception:
                    feature_scores[feat] = 0.0
                    feature_p_values[feat] = 1.0
                    feature_flagged[feat] = False

            detection_method = "permutation_calibrated"
        else:
            # Legacy raw threshold detection
            analyzer = analyzer_class(**default_kwargs)

            for feat in feature_cols:
                try:
                    result = analyzer.fit(df, feature_col=feat, target_col="target")
                    score = result.get("complexity_score", 0.0)
                    if score is None or np.isnan(score):
                        score = 0.0
                    feature_scores[feat] = float(score)
                except Exception:
                    feature_scores[feat] = 0.0
                feature_flagged[feat] = feature_scores[feat] >= threshold

            detection_method = "raw_threshold"

        # Compute detection metrics
        detected_corrupted = [f for f in corrupted_features if feature_flagged.get(f, False)]
        flagged_clean = [f for f in clean_features if feature_flagged.get(f, False)]

        n_corrupted = len(corrupted_features)
        n_clean = len(clean_features)

        detection_rate = len(detected_corrupted) / n_corrupted if n_corrupted > 0 else 0.0
        false_positive_rate = len(flagged_clean) / n_clean if n_clean > 0 else 0.0

        n_flagged = len(detected_corrupted) + len(flagged_clean)
        precision = len(detected_corrupted) / n_flagged if n_flagged > 0 else 0.0
        recall = detection_rate

        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

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
            feature_p_values=feature_p_values,
            detection_method=detection_method,
        )
    
    def run_full_suite(
        self,
        n_samples: int = 1000,
        n_features: int = 10,
        analyzer_class: Type[BootstrapStability] = BootstrapStability,
        threshold: float = 0.05,
        n_corrupted: int = 3,
        verbose: bool = True,
        use_permutation: bool = True,
        n_permutations: int = 30,
        n_jobs: int = -1,
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
        threshold : float, default=0.05
            Significance level (alpha) when use_permutation=True, or raw
            complexity score threshold when use_permutation=False.
        n_corrupted : int, default=3
            Number of features to corrupt per test.
        verbose : bool, default=True
            Whether to print progress information.
        use_permutation : bool, default=True
            If True, use PermutationBaseline for calibrated detection.
        n_permutations : int, default=20
            Number of permutations for the null distribution.
        n_jobs : int, default=-1
            Number of parallel jobs for permutation runs.
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
                    use_permutation=use_permutation,
                    n_permutations=n_permutations,
                    n_jobs=n_jobs,
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
