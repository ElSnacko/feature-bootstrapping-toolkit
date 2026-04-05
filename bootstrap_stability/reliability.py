"""
Reliability Score Computation Module

This module provides a documented, configurable reliability scoring system for features.
The reliability score is a weighted combination of four components:

1. **Importance** (default weight: 30%)
   - Based on feature importance (e.g., SHAP rank)
   - Higher importance = higher reliability

2. **Stability** (default weight: 40%)
   - Based on complexity score from bootstrap stability
   - Lower complexity = higher stability = higher reliability

3. **Coverage** (default weight: 15%)
   - Ratio of non-NaN metric evaluations
   - Higher coverage = higher reliability

4. **Consistency** (default weight: 15%)
   - Based on cross-seed standard deviation
   - Lower std = higher consistency = higher reliability

Final Score Range: [0, 1] where 1 = most reliable
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd


@dataclass
class ReliabilityConfig:
    """
    Configuration for reliability score computation.
    
    Attributes
    ----------
    importance_weight : float
        Weight for the importance component (default: 0.30)
    stability_weight : float
        Weight for the stability component (default: 0.40)
    coverage_weight : float
        Weight for the coverage component (default: 0.15)
    consistency_weight : float
        Weight for the consistency component (default: 0.15)
    normalization_method : str
        Method for normalizing values to [0, 1] range: "minmax", "rank", or "zscore"
    complexity_min : float
        Minimum expected complexity score for normalization (default: 0.0)
    complexity_max : float
        Maximum expected complexity score for normalization (default: 1.0)
    importance_min : float
        Minimum expected importance score for normalization (default: 0.0)
    importance_max : float
        Maximum expected importance score for normalization (default: 1.0)
    cross_seed_std_min : float
        Minimum expected cross-seed std for normalization (default: 0.0)
    cross_seed_std_max : float
        Maximum expected cross-seed std for normalization (default: 1.0)
    """
    
    # Component weights (must sum to 1.0)
    importance_weight: float = 0.30
    stability_weight: float = 0.40
    coverage_weight: float = 0.15
    consistency_weight: float = 0.15
    
    # Normalization method
    normalization_method: str = "minmax"  # "minmax", "rank", "zscore"
    
    # Normalization bounds for min-max scaling
    complexity_min: float = 0.0
    complexity_max: float = 1.0
    importance_min: float = 0.0
    importance_max: float = 1.0
    cross_seed_std_min: float = 0.0
    cross_seed_std_max: float = 1.0
    
    def validate(self) -> None:
        """Validate that weights sum to 1.0 and values are valid."""
        weight_sum = (
            self.importance_weight + 
            self.stability_weight + 
            self.coverage_weight + 
            self.consistency_weight
        )
        if abs(weight_sum - 1.0) > 1e-6:
            raise ValueError(
                f"Weights must sum to 1.0, got {weight_sum:.6f}. "
                f"importance={self.importance_weight}, stability={self.stability_weight}, "
                f"coverage={self.coverage_weight}, consistency={self.consistency_weight}"
            )
        
        if self.normalization_method not in ("minmax", "rank", "zscore"):
            raise ValueError(
                f"Unknown normalization method: {self.normalization_method}. "
                "Use 'minmax', 'rank', or 'zscore'."
            )
        
        # Validate bounds
        if self.complexity_max <= self.complexity_min:
            raise ValueError(
                f"complexity_max ({self.complexity_max}) must be > complexity_min ({self.complexity_min})"
            )
        if self.importance_max <= self.importance_min:
            raise ValueError(
                f"importance_max ({self.importance_max}) must be > importance_min ({self.importance_min})"
            )
        if self.cross_seed_std_max <= self.cross_seed_std_min:
            raise ValueError(
                f"cross_seed_std_max ({self.cross_seed_std_max}) must be > cross_seed_std_min ({self.cross_seed_std_min})"
            )


@dataclass
class ReliabilityResult:
    """
    Result of reliability score computation for a single feature.
    
    Attributes
    ----------
    feature_name : str
        Name of the feature
    reliability_score : float
        Overall reliability score in range [0, 1], where 1 is most reliable
    stability_component : float
        Stability component score in range [0, 1]
    importance_component : float
        Importance component score in range [0, 1]
    coverage_component : float
        Coverage component score in range [0, 1]
    consistency_component : float
        Consistency component score in range [0, 1]
    formula : str
        Human-readable formula used for computation
    """
    
    feature_name: str
    reliability_score: float
    stability_component: float
    importance_component: float
    coverage_component: float
    consistency_component: float
    formula: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "feature_name": self.feature_name,
            "reliability_score": self.reliability_score,
            "stability_component": self.stability_component,
            "importance_component": self.importance_component,
            "coverage_component": self.coverage_component,
            "consistency_component": self.consistency_component,
            "formula": self.formula,
        }
    
    def __repr__(self) -> str:
        return (
            f"ReliabilityResult(feature='{self.feature_name}', "
            f"score={self.reliability_score:.3f}, "
            f"stability={self.stability_component:.3f}, "
            f"importance={self.importance_component:.3f}, "
            f"coverage={self.coverage_component:.3f}, "
            f"consistency={self.consistency_component:.3f})"
        )


class ReliabilityScorer:
    """
    Compute feature reliability scores with documented formulas.
    
    The reliability score is a weighted combination of four components:
    
    1. **Importance** (default weight: 30%)
       - Based on feature importance (e.g., SHAP importance)
       - Higher importance = higher reliability
       - Formula: `importance = normalize(importance_score)`
    
    2. **Stability** (default weight: 40%)
       - Based on complexity score from bootstrap stability analysis
       - Lower complexity = higher stability = higher reliability
       - Formula: `stability = 1 - normalize(complexity_score)`
    
    3. **Coverage** (default weight: 15%)
       - Ratio of non-NaN metric evaluations
       - Higher coverage = higher reliability
       - Formula: `coverage = coverage_ratio` (already in [0, 1])
    
    4. **Consistency** (default weight: 15%)
       - Based on cross-seed standard deviation
       - Lower std = higher consistency = higher reliability
       - Formula: `consistency = 1 - normalize(cross_seed_std)`
    
    **Final Score:**
    ```
    reliability = w_importance * importance + 
                  w_stability * stability + 
                  w_coverage * coverage + 
                  w_consistency * consistency
    ```
    
    Range: [0, 1] where 1 = most reliable.
    
    Examples
    --------
    >>> config = ReliabilityConfig(
    ...     importance_weight=0.30,
    ...     stability_weight=0.40,
    ...     coverage_weight=0.15,
    ...     consistency_weight=0.15
    ... )
    >>> scorer = ReliabilityScorer(config)
    >>> result = scorer.compute(
    ...     feature_name="PAY_AMT1",
    ...     complexity_score=0.15,
    ...     importance_score=0.85,
    ...     coverage_ratio=0.95,
    ...     cross_seed_std=0.05
    ... )
    >>> print(f"Reliability: {result.reliability_score:.3f}")
    """
    
    def __init__(self, config: Optional[ReliabilityConfig] = None):
        """
        Initialize the ReliabilityScorer.
        
        Parameters
        ----------
        config : ReliabilityConfig, optional
            Configuration for reliability score computation.
            If not provided, uses default weights.
        """
        self.config = config or ReliabilityConfig()
        self.config.validate()
    
    def _normalize_value(
        self, 
        value: float, 
        min_val: float, 
        max_val: float,
        invert: bool = False
    ) -> float:
        """
        Normalize a single value to [0, 1] range using min-max scaling.
        
        Parameters
        ----------
        value : float
            Value to normalize
        min_val : float
            Minimum value for scaling
        max_val : float
            Maximum value for scaling
        invert : bool
            If True, return 1 - normalized_value
        
        Returns
        -------
        float
            Normalized value in [0, 1]
        """
        if np.isnan(value):
            return 0.5  # Default for NaN values
        
        if max_val <= min_val:
            return 0.5
        
        normalized = (value - min_val) / (max_val - min_val)
        # Clamp to [0, 1]
        normalized = max(0.0, min(1.0, normalized))
        
        if invert:
            normalized = 1.0 - normalized
        
        return normalized
    
    def _normalize_series(
        self, 
        series: pd.Series, 
        invert: bool = False
    ) -> pd.Series:
        """
        Normalize a pandas Series to [0, 1] range.
        
        Parameters
        ----------
        series : pd.Series
            Series to normalize
        invert : bool
            If True, return 1 - normalized_value
        
        Returns
        -------
        pd.Series
            Normalized series in [0, 1]
        """
        if self.config.normalization_method == "minmax":
            mn, mx = series.min(), series.max()
            if mx - mn == 0 or np.isnan(mx - mn):
                return pd.Series(0.5, index=series.index)
            normalized = (series - mn) / (mx - mn)
        elif self.config.normalization_method == "rank":
            normalized = series.rank(pct=True)
        elif self.config.normalization_method == "zscore":
            std = series.std()
            if std == 0 or np.isnan(std):
                return pd.Series(0.5, index=series.index)
            normalized = (series - series.mean()) / std
            # Convert z-score to [0, 1]
            mn, mx = normalized.min(), normalized.max()
            if mx - mn == 0:
                return pd.Series(0.5, index=series.index)
            normalized = (normalized - mn) / (mx - mn)
        else:
            raise ValueError(f"Unknown normalization: {self.config.normalization_method}")
        
        if invert:
            normalized = 1 - normalized
        
        return normalized
    
    def _compute_stability_component(self, complexity_score: float) -> float:
        """
        Compute stability component from complexity score.
        
        Formula: stability = 1 - normalize(complexity_score)
        
        Lower complexity = higher stability.
        
        Parameters
        ----------
        complexity_score : float
            Complexity score from bootstrap stability analysis
        
        Returns
        -------
        float
            Stability component in [0, 1]
        """
        return self._normalize_value(
            complexity_score,
            self.config.complexity_min,
            self.config.complexity_max,
            invert=True
        )
    
    def _compute_importance_component(self, importance_score: float) -> float:
        """
        Compute importance component from importance score.
        
        Formula: importance = normalize(importance_score)
        
        Higher importance = higher reliability.
        
        Parameters
        ----------
        importance_score : float
            Importance score (e.g., SHAP importance)
        
        Returns
        -------
        float
            Importance component in [0, 1]
        """
        return self._normalize_value(
            importance_score,
            self.config.importance_min,
            self.config.importance_max,
            invert=False
        )
    
    def _compute_coverage_component(self, coverage_ratio: float) -> float:
        """
        Compute coverage component from coverage ratio.
        
        Coverage is already in [0, 1], so no transformation needed.
        Just handle NaN values.
        
        Parameters
        ----------
        coverage_ratio : float
            Ratio of non-NaN metric evaluations
        
        Returns
        -------
        float
            Coverage component in [0, 1]
        """
        if np.isnan(coverage_ratio):
            return 0.5
        return max(0.0, min(1.0, coverage_ratio))
    
    def _compute_consistency_component(self, cross_seed_std: float) -> float:
        """
        Compute consistency component from cross-seed standard deviation.
        
        Formula: consistency = 1 - normalize(cross_seed_std)
        
        Lower std = higher consistency.
        
        Parameters
        ----------
        cross_seed_std : float
            Standard deviation across different random seeds
        
        Returns
        -------
        float
            Consistency component in [0, 1]
        """
        return self._normalize_value(
            cross_seed_std,
            self.config.cross_seed_std_min,
            self.config.cross_seed_std_max,
            invert=True
        )
    
    def _build_formula_string(self) -> str:
        """Build a human-readable formula string."""
        return (
            f"reliability = "
            f"{self.config.importance_weight:.0%} * importance + "
            f"{self.config.stability_weight:.0%} * stability + "
            f"{self.config.coverage_weight:.0%} * coverage + "
            f"{self.config.consistency_weight:.0%} * consistency"
        )
    
    def compute(
        self,
        feature_name: str,
        complexity_score: float,
        importance_score: float,
        coverage_ratio: float,
        cross_seed_std: float
    ) -> ReliabilityResult:
        """
        Compute reliability score for a single feature.
        
        Parameters
        ----------
        feature_name : str
            Name of the feature
        complexity_score : float
            Complexity score from bootstrap stability analysis.
            Lower values indicate higher stability.
        importance_score : float
            Feature importance score (e.g., SHAP importance).
            Higher values indicate higher importance.
        coverage_ratio : float
            Ratio of non-NaN metric evaluations.
            Range: [0, 1] where 1 means all metrics were evaluated.
        cross_seed_std : float
            Standard deviation of scores across different random seeds.
            Lower values indicate higher consistency.
        
        Returns
        -------
        ReliabilityResult
            Object containing reliability score and all component scores.
        
        Examples
        --------
        >>> scorer = ReliabilityScorer()
        >>> result = scorer.compute(
        ...     feature_name="PAY_AMT1",
        ...     complexity_score=0.15,
        ...     importance_score=0.85,
        ...     coverage_ratio=0.95,
        ...     cross_seed_std=0.05
        ... )
        >>> print(f"Reliability: {result.reliability_score:.3f}")
        """
        # Compute individual components
        stability = self._compute_stability_component(complexity_score)
        importance = self._compute_importance_component(importance_score)
        coverage = self._compute_coverage_component(coverage_ratio)
        consistency = self._compute_consistency_component(cross_seed_std)
        
        # Compute weighted average
        reliability_score = (
            self.config.importance_weight * importance +
            self.config.stability_weight * stability +
            self.config.coverage_weight * coverage +
            self.config.consistency_weight * consistency
        )
        
        # Build formula string
        formula = self._build_formula_string()
        
        return ReliabilityResult(
            feature_name=feature_name,
            reliability_score=reliability_score,
            stability_component=stability,
            importance_component=importance,
            coverage_component=coverage,
            consistency_component=consistency,
            formula=formula,
        )
    
    def compute_all(
        self,
        features_data: Dict[str, Dict[str, float]]
    ) -> Dict[str, ReliabilityResult]:
        """
        Compute reliability scores for multiple features.
        
        Parameters
        ----------
        features_data : dict
            Dictionary mapping feature names to their data.
            Each feature's data should contain:
            - complexity_score: float
            - importance_score: float
            - coverage_ratio: float
            - cross_seed_std: float
        
        Returns
        -------
        dict
            Dictionary mapping feature names to ReliabilityResult objects.
        
        Examples
        --------
        >>> scorer = ReliabilityScorer()
        >>> features = {
        ...     "PAY_AMT1": {
        ...         "complexity_score": 0.15,
        ...         "importance_score": 0.85,
        ...         "coverage_ratio": 0.95,
        ...         "cross_seed_std": 0.05
        ...     },
        ...     "BILL_AMT1": {
        ...         "complexity_score": 0.20,
        ...         "importance_score": 0.75,
        ...         "coverage_ratio": 0.90,
        ...         "cross_seed_std": 0.08
        ...     }
        ... }
        >>> results = scorer.compute_all(features)
        >>> for name, result in results.items():
        ...     print(f"{name}: {result.reliability_score:.3f}")
        """
        results = {}
        for feature_name, data in features_data.items():
            results[feature_name] = self.compute(
                feature_name=feature_name,
                complexity_score=data.get("complexity_score", np.nan),
                importance_score=data.get("importance_score", np.nan),
                coverage_ratio=data.get("coverage_ratio", np.nan),
                cross_seed_std=data.get("cross_seed_std", np.nan),
            )
        return results
    
    def compute_from_dataframe(
        self,
        df: pd.DataFrame,
        complexity_col: str = "complexity_score",
        importance_col: str = "importance_score",
        coverage_col: str = "coverage_ratio",
        consistency_col: str = "cross_seed_std",
        feature_col: str = "feature_name"
    ) -> pd.DataFrame:
        """
        Compute reliability scores from a DataFrame.
        
        This method normalizes values across all features in the DataFrame
        using the configured normalization method.
        
        Parameters
        ----------
        df : pd.DataFrame
            DataFrame containing feature metrics
        complexity_col : str
            Column name for complexity scores (default: "complexity_score")
        importance_col : str
            Column name for importance scores (default: "importance_score")
        coverage_col : str
            Column name for coverage ratios (default: "coverage_ratio")
        consistency_col : str
            Column name for cross-seed std (default: "cross_seed_std")
        feature_col : str
            Column name for feature names (default: "feature_name")
        
        Returns
        -------
        pd.DataFrame
            DataFrame with reliability scores and component scores added.
            New columns:
            - reliability_score: Overall reliability score
            - reliability_stability: Stability component
            - reliability_importance: Importance component
            - reliability_coverage: Coverage component
            - reliability_consistency: Consistency component
        
        Examples
        --------
        >>> import pandas as pd
        >>> scorer = ReliabilityScorer()
        >>> df = pd.DataFrame({
        ...     "feature_name": ["PAY_AMT1", "BILL_AMT1"],
        ...     "complexity_score": [0.15, 0.20],
        ...     "importance_score": [0.85, 0.75],
        ...     "coverage_ratio": [0.95, 0.90],
        ...     "cross_seed_std": [0.05, 0.08]
        ... })
        >>> result_df = scorer.compute_from_dataframe(df)
        >>> print(result_df[["feature_name", "reliability_score"]])
        """
        df = df.copy()
        
        # Normalize components using series normalization
        stability = self._normalize_series(df[complexity_col], invert=True)
        importance = self._normalize_series(df[importance_col], invert=False)
        coverage = df[coverage_col].fillna(0.5).clip(0, 1)
        consistency = self._normalize_series(df[consistency_col], invert=True)
        
        # Compute weighted average
        df["reliability_score"] = (
            self.config.importance_weight * importance +
            self.config.stability_weight * stability +
            self.config.coverage_weight * coverage +
            self.config.consistency_weight * consistency
        )
        
        # Add component scores for transparency
        df["reliability_stability"] = stability
        df["reliability_importance"] = importance
        df["reliability_coverage"] = coverage
        df["reliability_consistency"] = consistency
        
        return df
    
    def get_formula_documentation(self) -> str:
        """
        Return detailed formula documentation.
        
        Returns
        -------
        str
            Detailed documentation of the reliability score formula.
        """
        doc = f"""
Reliability Score Formula Documentation
======================================

The reliability score is a weighted combination of four components:

1. IMPORTANCE COMPONENT ({self.config.importance_weight:.0%} weight)
   - Measures how important a feature is to the model
   - Formula: importance = normalize(importance_score)
   - Higher importance → Higher reliability
   - Normalization range: [{self.config.importance_min}, {self.config.importance_max}]

2. STABILITY COMPONENT ({self.config.stability_weight:.0%} weight)
   - Measures how stable the feature's behavior is across bootstrap samples
   - Formula: stability = 1 - normalize(complexity_score)
   - Lower complexity → Higher stability → Higher reliability
   - Normalization range: [{self.config.complexity_min}, {self.config.complexity_max}]

3. COVERAGE COMPONENT ({self.config.coverage_weight:.0%} weight)
   - Measures how completely the feature was evaluated
   - Formula: coverage = coverage_ratio (already in [0, 1])
   - Higher coverage → Higher reliability

4. CONSISTENCY COMPONENT ({self.config.consistency_weight:.0%} weight)
   - Measures how consistent the feature is across different random seeds
   - Formula: consistency = 1 - normalize(cross_seed_std)
   - Lower std → Higher consistency → Higher reliability
   - Normalization range: [{self.config.cross_seed_std_min}, {self.config.cross_seed_std_max}]

FINAL SCORE:
   reliability = {self.config.importance_weight:.2f} * importance + 
                 {self.config.stability_weight:.2f} * stability + 
                 {self.config.coverage_weight:.2f} * coverage + 
                 {self.config.consistency_weight:.2f} * consistency

Range: [0, 1] where 1 = most reliable

Normalization Method: {self.config.normalization_method}
"""
        return doc


# Default configuration instance
DEFAULT_RELIABILITY_CONFIG = ReliabilityConfig()


def compute_reliability_score(
    complexity_score: float,
    importance_score: float,
    coverage_ratio: float,
    cross_seed_std: float,
    config: Optional[ReliabilityConfig] = None
) -> float:
    """
    Convenience function to compute reliability score for a single feature.
    
    Parameters
    ----------
    complexity_score : float
        Complexity score from bootstrap stability analysis
    importance_score : float
        Feature importance score
    coverage_ratio : float
        Ratio of non-NaN metric evaluations
    cross_seed_std : float
        Standard deviation across different random seeds
    config : ReliabilityConfig, optional
        Configuration for reliability computation
    
    Returns
    -------
    float
        Reliability score in [0, 1]
    
    Examples
    --------
    >>> score = compute_reliability_score(
    ...     complexity_score=0.15,
    ...     importance_score=0.85,
    ...     coverage_ratio=0.95,
    ...     cross_seed_std=0.05
    ... )
    >>> print(f"Reliability: {score:.3f}")
    """
    scorer = ReliabilityScorer(config)
    result = scorer.compute(
        feature_name="_",
        complexity_score=complexity_score,
        importance_score=importance_score,
        coverage_ratio=coverage_ratio,
        cross_seed_std=cross_seed_std,
    )
    return result.reliability_score
