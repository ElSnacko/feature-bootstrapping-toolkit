"""
Meta-Bootstrap Module for Complexity Score Confidence Intervals

This module provides the MetaBootstrap class that orchestrates multiple stability
analysis runs across different data splits to produce confidence intervals for
complexity scores.

The key insight is that if a feature's complexity score swings significantly between
random splits, the score isn't measuring a stable structural property. This module
quantifies that uncertainty.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Union, Any, Type
import logging

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold, train_test_split
from joblib import Parallel, delayed

logger = logging.getLogger(__name__)


class SplitStrategy(Enum):
    """Strategy for splitting data in meta-bootstrap analysis."""
    KFOLD = "kfold"
    REPEATED_RANDOM = "repeated_random"
    BOOTSTRAP = "bootstrap"


@dataclass
class MetaBootstrapResult:
    """
    Container for meta-bootstrap results for a single feature.
    
    Attributes
    ----------
    feature_name : str
        Name of the feature analyzed.
    mean_complexity : float
        Mean complexity score across all splits.
    std_complexity : float
        Standard deviation of complexity scores.
    ci_lower : float
        Lower bound of 95% confidence interval.
    ci_upper : float
        Upper bound of 95% confidence interval.
    all_scores : List[float]
        All complexity scores from each split.
    n_splits : int
        Number of splits performed.
    split_strategy : str
        Strategy used for splitting (kfold, repeated_random, bootstrap).
    complexity_scores_by_category : Dict[str, Dict]
        Per-category statistics (target_agnostic, target_dependent).
    split_results : List[Dict]
        Raw results from each split (optional, for debugging).
    """
    feature_name: str
    mean_complexity: float
    std_complexity: float
    ci_lower: float
    ci_upper: float
    all_scores: List[float]
    n_splits: int
    split_strategy: str
    complexity_scores_by_category: Dict[str, Dict] = field(default_factory=dict)
    split_results: List[Dict] = field(default_factory=list)
    
    def summary(self) -> Dict[str, Any]:
        """Return summary dictionary."""
        return {
            'feature': self.feature_name,
            'mean_complexity': self.mean_complexity,
            'std_complexity': self.std_complexity,
            'ci_lower': self.ci_lower,
            'ci_upper': self.ci_upper,
            'ci_width': self.ci_upper - self.ci_lower,
            'n_splits': self.n_splits,
            'split_strategy': self.split_strategy,
        }


def _compute_ci_t_distribution(scores: np.ndarray, ci_level: float = 0.95) -> tuple:
    """
    Compute confidence interval using t-distribution.
    
    Parameters
    ----------
    scores : np.ndarray
        Array of scores.
    ci_level : float
        Confidence level (default 0.95 for 95% CI).
    
    Returns
    -------
    tuple
        (ci_lower, ci_upper)
    """
    n = len(scores)
    if n < 2:
        return float(scores[0]), float(scores[0])
    
    mean = np.mean(scores)
    std = np.std(scores, ddof=1)
    se = std / np.sqrt(n)
    
    # t-critical value for two-tailed CI
    alpha = 1 - ci_level
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    
    ci_lower = mean - t_crit * se
    ci_upper = mean + t_crit * se
    
    return ci_lower, ci_upper


def _compute_category_stats(
    category_scores: Dict[str, List[float]],
    ci_level: float = 0.95
) -> Dict[str, Dict]:
    """
    Compute statistics for each metric category.
    
    Parameters
    ----------
    category_scores : Dict[str, List[float]]
        Dictionary mapping category names to lists of scores.
    ci_level : float
        Confidence level.
    
    Returns
    -------
    Dict[str, Dict]
        Per-category statistics with mean, std, ci_lower, ci_upper.
    """
    result = {}
    for category, scores in category_scores.items():
        scores_arr = np.array(scores)
        if len(scores_arr) > 0:
            ci_lower, ci_upper = _compute_ci_t_distribution(scores_arr, ci_level)
            result[category] = {
                'mean': float(np.mean(scores_arr)),
                'std': float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'n': len(scores_arr),
            }
    return result


class MetaBootstrap:
    """
    Orchestrates multiple stability analysis runs to produce confidence intervals.
    
    This class wraps BootstrapStability or SHAPStability and runs the analysis
    across multiple data splits to quantify uncertainty in complexity scores.
    
    Parameters
    ----------
    n_splits : int, default=10
        Number of data splits (K for K-fold, or N for repeated random/bootstrap).
    strategy : SplitStrategy or str, default=SplitStrategy.REPEATED_RANDOM
        Strategy for splitting data:
        - 'kfold': K-fold cross-validation
        - 'repeated_random': Repeated random train/test splits
        - 'bootstrap': Bootstrap sampling with replacement
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs (-1 uses all cores).
    ci_level : float, default=0.95
        Confidence interval level (0.95 for 95% CI).
    train_frac : float, default=0.8
        Fraction of data for training in repeated_random strategy.
    
    Examples
    --------
    >>> from bootstrap_stability import MetaBootstrap, BootstrapStability
    >>> meta = MetaBootstrap(n_splits=10, strategy='repeated_random')
    >>> results = meta.fit(X, y, analyzer_class=BootstrapStability)
    >>> results['PAY_AMT1'].mean_complexity
    -150.0
    >>> results['PAY_AMT1'].ci_lower
    -230.0
    >>> report = meta.get_stability_report()
    """
    
    def __init__(
        self,
        n_splits: int = 10,
        strategy: Union[SplitStrategy, str] = SplitStrategy.REPEATED_RANDOM,
        random_state: int = 42,
        n_jobs: int = -1,
        ci_level: float = 0.95,
        train_frac: float = 0.8,
    ):
        self.n_splits = n_splits
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.ci_level = ci_level
        self.train_frac = train_frac
        
        # Handle string or enum strategy
        if isinstance(strategy, str):
            strategy_map = {
                'kfold': SplitStrategy.KFOLD,
                'repeated_random': SplitStrategy.REPEATED_RANDOM,
                'bootstrap': SplitStrategy.BOOTSTRAP,
            }
            self.strategy = strategy_map[strategy.lower()]
        else:
            self.strategy = strategy
        
        # Results storage
        self._results: Dict[str, MetaBootstrapResult] = {}
        self._feature_names: List[str] = []
    
    def _get_kfold_splits(self, X: pd.DataFrame) -> List[tuple]:
        """Generate K-fold split indices."""
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        return list(kf.split(X))
    
    def _get_repeated_random_splits(self, X: pd.DataFrame) -> List[tuple]:
        """Generate repeated random train/test split indices."""
        splits = []
        rng = np.random.RandomState(self.random_state)
        
        for i in range(self.n_splits):
            seed = rng.randint(0, 2**31 - 1)
            train_idx, test_idx = train_test_split(
                np.arange(len(X)),
                train_size=self.train_frac,
                random_state=seed
            )
            splits.append((train_idx, test_idx))
        
        return splits
    
    def _get_bootstrap_splits(self, X: pd.DataFrame) -> List[tuple]:
        """Generate bootstrap sample indices (sample with replacement)."""
        splits = []
        rng = np.random.RandomState(self.random_state)
        n_samples = len(X)
        
        for i in range(self.n_splits):
            # Bootstrap sample: sample with replacement to same size
            indices = rng.choice(n_samples, size=n_samples, replace=True)
            splits.append((indices, None))  # No test set for bootstrap
        
        return splits
    
    def _get_splits(self, X: pd.DataFrame) -> List[tuple]:
        """Get split indices based on strategy."""
        if self.strategy == SplitStrategy.KFOLD:
            return self._get_kfold_splits(X)
        elif self.strategy == SplitStrategy.REPEATED_RANDOM:
            return self._get_repeated_random_splits(X)
        elif self.strategy == SplitStrategy.BOOTSTRAP:
            return self._get_bootstrap_splits(X)
        else:
            raise ValueError(f"Unknown split strategy: {self.strategy}")
    
    def _run_single_split(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series],
        train_idx: np.ndarray,
        analyzer_class: Type,
        analyzer_kwargs: Dict[str, Any],
        feature_col: Optional[str] = None,
        target_col: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Run stability analysis on a single split.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.Series, optional
            Target series.
        train_idx : np.ndarray
            Indices for training data.
        analyzer_class : Type
            BootstrapStability or SHAPStability class.
        analyzer_kwargs : dict
            Additional arguments for the analyzer.
        feature_col : str, optional
            Feature column name (for BootstrapStability).
        target_col : str, optional
            Target column name (for BootstrapStability).
        
        Returns
        -------
        Dict[str, Any]
            Results from the analyzer.
        """
        # Create analyzer instance
        analyzer = analyzer_class(**analyzer_kwargs)
        
        # Prepare data for this split
        X_split = X.iloc[train_idx].reset_index(drop=True)
        
        if y is not None:
            y_split = y.iloc[train_idx].reset_index(drop=True)
            df_split = X_split.copy()
            if target_col:
                df_split[target_col] = y_split.values
        else:
            df_split = X_split
            y_split = None
        
        # Run analysis based on analyzer type
        class_name = analyzer_class.__name__
        
        if class_name == 'BootstrapStability':
            # For BootstrapStability, we fit one feature at a time
            if feature_col:
                result = analyzer.fit(df_split, feature_col=feature_col, target_col=target_col)
                return result
            else:
                # Fit all features (panel mode)
                feature_cols = [c for c in X.columns if c != target_col]
                results = {}
                for feat in feature_cols:
                    results[feat] = analyzer.fit(df_split, feature_col=feat, target_col=target_col)
                return results
        
        elif class_name == 'SHAPStability':
            # For SHAPStability, pass X and y directly
            result = analyzer.fit(X_split, y_split)
            return result
        
        else:
            raise ValueError(f"Unsupported analyzer class: {class_name}")
    
    def fit(
        self,
        X: pd.DataFrame,
        y: Optional[pd.Series] = None,
        analyzer_class: Type = None,
        feature_col: Optional[str] = None,
        target_col: Optional[str] = None,
        **analyzer_kwargs
    ) -> Dict[str, MetaBootstrapResult]:
        """
        Run stability analysis across multiple splits.
        
        Parameters
        ----------
        X : pd.DataFrame
            Feature dataframe.
        y : pd.Series, optional
            Target series.
        analyzer_class : Type
            BootstrapStability or SHAPStability class (not instance).
        feature_col : str, optional
            Single feature to analyze (for BootstrapStability).
            If None, analyzes all features.
        target_col : str, optional
            Target column name (for BootstrapStability).
        **analyzer_kwargs
            Additional arguments passed to the analyzer constructor.
        
        Returns
        -------
        Dict[str, MetaBootstrapResult]
            Dictionary mapping feature names to their meta-bootstrap results.
        
        Examples
        --------
        >>> from bootstrap_stability import MetaBootstrap, BootstrapStability
        >>> meta = MetaBootstrap(n_splits=10)
        >>> results = meta.fit(X, y, analyzer_class=BootstrapStability)
        """
        if analyzer_class is None:
            # Import here to avoid circular imports
            from .analyzer import BootstrapStability
            analyzer_class = BootstrapStability
        
        # Get splits
        splits = self._get_splits(X)
        self._feature_names = [feature_col] if feature_col else list(X.columns)
        
        # Determine if we're analyzing single feature or panel
        single_feature = feature_col is not None
        
        # Run splits in parallel
        all_split_results = Parallel(n_jobs=self.n_jobs)(
            delayed(self._run_single_split)(
                X, y, train_idx, analyzer_class, analyzer_kwargs,
                feature_col=feature_col, target_col=target_col
            )
            for train_idx, _ in splits
        )
        
        # Aggregate results by feature
        results = {}
        
        if single_feature:
            # Single feature mode
            feature_scores = []
            category_scores = {'target_agnostic': [], 'target_dependent': []}
            split_results = []
            
            for split_result in all_split_results:
                if split_result and 'complexity_score' in split_result:
                    feature_scores.append(split_result['complexity_score'])
                    split_results.append(split_result)
                    
                    # Extract per-category scores
                    if 'complexity_by_category' in split_result:
                        for cat, data in split_result['complexity_by_category'].items():
                            if cat in category_scores and 'floor' in data:
                                category_scores[cat].append(data['floor'])
            
            if feature_scores:
                scores_arr = np.array(feature_scores)
                ci_lower, ci_upper = _compute_ci_t_distribution(scores_arr, self.ci_level)
                category_stats = _compute_category_stats(category_scores, self.ci_level)
                
                results[feature_col] = MetaBootstrapResult(
                    feature_name=feature_col,
                    mean_complexity=float(np.mean(scores_arr)),
                    std_complexity=float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
                    ci_lower=ci_lower,
                    ci_upper=ci_upper,
                    all_scores=feature_scores,
                    n_splits=len(feature_scores),
                    split_strategy=self.strategy.value,
                    complexity_scores_by_category=category_stats,
                    split_results=split_results,
                )
        else:
            # Panel mode - aggregate across features
            feature_scores_dict: Dict[str, List[float]] = {feat: [] for feat in self._feature_names}
            category_scores_dict: Dict[str, Dict[str, List[float]]] = {
                feat: {'target_agnostic': [], 'target_dependent': []}
                for feat in self._feature_names
            }
            split_results_dict: Dict[str, List[Dict]] = {
                feat: [] for feat in self._feature_names
            }
            
            for split_result in all_split_results:
                if isinstance(split_result, dict):
                    for feat, feat_result in split_result.items():
                        if feat in feature_scores_dict and isinstance(feat_result, dict):
                            if 'complexity_score' in feat_result:
                                feature_scores_dict[feat].append(feat_result['complexity_score'])
                                split_results_dict[feat].append(feat_result)
                                
                                # Extract per-category scores
                                if 'complexity_by_category' in feat_result:
                                    for cat, data in feat_result['complexity_by_category'].items():
                                        if cat in category_scores_dict[feat] and 'floor' in data:
                                            category_scores_dict[feat][cat].append(data['floor'])
            
            # Build results for each feature
            for feat in self._feature_names:
                scores = feature_scores_dict[feat]
                if scores:
                    scores_arr = np.array(scores)
                    ci_lower, ci_upper = _compute_ci_t_distribution(scores_arr, self.ci_level)
                    category_stats = _compute_category_stats(category_scores_dict[feat], self.ci_level)
                    
                    results[feat] = MetaBootstrapResult(
                        feature_name=feat,
                        mean_complexity=float(np.mean(scores_arr)),
                        std_complexity=float(np.std(scores_arr, ddof=1)) if len(scores_arr) > 1 else 0.0,
                        ci_lower=ci_lower,
                        ci_upper=ci_upper,
                        all_scores=scores,
                        n_splits=len(scores),
                        split_strategy=self.strategy.value,
                        complexity_scores_by_category=category_stats,
                        split_results=split_results_dict[feat],
                    )
        
        self._results = results
        return results
    
    def get_stability_report(self) -> pd.DataFrame:
        """
        Returns DataFrame with mean, std, CI for all features.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with columns: feature, mean, std, ci_lower, ci_upper, n_splits
        
        Examples
        --------
        >>> meta = MetaBootstrap(n_splits=10)
        >>> results = meta.fit(X, y, analyzer_class=BootstrapStability)
        >>> report = meta.get_stability_report()
        >>> print(report)
               feature   mean    std  ci_lower  ci_upper  n_splits
        0     PAY_AMT1 -150.0  120.0    -230.0     -70.0        10
        1    BILL_AMT1  -80.0   45.0    -110.0     -50.0        10
        """
        if not self._results:
            raise ValueError("No results available. Call fit() first.")
        
        rows = []
        for feat, result in self._results.items():
            rows.append({
                'feature': feat,
                'mean': result.mean_complexity,
                'std': result.std_complexity,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'n_splits': result.n_splits,
            })
        
        df = pd.DataFrame(rows)
        df = df.sort_values('mean').reset_index(drop=True)
        return df
    
    def get_detailed_report(self) -> pd.DataFrame:
        """
        Returns detailed DataFrame including per-category statistics.
        
        Returns
        -------
        pd.DataFrame
            DataFrame with per-category mean, std, CI columns.
        """
        if not self._results:
            raise ValueError("No results available. Call fit() first.")
        
        rows = []
        for feat, result in self._results.items():
            row = {
                'feature': feat,
                'mean_complexity': result.mean_complexity,
                'std_complexity': result.std_complexity,
                'ci_lower': result.ci_lower,
                'ci_upper': result.ci_upper,
                'ci_width': result.ci_upper - result.ci_lower,
                'n_splits': result.n_splits,
                'split_strategy': result.split_strategy,
            }
            
            # Add per-category stats
            for cat, stats_dict in result.complexity_scores_by_category.items():
                row[f'{cat}_mean'] = stats_dict.get('mean', np.nan)
                row[f'{cat}_std'] = stats_dict.get('std', np.nan)
                row[f'{cat}_ci_lower'] = stats_dict.get('ci_lower', np.nan)
                row[f'{cat}_ci_upper'] = stats_dict.get('ci_upper', np.nan)
            
            rows.append(row)
        
        df = pd.DataFrame(rows)
        df = df.sort_values('mean_complexity').reset_index(drop=True)
        return df
    
    @property
    def results(self) -> Dict[str, MetaBootstrapResult]:
        """Access the results dictionary."""
        return self._results
