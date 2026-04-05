"""
Train/OOT SHAP Stability Module

This module provides tools for measuring SHAP stability between train and
out-of-time (OOT) periods. This is the key production stability question:
do features contribute to predictions the same way in the OOT period as
they did in training?
"""

import numpy as np
import pandas as pd
from datetime import datetime
from typing import Callable, Dict, List, Optional, Union, Any
from dataclasses import dataclass

from .core import VERSION


@dataclass
class DriftResult:
    """Container for drift metric results."""
    name: str
    train_value: float
    oot_value: float
    drift: float
    threshold: float
    flagged: bool
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "train_value": float(self.train_value),
            "oot_value": float(self.oot_value),
            "drift": float(self.drift),
            "threshold": float(self.threshold),
            "flagged": bool(self.flagged),
        }


def _get_explainer(model, explainer_type: str, explainer_kwargs: dict = None):
    """Get SHAP explainer for the model."""
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required. Install with: pip install shap")
    
    kwargs = explainer_kwargs or {}
    
    if explainer_type == "tree":
        return shap.TreeExplainer(model, **kwargs)
    elif explainer_type == "kernel":
        return shap.KernelExplainer(model.predict, **kwargs)
    elif explainer_type == "linear":
        return shap.LinearExplainer(model, **kwargs)
    elif explainer_type == "deep":
        return shap.DeepExplainer(model, **kwargs)
    elif explainer_type == "auto":
        return shap.Explainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


def _compute_shap_values(
    model,
    X: np.ndarray,
    explainer_type: str,
    explainer_kwargs: dict = None,
    subsample: int = None,
    random_state: int = None,
) -> np.ndarray:
    """Compute SHAP values for a model and dataset."""
    # Subsample if requested
    if subsample is not None and len(X) > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X
    
    # Get explainer and compute SHAP values
    explainer = _get_explainer(model, explainer_type, explainer_kwargs)
    shap_values = explainer.shap_values(X_eval)
    
    # Handle list output (binary classification)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
    
    return np.array(shap_values)


def compute_shap_rank_correlation(
    train_shap: np.ndarray,
    oot_shap: np.ndarray,
) -> float:
    """
    Compute Spearman rank correlation of feature importance between train and OOT.
    
    Parameters
    ----------
    train_shap : np.ndarray
        SHAP values from train set. Shape: (n_samples, n_features)
    oot_shap : np.ndarray
        SHAP values from OOT set. Shape: (n_samples, n_features)
    
    Returns
    -------
    float
        Spearman rank correlation. 1.0 = perfect agreement, -1.0 = opposite.
    """
    from scipy.stats import spearmanr
    
    # Compute mean |SHAP| per feature
    train_importance = np.abs(train_shap).mean(axis=0)
    oot_importance = np.abs(oot_shap).mean(axis=0)
    
    # Compute ranks
    train_ranks = np.argsort(np.argsort(-train_importance))
    oot_ranks = np.argsort(np.argsort(-oot_importance))
    
    # Spearman correlation
    corr, _ = spearmanr(train_ranks, oot_ranks)
    return float(corr)


def compute_direction_flip_rate(
    train_shap: np.ndarray,
    oot_shap: np.ndarray,
) -> float:
    """
    Compute fraction of samples where SHAP sign flips between train and OOT.
    
    For samples that appear in both datasets, compute the fraction where
    the SHAP sign differs.
    
    Parameters
    ----------
    train_shap : np.ndarray
        SHAP values from train set.
    oot_shap : np.ndarray
        SHAP values from OOT set.
    
    Returns
    -------
    float
        Direction flip rate. 0.0 = no flips, 1.0 = all flipped.
    """
    # Compute mean SHAP sign per feature
    train_sign = np.sign(train_shap.mean(axis=0))
    oot_sign = np.sign(oot_shap.mean(axis=0))
    
    # Handle zeros
    train_sign = np.where(train_sign == 0, 1, train_sign)
    oot_sign = np.where(oot_sign == 0, 1, oot_sign)
    
    # Count flips
    flips = (train_sign != oot_sign).sum()
    total = len(train_sign)
    
    return float(flips / total) if total > 0 else 0.0


def compute_magnitude_drift(
    train_shap: np.ndarray,
    oot_shap: np.ndarray,
) -> float:
    """
    Compute mean change in |SHAP| magnitude between train and OOT.
    
    Parameters
    ----------
    train_shap : np.ndarray
        SHAP values from train set.
    oot_shap : np.ndarray
        SHAP values from OOT set.
    
    Returns
    -------
    float
        Mean magnitude drift. 0.0 = no change.
    """
    # Compute mean |SHAP| per feature
    train_mag = np.abs(train_shap).mean(axis=0)
    oot_mag = np.abs(oot_shap).mean(axis=0)
    
    # Compute relative change
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = np.abs(oot_mag - train_mag) / (train_mag + 1e-10)
        relative_change = np.where(np.isfinite(relative_change), relative_change, 0)
    
    return float(np.mean(relative_change))


def compute_topk_overlap(
    train_shap: np.ndarray,
    oot_shap: np.ndarray,
    k: int = 10,
) -> float:
    """
    Compute Jaccard overlap of top-k features between train and OOT.
    
    Parameters
    ----------
    train_shap : np.ndarray
        SHAP values from train set.
    oot_shap : np.ndarray
        SHAP values from OOT set.
    k : int
        Number of top features to consider.
    
    Returns
    -------
    float
        Jaccard overlap. 1.0 = perfect overlap, 0.0 = no overlap.
    """
    n_features = train_shap.shape[1]
    k = min(k, n_features)
    
    # Compute mean |SHAP| per feature
    train_importance = np.abs(train_shap).mean(axis=0)
    oot_importance = np.abs(oot_shap).mean(axis=0)
    
    # Get top-k indices
    train_topk = set(np.argsort(-train_importance)[:k])
    oot_topk = set(np.argsort(-oot_importance)[:k])
    
    # Jaccard overlap
    intersection = len(train_topk & oot_topk)
    union = len(train_topk | oot_topk)
    
    return float(intersection / union) if union > 0 else 0.0


def compute_per_feature_drift(
    train_shap: np.ndarray,
    oot_shap: np.ndarray,
    feature_names: List[str],
) -> Dict[str, Dict]:
    """
    Compute per-feature drift metrics.
    
    Parameters
    ----------
    train_shap : np.ndarray
        SHAP values from train set.
    oot_shap : np.ndarray
        SHAP values from OOT set.
    feature_names : List[str]
        Feature names.
    
    Returns
    -------
    Dict[str, Dict]
        Per-feature drift metrics.
    """
    n_features = len(feature_names)
    
    # Compute mean |SHAP| per feature
    train_importance = np.abs(train_shap).mean(axis=0)
    oot_importance = np.abs(oot_shap).mean(axis=0)
    
    # Compute ranks
    train_ranks = np.argsort(np.argsort(-train_importance)) + 1
    oot_ranks = np.argsort(np.argsort(-oot_importance)) + 1
    
    # Compute signs
    train_sign = np.sign(train_shap.mean(axis=0))
    oot_sign = np.sign(oot_shap.mean(axis=0))
    
    # Per-feature metrics
    feature_drift = {}
    for f, fname in enumerate(feature_names):
        # Rank change
        rank_change = int(train_ranks[f] - oot_ranks[f])
        
        # Magnitude ratio
        train_mag = train_importance[f]
        oot_mag = oot_importance[f]
        magnitude_ratio = float(oot_mag / (train_mag + 1e-10))
        
        # Direction consistency
        train_s = train_sign[f] if train_sign[f] != 0 else 1
        oot_s = oot_sign[f] if oot_sign[f] != 0 else 1
        direction_consistent = int(train_s == oot_s)
        
        # Wasserstein distance
        from scipy.stats import wasserstein_distance
        wasserstein = float(wasserstein_distance(train_shap[:, f], oot_shap[:, f]))
        
        # Composite drift score
        drift_score = 0.0
        if not direction_consistent:
            drift_score += 0.4  # Direction flip is serious
        drift_score += 0.3 * min(abs(rank_change) / n_features, 1.0)
        drift_score += 0.3 * min(abs(magnitude_ratio - 1) / 2, 1.0)
        
        feature_drift[fname] = {
            "rank_train": int(train_ranks[f]),
            "rank_oot": int(oot_ranks[f]),
            "rank_change": rank_change,
            "magnitude_train": float(train_mag),
            "magnitude_oot": float(oot_mag),
            "magnitude_ratio": magnitude_ratio,
            "direction_consistent": bool(direction_consistent),
            "wasserstein": wasserstein,
            "drift_score": float(drift_score),
        }
    
    return feature_drift


def compute_overall_drift_score(
    rank_correlation: float,
    direction_flip_rate: float,
    topk_overlap: float,
    magnitude_drift: float,
    weights: dict = None,
) -> float:
    """
    Compute overall drift score from train to OOT.
    
    Parameters
    ----------
    rank_correlation : float
        Spearman rank correlation of feature importance.
    direction_flip_rate : float
        Fraction of features with sign flip.
    topk_overlap : float
        Jaccard overlap of top-k features.
    magnitude_drift : float
        Mean relative magnitude change.
    weights : dict, optional
        Weights for each component.
    
    Returns
    -------
    float
        Overall drift score in [0, 1]. 0 = no drift, 1 = maximum drift.
    """
    if weights is None:
        weights = {
            "rank_correlation": 0.30,
            "direction_flip_rate": 0.30,
            "topk_overlap": 0.20,
            "magnitude_drift": 0.20,
        }
    
    # Convert all to drift (higher = more drift)
    rank_drift = 1 - max(rank_correlation, 0)  # Correlation to drift
    direction_drift = direction_flip_rate  # Already a drift metric
    topk_drift = 1 - topk_overlap  # Overlap to drift
    magnitude_drift_clipped = min(magnitude_drift, 1.0)  # Clip to [0, 1]
    
    # Weighted average
    total_weight = sum(weights.values())
    drift_score = (
        weights["rank_correlation"] * rank_drift +
        weights["direction_flip_rate"] * direction_drift +
        weights["topk_overlap"] * topk_drift +
        weights["magnitude_drift"] * magnitude_drift_clipped
    ) / total_weight
    
    return float(np.clip(drift_score, 0, 1))


def get_drift_grade(drift_score: float) -> str:
    """
    Convert drift score to letter grade.
    
    Parameters
    ----------
    drift_score : float
        Overall drift score in [0, 1].
    
    Returns
    -------
    str
        Letter grade: A, B, C, D, or F.
    """
    if drift_score < 0.10:
        return "A"
    elif drift_score < 0.25:
        return "B"
    elif drift_score < 0.40:
        return "C"
    elif drift_score < 0.60:
        return "D"
    else:
        return "F"


class TrainOOTStability:
    """
    Measure SHAP stability between train and out-of-time periods.
    
    This is the key production stability question: do features contribute
    to predictions the same way in the OOT period as in training?
    
    Parameters
    ----------
    model_factory : Callable
        Function that returns an unfitted model instance.
    explainer_type : str, default='tree'
        Type of SHAP explainer.
    explainer_kwargs : dict, optional
        Additional arguments for explainer.
    shap_subsample : int, optional
        Subsample for SHAP computation.
    top_k : int, default=10
        Number of top features for overlap computation.
    random_state : int, default=42
        Random seed.
    verbose : int, default=1
        Verbosity level.
    
    Examples
    --------
    >>> from bootstrap_stability import TrainOOTStability
    >>> 
    >>> oot_stability = TrainOOTStability(model_factory=create_model)
    >>> results = oot_stability.fit(X_train, y_train, X_oot, y_oot)
    >>> print(f"Drift grade: {results['drift_grade']}")
    """
    
    def __init__(
        self,
        model_factory: Callable[[], Any],
        explainer_type: str = 'tree',
        explainer_kwargs: dict = None,
        shap_subsample: int = None,
        top_k: int = 10,
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.model_factory = model_factory
        self.explainer_type = explainer_type
        self.explainer_kwargs = explainer_kwargs or {}
        self.shap_subsample = shap_subsample
        self.top_k = top_k
        self.random_state = random_state
        self.verbose = verbose
    
    def _print(self, msg: str, level: int = 1):
        """Print message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(msg)
    
    def fit(
        self,
        X_train: Union[pd.DataFrame, np.ndarray],
        y_train: Union[pd.Series, np.ndarray],
        X_oot: Union[pd.DataFrame, np.ndarray],
        y_oot: Union[pd.Series, np.ndarray] = None,
        feature_names: list = None,
    ) -> dict:
        """
        Compare SHAP patterns between train and OOT.
        
        Parameters
        ----------
        X_train : pd.DataFrame or np.ndarray
            Training features.
        y_train : pd.Series or np.ndarray
            Training target.
        X_oot : pd.DataFrame or np.ndarray
            OOT features.
        y_oot : pd.Series or np.ndarray, optional
            OOT target (not used for SHAP, only for metadata).
        feature_names : list, optional
            Feature names.
        
        Returns
        -------
        dict
            Results with keys:
            - 'train_shap_summary': SHAP statistics on train
            - 'oot_shap_summary': SHAP statistics on OOT
            - 'drift_metrics': Metrics comparing train vs OOT
            - 'feature_drift': Per-feature drift metrics
            - 'overall_drift_score': Composite drift score
            - 'drift_grade': Letter grade (A-F)
        """
        # Prepare data
        if isinstance(X_train, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X_train.columns)
            X_train = X_train.values
        elif isinstance(X_train, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X_train.shape[1])]
        
        if isinstance(X_oot, pd.DataFrame):
            X_oot = X_oot.values
        
        if isinstance(y_train, pd.Series):
            y_train = y_train.values
        if y_oot is not None and isinstance(y_oot, pd.Series):
            y_oot = y_oot.values
        
        n_train = len(X_train)
        n_oot = len(X_oot)
        n_features = len(feature_names)
        
        self._print(f"Train/OOT SHAP Stability Analysis")
        self._print(f"  Train: {n_train} samples")
        self._print(f"  OOT: {n_oot} samples")
        self._print(f"  Features: {n_features}")
        
        # Train model on train data
        self._print("Training model on train data...")
        model = self.model_factory()
        model.fit(X_train, y_train)
        
        # Compute SHAP values
        self._print("Computing SHAP values for train data...")
        train_shap = _compute_shap_values(
            model, X_train, self.explainer_type, self.explainer_kwargs,
            subsample=self.shap_subsample, random_state=self.random_state
        )
        
        self._print("Computing SHAP values for OOT data...")
        oot_shap = _compute_shap_values(
            model, X_oot, self.explainer_type, self.explainer_kwargs,
            subsample=self.shap_subsample, random_state=self.random_state + 1000
        )
        
        # Compute SHAP summaries
        train_shap_summary = self._compute_shap_summary(train_shap, feature_names)
        oot_shap_summary = self._compute_shap_summary(oot_shap, feature_names)
        
        # Compute drift metrics
        self._print("Computing drift metrics...")
        rank_correlation = compute_shap_rank_correlation(train_shap, oot_shap)
        direction_flip_rate = compute_direction_flip_rate(train_shap, oot_shap)
        magnitude_drift = compute_magnitude_drift(train_shap, oot_shap)
        topk_overlap = compute_topk_overlap(train_shap, oot_shap, k=self.top_k)
        
        drift_metrics = {
            "rank_correlation": DriftResult(
                name="rank_correlation",
                train_value=rank_correlation,
                oot_value=rank_correlation,  # Same metric
                drift=1 - rank_correlation,
                threshold=0.8,
                flagged=rank_correlation < 0.8,
            ),
            "direction_flip_rate": DriftResult(
                name="direction_flip_rate",
                train_value=0.0,  # Not applicable per-period
                oot_value=direction_flip_rate,
                drift=direction_flip_rate,
                threshold=0.15,
                flagged=direction_flip_rate > 0.15,
            ),
            "magnitude_drift": DriftResult(
                name="magnitude_drift",
                train_value=0.0,
                oot_value=magnitude_drift,
                drift=magnitude_drift,
                threshold=0.30,
                flagged=magnitude_drift > 0.30,
            ),
            "topk_overlap": DriftResult(
                name="topk_overlap",
                train_value=1.0,
                oot_value=topk_overlap,
                drift=1 - topk_overlap,
                threshold=0.7,
                flagged=topk_overlap < 0.7,
            ),
        }
        
        # Compute per-feature drift
        feature_drift = compute_per_feature_drift(train_shap, oot_shap, feature_names)
        
        # Compute overall drift score
        overall_drift_score = compute_overall_drift_score(
            rank_correlation=rank_correlation,
            direction_flip_rate=direction_flip_rate,
            topk_overlap=topk_overlap,
            magnitude_drift=magnitude_drift,
        )
        
        drift_grade = get_drift_grade(overall_drift_score)
        
        # Identify drifted features
        drifted_features = [
            fname for fname, metrics in feature_drift.items()
            if metrics["drift_score"] > 0.25
        ]
        
        self._print(f"\nDrift Summary:")
        self._print(f"  Overall drift score: {overall_drift_score:.3f}")
        self._print(f"  Drift grade: {drift_grade}")
        self._print(f"  Rank correlation: {rank_correlation:.3f}")
        self._print(f"  Direction flip rate: {direction_flip_rate:.3f}")
        self._print(f"  Top-{self.top_k} overlap: {topk_overlap:.3f}")
        if drifted_features:
            self._print(f"  Features with significant drift: {drifted_features}")
        
        results = {
            "meta": {
                "n_train": n_train,
                "n_oot": n_oot,
                "n_features": n_features,
                "feature_names": feature_names,
                "explainer_type": self.explainer_type,
                "top_k": self.top_k,
                "version": VERSION,
                "random_state": self.random_state,
                "run_timestamp": datetime.utcnow().isoformat(),
            },
            "train_shap_summary": train_shap_summary,
            "oot_shap_summary": oot_shap_summary,
            "drift_metrics": {k: v.to_dict() for k, v in drift_metrics.items()},
            "feature_drift": feature_drift,
            "overall_drift_score": float(overall_drift_score),
            "drift_grade": drift_grade,
            "drifted_features": drifted_features,
        }
        
        return results
    
    def _compute_shap_summary(
        self,
        shap_values: np.ndarray,
        feature_names: List[str],
    ) -> dict:
        """Compute summary statistics for SHAP values."""
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        std_abs_shap = np.abs(shap_values).std(axis=0)
        mean_shap = shap_values.mean(axis=0)
        
        # Rankings
        ranks = np.argsort(np.argsort(-mean_abs_shap)) + 1
        
        # Direction positive rate
        direction_positive = (shap_values > 0).mean(axis=0)
        
        return {
            "mean_abs_shap": {
                fname: float(mean_abs_shap[f])
                for f, fname in enumerate(feature_names)
            },
            "std_abs_shap": {
                fname: float(std_abs_shap[f])
                for f, fname in enumerate(feature_names)
            },
            "mean_shap": {
                fname: float(mean_shap[f])
                for f, fname in enumerate(feature_names)
            },
            "rankings": {
                fname: int(ranks[f])
                for f, fname in enumerate(feature_names)
            },
            "direction_positive_rate": {
                fname: float(direction_positive[f])
                for f, fname in enumerate(feature_names)
            },
        }


def print_oot_report(results: dict) -> None:
    """
    Print a formatted report of train/OOT stability results.
    
    Parameters
    ----------
    results : dict
        Results from TrainOOTStability.fit().
    """
    print("\n" + "=" * 80)
    print("TRAIN/OOT SHAP STABILITY REPORT")
    print("=" * 80)
    
    meta = results["meta"]
    print(f"\nDataset: {meta['n_train']} train samples, {meta['n_oot']} OOT samples")
    print(f"Features: {meta['n_features']}")
    
    print(f"\n{'='*40}")
    print(f"OVERALL DRIFT SCORE: {results['overall_drift_score']:.3f}")
    print(f"DRIFT GRADE: {results['drift_grade']}")
    print(f"{'='*40}")
    
    print("\nDRIFT METRICS:")
    print("-" * 60)
    for metric_name, metric in results["drift_metrics"].items():
        flag = " [!]" if metric["flagged"] else ""
        print(f"  {metric_name:<25} {metric['drift']:.3f}{flag}")
    
    print("\nTOP FEATURES BY DRIFT:")
    print("-" * 60)
    feature_drift = results["feature_drift"]
    sorted_features = sorted(
        feature_drift.items(),
        key=lambda x: x[1]["drift_score"],
        reverse=True
    )
    
    for fname, metrics in sorted_features[:10]:
        drift = metrics["drift_score"]
        rank_chg = metrics["rank_change"]
        dir_cons = "✓" if metrics["direction_consistent"] else "✗"
        print(f"  {fname:<30} drift={drift:.3f}  rank_chg={rank_chg:+3d}  dir={dir_cons}")
    
    if results["drifted_features"]:
        print(f"\n[!] Features with significant drift: {results['drifted_features']}")
    
    print("\n" + "=" * 80)
