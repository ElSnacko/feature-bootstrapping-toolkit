"""
SHAP Stability Metrics Module

This module provides metrics for measuring stability of SHAP-based feature
contributions across bootstrap resamples. Unlike marginal stability which
measures if feature distributions stabilize, these metrics measure if
feature contributions to model decisions stabilize.

Core Metrics:
- Rank Stability: Stability of feature importance rankings
- Wasserstein Distance: Distributional distance of SHAP values
- Direction Consistency: Fraction of samples with consistent SHAP sign
- Magnitude CV: Coefficient of variation of |SHAP| values
"""

import numpy as np
from scipy import stats
from scipy.spatial.distance import jensenshannon
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class SHAPMetricResult:
    """Container for a single SHAP stability metric result."""
    name: str
    values: np.ndarray  # Per-resample values
    mean: float
    stderr: float
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "mean": float(self.mean),
            "stderr": float(self.stderr),
            "values": self.values.tolist() if self.values is not None else [],
        }


def compute_rank_stability(shap_values_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute SHAP rank stability across resamples using Kendall's W.
    
    For each resample, compute feature importance rankings based on mean |SHAP|.
    Then compute Kendall's coefficient of concordance (W) across all resamples.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays, each with shape (n_samples, n_features).
        Each array corresponds to a different bootstrap resample.
    
    Returns
    -------
    np.ndarray
        Rank stability score per feature. Values range from 0 (unstable) to 1 (stable).
        Actually returns 1 - W_instability for each feature's rank consistency.
    
    Notes
    -----
    Kendall's W measures agreement among rankings:
    - W = 1: Perfect concordance (all rankings identical)
    - W = 0: No concordance (rankings random)
    
    We convert to instability: RankInstability = 1 - W
    """
    if len(shap_values_list) < 2:
        return np.array([])
    
    R = len(shap_values_list)  # Number of resamples
    M = shap_values_list[0].shape[1]  # Number of features
    
    # Compute mean |SHAP| per feature per resample
    mean_abs_shap = np.zeros((R, M))
    for r, shap_vals in enumerate(shap_values_list):
        mean_abs_shap[r, :] = np.abs(shap_vals).mean(axis=0)
    
    # Compute rankings (1 = most important)
    rankings = np.zeros((R, M), dtype=int)
    for r in range(R):
        rankings[r, :] = stats.rankdata(-mean_abs_shap[r, :]).astype(int)
    
    # Compute rank sums per feature
    rank_sums = rankings.sum(axis=0)
    
    # Mean rank sum
    mean_rank_sum = np.mean(rank_sums)
    
    # Sum of squared deviations
    S = np.sum((rank_sums - mean_rank_sum) ** 2)
    
    # Kendall's W
    # W = 12S / (R^2 * (M^3 - M))
    denominator = R ** 2 * (M ** 3 - M)
    if denominator == 0:
        return np.ones(M)
    
    W = 12 * S / denominator
    
    # Bound W to [0, 1]
    W = np.clip(W, 0, 1)
    
    # Return as stability (1 - instability)
    # Since W is a global measure, we return it for all features
    return np.full(M, W)


def compute_per_feature_rank_stability(shap_values_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute per-feature rank stability using pairwise rank correlation.
    
    For each feature, compute the variance of its rank across resamples.
    Lower variance = more stable ranking.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays, each with shape (n_samples, n_features).
    
    Returns
    -------
    np.ndarray
        Per-feature rank stability score in [0, 1].
        1 = feature's rank is perfectly stable, 0 = highly unstable.
    """
    if len(shap_values_list) < 2:
        return np.array([])
    
    R = len(shap_values_list)
    M = shap_values_list[0].shape[1]
    
    # Compute mean |SHAP| per feature per resample
    mean_abs_shap = np.zeros((R, M))
    for r, shap_vals in enumerate(shap_values_list):
        mean_abs_shap[r, :] = np.abs(shap_vals).mean(axis=0)
    
    # Compute rankings (1 = most important)
    rankings = np.zeros((R, M), dtype=int)
    for r in range(R):
        rankings[r, :] = stats.rankdata(-mean_abs_shap[r, :]).astype(int)
    
    # For each feature, compute rank stability
    stability = np.zeros(M)
    for f in range(M):
        feature_ranks = rankings[:, f]
        # Normalize by max possible rank variance
        # Max variance occurs when ranks are uniformly distributed
        rank_variance = np.var(feature_ranks, ddof=1)
        # Max variance is over rank values 1..M (feature count), not 1..R (resample count)
        max_variance = np.var(np.arange(1, M + 1), ddof=1) if M > 1 else 1
        # Convert to stability: low variance = high stability
        stability[f] = 1 - np.sqrt(rank_variance / max_variance) if max_variance > 0 else 1
    
    return np.clip(stability, 0, 1)


def compute_shap_wasserstein(
    reference_shap: np.ndarray,
    shap_values_list: List[np.ndarray],
    per_feature: bool = True,
) -> np.ndarray:
    """
    Compute Wasserstein distance between reference and resample SHAP distributions.
    
    Parameters
    ----------
    reference_shap : np.ndarray
        Reference SHAP values with shape (n_samples, n_features).
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
    per_feature : bool, default=True
        If True, return per-feature distances. If False, return global distance.
    
    Returns
    -------
    np.ndarray
        Mean Wasserstein distance per feature (if per_feature=True)
        or single global distance (if per_feature=False).
    """
    if not shap_values_list:
        return np.array([])
    
    M = reference_shap.shape[1]
    
    if per_feature:
        distances = np.zeros((len(shap_values_list), M))
        for r, shap_vals in enumerate(shap_values_list):
            for f in range(M):
                distances[r, f] = stats.wasserstein_distance(
                    reference_shap[:, f],
                    shap_vals[:, f]
                )
        return distances.mean(axis=0)
    else:
        distances = []
        for shap_vals in shap_values_list:
            # Flatten all features
            dist = stats.wasserstein_distance(
                reference_shap.flatten(),
                shap_vals.flatten()
            )
            distances.append(dist)
        return np.array([np.mean(distances)])


def compute_shap_js_divergence(
    reference_shap: np.ndarray,
    shap_values_list: List[np.ndarray],
    n_bins: int = 50,
    per_feature: bool = True,
) -> np.ndarray:
    """
    Compute Jensen-Shannon divergence between reference and resample SHAP distributions.
    
    Parameters
    ----------
    reference_shap : np.ndarray
        Reference SHAP values with shape (n_samples, n_features).
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
    n_bins : int, default=50
        Number of bins for histogram estimation.
    per_feature : bool, default=True
        If True, return per-feature divergence. If False, return global divergence.
    
    Returns
    -------
    np.ndarray
        Mean JS divergence per feature (if per_feature=True)
        or single global divergence (if per_feature=False).
    """
    if not shap_values_list:
        return np.array([])
    
    M = reference_shap.shape[1]
    
    def _compute_js_1d(ref: np.ndarray, other: np.ndarray) -> float:
        """Compute JS divergence for 1D distributions."""
        # Combine to get common bin edges
        combined = np.concatenate([ref, other])
        bin_edges = np.linspace(combined.min(), combined.max(), n_bins + 1)
        
        # Compute histograms
        p, _ = np.histogram(ref, bins=bin_edges, density=True)
        q, _ = np.histogram(other, bins=bin_edges, density=True)
        
        # Normalize
        p = p / (p.sum() + 1e-10)
        q = q / (q.sum() + 1e-10)
        
        # JS divergence (jensenshannon returns sqrt of JS)
        return float(jensenshannon(p, q) ** 2)
    
    if per_feature:
        divergences = np.zeros((len(shap_values_list), M))
        for r, shap_vals in enumerate(shap_values_list):
            for f in range(M):
                divergences[r, f] = _compute_js_1d(reference_shap[:, f], shap_vals[:, f])
        return divergences.mean(axis=0)
    else:
        divergences = []
        for shap_vals in shap_values_list:
            div = _compute_js_1d(reference_shap.flatten(), shap_vals.flatten())
            divergences.append(div)
        return np.array([np.mean(divergences)])


def compute_direction_consistency(
    shap_values_list: List[np.ndarray],
    reference: str = "majority",
    reference_shap: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Compute direction consistency: fraction of samples where SHAP sign is consistent.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
        Each has shape (n_samples, n_features).
    reference : str, default="majority"
        Reference for sign comparison:
        - "majority": Use most common sign per sample/feature as reference
        - "first": Use first resample as reference
        - "provided": Use reference_shap parameter
    reference_shap : Optional[np.ndarray]
        Reference SHAP values if reference="provided".
    
    Returns
    -------
    np.ndarray
        Per-feature direction consistency score in [0, 1].
        1 = all samples have consistent SHAP sign across resamples.
    
    Notes
    -----
    For each sample and feature, we compute the fraction of resamples
    where the SHAP sign matches the reference sign. Then average across samples.
    """
    if len(shap_values_list) < 2:
        return np.array([])
    
    R = len(shap_values_list)
    n_samples, M = shap_values_list[0].shape
    
    # Stack all SHAP values: shape (R, n_samples, M)
    all_shap = np.stack(shap_values_list, axis=0)
    
    # Compute reference signs
    if reference == "majority":
        # Most common sign per (sample, feature)
        mean_shap = all_shap.mean(axis=0)
        reference_signs = np.sign(mean_shap)
    elif reference == "first":
        reference_signs = np.sign(shap_values_list[0])
    elif reference == "provided" and reference_shap is not None:
        reference_signs = np.sign(reference_shap)
    else:
        raise ValueError(f"Unknown reference type: {reference}")
    
    # Handle zero reference signs (can happen with exact zero SHAP)
    reference_signs = np.where(reference_signs == 0, 1, reference_signs)
    
    # Vectorized: compare signs across all resamples simultaneously
    # all_shap shape: (R, n_samples, M); reference_signs shape: (n_samples, M)
    consistency = (np.sign(all_shap) == reference_signs[np.newaxis]).mean(axis=0)

    # Average across samples to get per-feature consistency
    return consistency.mean(axis=0)


def compute_magnitude_cv(shap_values_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute coefficient of variation of |SHAP| values across resamples.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
    
    Returns
    -------
    np.ndarray
        Per-feature coefficient of variation (CV = std/mean) of mean |SHAP|.
        Lower values indicate more stable magnitude.
    
    Notes
    -----
    For each resample, compute mean |SHAP| per feature. Then compute
    CV = std(mean_|SHAP|) / mean(mean_|SHAP|) across resamples.
    """
    if len(shap_values_list) < 2:
        return np.array([])
    
    R = len(shap_values_list)
    M = shap_values_list[0].shape[1]
    
    # Compute mean |SHAP| per feature per resample
    mean_abs_shap = np.zeros((R, M))
    for r, shap_vals in enumerate(shap_values_list):
        mean_abs_shap[r, :] = np.abs(shap_vals).mean(axis=0)
    
    # Compute CV per feature
    cv = np.zeros(M)
    for f in range(M):
        mean_val = mean_abs_shap[:, f].mean()
        std_val = mean_abs_shap[:, f].std(ddof=1)
        cv[f] = std_val / mean_val if mean_val != 0 else 0.0
    
    return cv


def compute_magnitude_iqr(shap_values_list: List[np.ndarray]) -> np.ndarray:
    """
    Compute IQR-based magnitude stability metric.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
    
    Returns
    -------
    np.ndarray
        Per-feature IQR/median ratio of mean |SHAP| values.
        Lower values indicate more stable magnitude.
    """
    if len(shap_values_list) < 2:
        return np.array([])
    
    R = len(shap_values_list)
    M = shap_values_list[0].shape[1]
    
    # Compute mean |SHAP| per feature per resample
    mean_abs_shap = np.zeros((R, M))
    for r, shap_vals in enumerate(shap_values_list):
        mean_abs_shap[r, :] = np.abs(shap_vals).mean(axis=0)
    
    # Compute IQR/median per feature
    iqr_ratio = np.zeros(M)
    for f in range(M):
        q1 = np.percentile(mean_abs_shap[:, f], 25)
        q3 = np.percentile(mean_abs_shap[:, f], 75)
        median = np.percentile(mean_abs_shap[:, f], 50)
        iqr_ratio[f] = (q3 - q1) / median if median != 0 else 0.0
    
    return iqr_ratio


def compute_topk_overlap(
    shap_values_list: List[np.ndarray],
    k: int = 10,
    reference: str = "majority",
) -> float:
    """
    Compute overlap of top-k features across resamples.
    
    Parameters
    ----------
    shap_values_list : List[np.ndarray]
        List of SHAP value arrays from bootstrap resamples.
    k : int, default=10
        Number of top features to consider.
    reference : str, default="majority"
        Reference for comparison:
        - "majority": Features that appear in top-k most frequently
        - "first": Top-k from first resample
    
    Returns
    -------
    float
        Average Jaccard overlap of top-k feature sets with reference.
        1.0 = perfect overlap, 0.0 = no overlap.
    """
    if len(shap_values_list) < 2:
        return 1.0
    
    R = len(shap_values_list)
    M = shap_values_list[0].shape[1]
    k = min(k, M)
    
    # Get top-k feature indices per resample
    topk_sets = []
    for shap_vals in shap_values_list:
        mean_abs = np.abs(shap_vals).mean(axis=0)
        topk = set(np.argsort(-mean_abs)[:k])
        topk_sets.append(topk)
    
    # Compute reference set
    if reference == "majority":
        # Count appearances in top-k
        feature_counts = np.zeros(M)
        for topk in topk_sets:
            for f in topk:
                feature_counts[f] += 1
        reference_set = set(np.argsort(-feature_counts)[:k])
    elif reference == "first":
        reference_set = topk_sets[0]
    else:
        raise ValueError(f"Unknown reference type: {reference}")
    
    # Compute average Jaccard overlap
    overlaps = []
    for topk in topk_sets:
        intersection = len(reference_set & topk)
        union = len(reference_set | topk)
        overlaps.append(intersection / union if union > 0 else 0.0)
    
    return np.mean(overlaps)


class SHAPMetricRunner:
    """
    Compute SHAP stability metrics for bootstrap resamples.
    
    This class is analogous to MetricRunner in core.py but for SHAP values.
    It computes all SHAP stability metrics for a set of bootstrap resamples.
    
    Parameters
    ----------
    reference_shap : np.ndarray
        Reference SHAP values (typically from full-data model).
        Shape: (n_samples, n_features)
    feature_names : List[str]
        Names of features.
    n_bins : int, default=50
        Number of bins for JS divergence computation.
    top_k : int, default=10
        Number of top features for overlap computation.
    """
    
    def __init__(
        self,
        reference_shap: np.ndarray,
        feature_names: List[str],
        n_bins: int = 50,
        top_k: int = 10,
    ):
        self.reference_shap = reference_shap
        self.feature_names = feature_names
        self.n_bins = n_bins
        self.top_k = top_k
        self.n_features = len(feature_names)
    
    def compute_all_metrics(
        self,
        shap_values_list: List[np.ndarray],
    ) -> Dict[str, SHAPMetricResult]:
        """
        Compute all SHAP stability metrics.
        
        Parameters
        ----------
        shap_values_list : List[np.ndarray]
            List of SHAP value arrays from bootstrap resamples.
        
        Returns
        -------
        Dict[str, SHAPMetricResult]
            Dictionary mapping metric names to results.
            Includes both per-feature and global metrics.
        """
        if not shap_values_list:
            return {}
        
        results = {}
        
        # Global rank stability (Kendall's W)
        rank_stability_global = compute_rank_stability(shap_values_list)
        results["rank_stability_global"] = SHAPMetricResult(
            name="rank_stability_global",
            values=rank_stability_global,
            mean=float(rank_stability_global.mean()),
            stderr=0.0,  # Global metric, no stderr
        )
        
        # Per-feature rank stability
        rank_stability_per_feature = compute_per_feature_rank_stability(shap_values_list)
        results["rank_stability"] = SHAPMetricResult(
            name="rank_stability",
            values=rank_stability_per_feature,
            mean=float(rank_stability_per_feature.mean()),
            stderr=float(rank_stability_per_feature.std(ddof=1) / np.sqrt(len(rank_stability_per_feature))),
        )
        
        # Wasserstein distance
        wasserstein = compute_shap_wasserstein(
            self.reference_shap, shap_values_list, per_feature=True
        )
        results["wasserstein"] = SHAPMetricResult(
            name="wasserstein",
            values=wasserstein,
            mean=float(wasserstein.mean()),
            stderr=float(wasserstein.std(ddof=1) / np.sqrt(len(wasserstein))),
        )
        
        # JS divergence
        js_div = compute_shap_js_divergence(
            self.reference_shap, shap_values_list, n_bins=self.n_bins, per_feature=True
        )
        results["js_divergence"] = SHAPMetricResult(
            name="js_divergence",
            values=js_div,
            mean=float(js_div.mean()),
            stderr=float(js_div.std(ddof=1) / np.sqrt(len(js_div))),
        )
        
        # Direction consistency
        direction_cons = compute_direction_consistency(
            shap_values_list, reference="majority"
        )
        results["direction_consistency"] = SHAPMetricResult(
            name="direction_consistency",
            values=direction_cons,
            mean=float(direction_cons.mean()),
            stderr=float(direction_cons.std(ddof=1) / np.sqrt(len(direction_cons))),
        )
        
        # Magnitude CV
        mag_cv = compute_magnitude_cv(shap_values_list)
        results["magnitude_cv"] = SHAPMetricResult(
            name="magnitude_cv",
            values=mag_cv,
            mean=float(mag_cv.mean()),
            stderr=float(mag_cv.std(ddof=1) / np.sqrt(len(mag_cv))),
        )
        
        # Magnitude IQR
        mag_iqr = compute_magnitude_iqr(shap_values_list)
        results["magnitude_iqr"] = SHAPMetricResult(
            name="magnitude_iqr",
            values=mag_iqr,
            mean=float(mag_iqr.mean()),
            stderr=float(mag_iqr.std(ddof=1) / np.sqrt(len(mag_iqr))),
        )
        
        # Top-k overlap
        topk_overlap = compute_topk_overlap(shap_values_list, k=self.top_k)
        results["topk_overlap"] = SHAPMetricResult(
            name="topk_overlap",
            values=np.array([topk_overlap]),
            mean=float(topk_overlap),
            stderr=0.0,  # Single value
        )
        
        return results
    
    def get_feature_results(self, metrics: Dict[str, SHAPMetricResult]) -> List[Dict]:
        """
        Get per-feature summary of all metrics.
        
        Parameters
        ----------
        metrics : Dict[str, SHAPMetricResult]
            Results from compute_all_metrics.
        
        Returns
        -------
        List[Dict]
            List of dictionaries, one per feature, with all metric values.
        """
        feature_results = []
        for i, fname in enumerate(self.feature_names):
            feat_result = {"feature": fname}
            for metric_name, metric_result in metrics.items():
                if len(metric_result.values) == self.n_features:
                    feat_result[metric_name] = float(metric_result.values[i])
                elif len(metric_result.values) == 1:
                    feat_result[metric_name] = float(metric_result.values[0])
            feature_results.append(feat_result)
        return feature_results


def aggregate_shap_metrics(
    all_pool_metrics: List[Dict[str, SHAPMetricResult]],
    metric_names: List[str],
) -> Dict[str, Dict]:
    """
    Aggregate SHAP metrics across pool sizes.
    
    Parameters
    ----------
    all_pool_metrics : List[Dict[str, SHAPMetricResult]]
        List of metric results, one per pool size.
    metric_names : List[str]
        Names of metrics to aggregate.
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping metric names to aggregated results with
        'means', 'stderrs' lists across pool sizes.
    """
    aggregated = {metric: {"means": [], "stderrs": []} for metric in metric_names}
    
    for pool_metrics in all_pool_metrics:
        for metric in metric_names:
            if metric in pool_metrics:
                aggregated[metric]["means"].append(pool_metrics[metric].mean)
                aggregated[metric]["stderrs"].append(pool_metrics[metric].stderr)
            else:
                aggregated[metric]["means"].append(np.nan)
                aggregated[metric]["stderrs"].append(np.nan)
    
    return aggregated
