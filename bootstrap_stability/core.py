import logging
import warnings
from typing import Optional

import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon

logger = logging.getLogger(__name__)


VERSION = "1.0.0"

DEFAULT_WEIGHTS = {
    "target_dependent": {
        "spearman": 0.35,
        "monotonicity": 0.25,
        "woe_sd": 0.25,
        "iv": 0.15,
    },
    "distributional": {
        "wasserstein": 0.45,
        "ks": 0.30,
        "js": 0.25,
    },
}

DISTRIBUTIONAL_METRICS = ["wasserstein", "ks", "js"]
TARGET_METRICS = ["spearman", "iv"]

# Metric categorization for target-agnostic vs target-dependent separation
TARGET_AGNOSTIC_METRICS = {"wasserstein", "ks", "js"}
TARGET_DEPENDENT_METRICS = {"spearman", "iv", "monotonicity"}


def get_metric_category(metric_name: str) -> str:
    """
    Returns the category of a metric based on its target dependency.
    
    Parameters
    ----------
    metric_name : str
        Name of the metric to categorize.
    
    Returns
    -------
    str
        One of 'target_agnostic', 'target_dependent', or 'unknown'.
        
    Examples
    --------
    >>> get_metric_category('wasserstein')
    'target_agnostic'
    >>> get_metric_category('iv')
    'target_dependent'
    >>> get_metric_category('unknown_metric')
    'unknown'
    """
    metric_lower = metric_name.lower()
    if metric_lower in TARGET_AGNOSTIC_METRICS:
        return "target_agnostic"
    elif metric_lower in TARGET_DEPENDENT_METRICS:
        return "target_dependent"
    else:
        return "unknown"


class ImbalanceError(Exception):
    pass


def check_imbalance(y, threshold=0.05, allow_imbalance=False) -> dict:
    """Check class imbalance in binary target."""
    y = np.asarray(y)
    event_rate = y.mean()
    minority_proportion = min(event_rate, 1 - event_rate)

    if minority_proportion < 0.01:
        severity = "critical"
    elif minority_proportion < 0.05:
        severity = "severe"
    elif minority_proportion < 0.10:
        severity = "moderate"
    else:
        severity = "none"

    imbalance_flag = minority_proportion < threshold

    if imbalance_flag and not allow_imbalance:
        raise ImbalanceError(
            f"Minority class proportion {minority_proportion:.3f} is below threshold "
            f"{threshold}. Set allow_imbalance=True to proceed."
        )

    return {
        "event_rate": float(event_rate),
        "minority_proportion": float(minority_proportion),
        "imbalance_flag": imbalance_flag,
        "severity": severity,
    }


def adjust_min_events_for_imbalance(base_min_events, imbalance_result) -> int:
    """Scale minimum event threshold based on imbalance severity."""
    multipliers = {"moderate": 1.5, "severe": 2.0, "critical": 3.0}
    mult = multipliers.get(imbalance_result["severity"], 1.0)
    return int(np.ceil(base_min_events * mult))


def check_truncation(x, spike_threshold=0.10, min_samples=100) -> dict:
    """Detect policy censoring via hard walls and boundary density spikes."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]

    if len(x) < min_samples:
        return {
            "censoring_flag": False,
            "censoring_detail": "insufficient samples",
            "lower_spike": False,
            "upper_spike": False,
            "lower_boundary": float(x.min()) if len(x) > 0 else np.nan,
            "upper_boundary": float(x.max()) if len(x) > 0 else np.nan,
        }

    x_min, x_max = x.min(), x.max()
    n = len(x)

    prop_at_min = np.mean(x == x_min)
    prop_at_max = np.mean(x == x_max)

    lower_spike = prop_at_min > spike_threshold
    upper_spike = prop_at_max > spike_threshold

    # Hard cutoff signal: outer 1% vs adjacent 4% band
    x_range = x_max - x_min
    if x_range > 0:
        lower_1pct = x_min + 0.01 * x_range
        lower_5pct = x_min + 0.05 * x_range
        upper_1pct = x_max - 0.01 * x_range
        upper_5pct = x_max - 0.05 * x_range

        outer_lower = np.mean(x <= lower_1pct)
        band_lower = np.mean((x > lower_1pct) & (x <= lower_5pct))
        outer_upper = np.mean(x >= upper_1pct)
        band_upper = np.mean((x >= upper_5pct) & (x < upper_1pct))

        if band_lower > 0 and outer_lower > 3 * band_lower:
            lower_spike = True
        if band_upper > 0 and outer_upper > 3 * band_upper:
            upper_spike = True

    censoring_flag = lower_spike or upper_spike
    details = []
    if lower_spike:
        details.append("lower boundary spike")
    if upper_spike:
        details.append("upper boundary spike")
    censoring_detail = "; ".join(details) if details else "none"

    return {
        "censoring_flag": censoring_flag,
        "censoring_detail": censoring_detail,
        "lower_spike": lower_spike,
        "upper_spike": upper_spike,
        "lower_boundary": float(x_min),
        "upper_boundary": float(x_max),
    }


def detect_feature_type(x, cardinality_threshold=10) -> str:
    """Classify feature as binary, categorical, or continuous."""
    x = np.asarray(x)
    x = x[~pd.isna(x)]
    unique_vals = np.unique(x)
    n_unique = len(unique_vals)

    if n_unique == 2:
        return "binary"
    elif n_unique <= cardinality_threshold:
        return "categorical"
    else:
        return "continuous"


def generate_pool_sequence(n, min_pool=50, linear_threshold=1000, n_points=25) -> np.ndarray:
    """Generate pool sizes with linear spacing at small n, log spacing at large n."""
    if n <= linear_threshold:
        sizes = np.linspace(min_pool, n, n_points)
    else:
        linear_part = np.linspace(min_pool, linear_threshold, n_points // 2)
        log_part = np.logspace(
            np.log10(linear_threshold),
            np.log10(n),
            n_points - n_points // 2 + 1
        )[1:]
        sizes = np.concatenate([linear_part, log_part])

    sizes = np.unique(np.round(sizes).astype(int))
    if sizes[-1] != n:
        sizes = np.append(sizes, n)
    return sizes


def filter_pool_sequence(pool_sequence, y_full, min_events=20):
    """Drop pool sizes where expected event count falls below min_events."""
    event_rate = np.mean(y_full)
    valid = []
    excluded = []
    for n_pool in pool_sequence:
        expected_events = n_pool * event_rate
        if expected_events >= min_events:
            valid.append(n_pool)
        else:
            excluded.append(n_pool)
    return np.array(valid), np.array(excluded)


def draw_pool(x, y, n_pool, seed):
    """Draw a pool of size n_pool without replacement."""
    rng = np.random.default_rng(seed)
    n = len(x)
    idx = rng.choice(n, size=min(n_pool, n), replace=False)
    x_pool = x[idx]
    y_pool = y[idx] if y is not None else None
    return x_pool, y_pool


def bootstrap_resample(x, y, resample_frac, seed):
    """Resample with replacement at resample_frac of pool size."""
    rng = np.random.default_rng(seed)
    n = len(x)
    n_boot = max(1, int(n * resample_frac))
    idx = rng.choice(n, size=n_boot, replace=True)
    x_boot = x[idx]
    y_boot = y[idx] if y is not None else None
    return x_boot, y_boot


def run_bootstrap_on_pool(x_pool, y_pool, resample_frac, n_resamples, base_seed, pool_idx, metric_fn):
    """Run n_resamples bootstrap draws on a pool, skipping degenerate ones."""
    results = []
    degen_count = 0

    for r in range(n_resamples):
        seed = base_seed + pool_idx * 1000 + r
        x_boot, y_boot = bootstrap_resample(x_pool, y_pool, resample_frac, seed)

        if np.std(x_boot) == 0:
            degen_count += 1
            continue

        if y_boot is not None and len(np.unique(y_boot)) < 2:
            degen_count += 1
            continue

        metrics = metric_fn(x_boot, y_boot)
        if metrics is not None:
            results.append(metrics)

    return results, degen_count


def _estimate_bandwidth(x) -> float:
    """Silverman's rule of thumb bandwidth."""
    n = len(x)
    std = np.std(x, ddof=1)
    iqr = np.percentile(x, 75) - np.percentile(x, 25)
    s = min(std, iqr / 1.34)
    if s == 0:
        s = std if std > 0 else 1.0
    return 1.06 * s * n ** (-0.2)


def _kde_pdf(x_eval, x_data, bandwidth) -> np.ndarray:
    """Gaussian KDE evaluated at x_eval, normalized to sum to 1."""
    x_data = np.asarray(x_data, dtype=float)
    x_eval = np.asarray(x_eval, dtype=float)
    n = len(x_data)
    diff = (x_eval[:, None] - x_data[None, :]) / bandwidth
    kernel = np.exp(-0.5 * diff ** 2) / (bandwidth * np.sqrt(2 * np.pi))
    pdf = kernel.mean(axis=1)
    total = pdf.sum()
    if total > 0:
        pdf = pdf / total
    return pdf


def _compute_woe_iv(x, y, n_bins=5):
    """Compute WOE and IV via quantile bins, recomputed per resample."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=int)

    try:
        bin_edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 3:
            return np.nan, [], False
        bin_indices = np.digitize(x, bin_edges[1:-1])
    except Exception:
        return np.nan, [], False

    n_bins_actual = len(bin_edges) - 1
    n_events = y.sum()
    n_nonevents = len(y) - n_events

    if n_events == 0 or n_nonevents == 0:
        return np.nan, [], False

    woes = []
    iv = 0.0

    for b in range(n_bins_actual):
        mask = bin_indices == b
        n_e = y[mask].sum() + 0.5
        n_ne = (1 - y[mask]).sum() + 0.5
        dist_e = n_e / (n_events + 0.5 * n_bins_actual)
        dist_ne = n_ne / (n_nonevents + 0.5 * n_bins_actual)
        if dist_e > 0 and dist_ne > 0:
            woe = np.log(dist_e / dist_ne)
        else:
            woe = 0.0
        woes.append(woe)
        iv += (dist_e - dist_ne) * woe

    monotone = True
    if len(woes) > 1:
        diffs = np.diff(woes)
        monotone = bool(np.all(diffs >= 0) or np.all(diffs <= 0))

    return float(iv), woes, monotone


class MetricRunner:
    """Initialized once from full dataset; called per resample to compute metrics."""

    def __init__(self, x_full, y_full, n_bins=5):
        self.n_bins = n_bins
        self.has_target = y_full is not None

        x_full = np.asarray(x_full, dtype=float)
        self.bandwidth = _estimate_bandwidth(x_full)
        self.eval_grid = np.linspace(x_full.min(), x_full.max(), 200)
        self.ref_kde = _kde_pdf(self.eval_grid, x_full, self.bandwidth)
        self.x_ref = x_full

    def __call__(self, x_boot, y_boot) -> dict:
        x_boot = np.asarray(x_boot, dtype=float)

        boot_kde = _kde_pdf(self.eval_grid, x_boot, self.bandwidth)

        # Wasserstein distance
        wasserstein = float(stats.wasserstein_distance(
            self.eval_grid, self.eval_grid,
            u_weights=self.ref_kde, v_weights=boot_kde
        ))

        # KS statistic
        ks_stat, _ = stats.ks_2samp(self.x_ref, x_boot)

        # JS divergence — clip to avoid log(0)
        p = np.clip(self.ref_kde, 1e-10, None)
        q = np.clip(boot_kde, 1e-10, None)
        p = p / p.sum()
        q = q / q.sum()
        js = float(jensenshannon(p, q))

        result = {
            "wasserstein": wasserstein,
            "ks": float(ks_stat),
            "js": js,
            "woe_profile": None,
            "monotone": None,
            "spearman": None,
            "iv": None,
        }

        if self.has_target and y_boot is not None:
            y_boot = np.asarray(y_boot, dtype=int)
            spearman_r, _ = stats.spearmanr(x_boot, y_boot)
            result["spearman"] = float(abs(spearman_r))

            iv, woes, monotone = _compute_woe_iv(x_boot, y_boot, self.n_bins)
            result["iv"] = iv
            result["woe_profile"] = woes
            result["monotone"] = monotone

        return result


class CategoricalMetricRunner:
    """Metric runner for categorical features.
    
    Uses Total Variation distance instead of Wasserstein for distribution comparison,
    and computes JS divergence for discrete/categorical data.
    """
    
    def __init__(self, x_full, y_full, n_bins=5):
        self.n_bins = n_bins
        self.has_target = y_full is not None
        x_full = np.asarray(x_full)
        self.ref_value_counts = self._compute_value_counts(x_full)
        self.x_ref = x_full
        self.categories = list(self.ref_value_counts.keys())
    
    def _compute_value_counts(self, x) -> dict:
        """Compute normalized value counts for categorical data."""
        x = np.asarray(x)
        x = x[~pd.isna(x)]
        if len(x) == 0:
            return {}
        unique, counts = np.unique(x, return_counts=True)
        total = counts.sum()
        return {val: count / total for val, count in zip(unique, counts)}
    
    def _compute_tv_distance(self, p_counts: dict, q_counts: dict) -> float:
        """Compute Total Variation distance between two categorical distributions."""
        all_keys = set(p_counts.keys()) | set(q_counts.keys())
        tv_sum = 0.0
        for key in all_keys:
            p_val = p_counts.get(key, 0.0)
            q_val = q_counts.get(key, 0.0)
            tv_sum += abs(p_val - q_val)
        return tv_sum / 2.0
    
    def _compute_js_divergence_categorical(self, p_counts: dict, q_counts: dict) -> float:
        """Compute Jensen-Shannon divergence for categorical distributions."""
        all_keys = set(p_counts.keys()) | set(q_counts.keys())
        p = np.array([p_counts.get(k, 0.0) for k in sorted(all_keys)])
        q = np.array([q_counts.get(k, 0.0) for k in sorted(all_keys)])
        # Normalize to ensure proper distributions
        p = p / p.sum() if p.sum() > 0 else p
        q = q / q.sum() if q.sum() > 0 else q
        # Clip to avoid log(0)
        p = np.clip(p, 1e-10, None)
        q = np.clip(q, 1e-10, None)
        return float(jensenshannon(p, q))
    
    def _compute_woe_iv_categorical(self, x, y) -> tuple:
        """Compute WOE and IV using value-based grouping for categorical data."""
        x = np.asarray(x)
        y = np.asarray(y, dtype=int)
        
        unique_vals = np.unique(x)
        if len(unique_vals) == 0:
            return np.nan, [], False
        
        n_events = y.sum()
        n_nonevents = len(y) - n_events
        
        if n_events == 0 or n_nonevents == 0:
            return np.nan, [], False
        
        woes = []
        iv = 0.0
        
        for val in sorted(unique_vals):
            mask = x == val
            n_e = y[mask].sum() + 0.5
            n_ne = (1 - y[mask]).sum() + 0.5
            dist_e = n_e / (n_events + 0.5 * len(unique_vals))
            dist_ne = n_ne / (n_nonevents + 0.5 * len(unique_vals))
            if dist_e > 0 and dist_ne > 0:
                woe = np.log(dist_e / dist_ne)
            else:
                woe = 0.0
            woes.append(woe)
            iv += (dist_e - dist_ne) * woe
        
        # Monotonicity doesn't apply meaningfully to unordered categories
        # Return False by default for categorical
        monotone = False
        
        return float(iv), woes, monotone
    
    def __call__(self, x_boot, y_boot) -> dict:
        """Return metrics dict with tv_distance, ks, js, woe_profile, monotone, spearman, iv."""
        x_boot = np.asarray(x_boot)
        
        boot_value_counts = self._compute_value_counts(x_boot)
        
        # Total Variation distance (replaces Wasserstein for categorical)
        tv_distance = self._compute_tv_distance(self.ref_value_counts, boot_value_counts)
        
        # KS statistic (still works for categorical when encoded as numeric)
        ks_stat, _ = stats.ks_2samp(self.x_ref, x_boot)
        
        # JS divergence for categorical
        js = self._compute_js_divergence_categorical(self.ref_value_counts, boot_value_counts)
        
        result = {
            "wasserstein": tv_distance,  # Use TV distance, but keep key name for compatibility
            "ks": float(ks_stat),
            "js": js,
            "woe_profile": None,
            "monotone": None,
            "spearman": None,
            "iv": None,
        }
        
        if self.has_target and y_boot is not None:
            y_boot = np.asarray(y_boot, dtype=int)
            # Spearman correlation (works with encoded categoricals)
            spearman_r, _ = stats.spearmanr(x_boot, y_boot)
            result["spearman"] = float(abs(spearman_r))
            
            # WOE/IV using category values as bins
            iv, woes, monotone = self._compute_woe_iv_categorical(x_boot, y_boot)
            result["iv"] = iv
            result["woe_profile"] = woes
            result["monotone"] = monotone
        
        return result


def get_metric_runner(x_full, y_full, n_bins=5, feature_type=None):
    """Factory function to get appropriate metric runner.
    
    Parameters
    ----------
    x_full : array-like
        Full feature array
    y_full : array-like, optional
        Full target array
    n_bins : int, default=5
        Number of bins for WOE/IV computation
    feature_type : str, optional
        Feature type: 'binary', 'categorical', or 'continuous'.
        If None, will be detected automatically.
    
    Returns
    -------
    MetricRunner or CategoricalMetricRunner
        Appropriate metric runner for the feature type
    """
    if feature_type is None:
        feature_type = detect_feature_type(x_full)
    
    if feature_type == "categorical":
        return CategoricalMetricRunner(x_full, y_full, n_bins)
    return MetricRunner(x_full, y_full, n_bins)


def aggregate_resample_metrics(resample_results, metric_names) -> dict:
    """Per metric: mean, stderr, and raw values list."""
    aggregated = {}
    for metric in metric_names:
        values = [r[metric] for r in resample_results if r.get(metric) is not None and not np.isnan(r.get(metric, np.nan))]
        if values:
            arr = np.array(values)
            aggregated[metric] = {
                "mean": float(np.mean(arr)),
                "stderr": float(np.std(arr, ddof=1) / np.sqrt(len(arr))) if len(arr) > 1 else 0.0,
                "values": values,
            }
        else:
            aggregated[metric] = {"mean": np.nan, "stderr": np.nan, "values": []}
    return aggregated


def aggregate_woe_profiles(resample_results, n_bins) -> dict:
    """Per bin: mean WOE, SD, and sign flip rate across resamples."""
    all_profiles = [r["woe_profile"] for r in resample_results if r.get("woe_profile")]
    if not all_profiles:
        return {}

    profiles_by_bin = {}
    for b in range(n_bins):
        bin_woes = [p[b] for p in all_profiles if len(p) > b]
        if not bin_woes:
            continue
        arr = np.array(bin_woes)
        majority_sign = np.sign(np.median(arr))
        sign_flips = np.mean(np.sign(arr) != majority_sign)
        profiles_by_bin[f"bin_{b + 1}"] = {
            "mean_woe": float(np.mean(arr)),
            "sd_woe": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
            "sign_flip_rate": float(sign_flips),
        }
    return profiles_by_bin


def _curve_fn(n, k, floor, alpha=0.5):
    """Learning curve function: k / (n ** alpha) + floor
    
    Parameters
    ----------
    n : array-like
        Pool sizes
    k : float
        Scale parameter
    floor : float
        Asymptotic floor value
    alpha : float, default=0.5
        Decay exponent (0.5 = CLT rate for sample means)
    
    Returns
    -------
    array-like
        Predicted instability values
    """
    return k / (n ** alpha) + floor


def _compute_r2(means, predicted):
    """Compute R² score for fit quality."""
    ss_res = np.sum((means - predicted) ** 2)
    ss_tot = np.sum((means - means.mean()) ** 2)
    if ss_tot > 0:
        return 1 - ss_res / ss_tot
    return np.nan


def fit_learning_curve(
    pool_sizes,
    means,
    r2_threshold=0.70,
    extrapolate_to=None,
    estimate_alpha: bool = False,
    alpha_bounds: tuple = (0.1, 1.0),
    fixed_alpha: float = 0.5,
) -> dict:
    """Fit k / (n ** alpha) + floor to the instability curve.
    
    Parameters
    ----------
    pool_sizes : array-like
        Pool sizes used in the learning curve
    means : array-like
        Mean instability values at each pool size
    r2_threshold : float, default=0.70
        Minimum R² for a fit to be considered valid
    extrapolate_to : list, optional
        Pool sizes to extrapolate to (default: [500, 1000])
    estimate_alpha : bool, default=False
        When True, estimate alpha from data; when False, use fixed_alpha
    alpha_bounds : tuple, default=(0.1, 1.0)
        Bounds for alpha estimation (min, max)
    fixed_alpha : float, default=0.5
        Alpha value to use when estimate_alpha=False
    
    Returns
    -------
    dict
        Fit results including:
        - k, floor, alpha: fitted parameters
        - r2: R² score of fit
        - alpha_estimated: whether alpha was estimated or fixed
        - r2_improvement: R² improvement vs fixed alpha (if estimated)
        - anomalous: whether fit is anomalous
        - fit_failed: whether fitting failed
        - extrapolations: predicted values at extrapolate_to sizes
    """
    if extrapolate_to is None:
        extrapolate_to = [500, 1000]

    pool_sizes = np.asarray(pool_sizes, dtype=float)
    means = np.asarray(means, dtype=float)

    valid_mask = ~np.isnan(means)
    pool_sizes = pool_sizes[valid_mask]
    means = means[valid_mask]

    result = {
        "k": np.nan,
        "floor": np.nan,
        "alpha": fixed_alpha,
        "alpha_estimated": False,
        "r2": np.nan,
        "r2_improvement": None,
        "anomalous": True,
        "fit_failed": True,
        "extrapolations": {n: np.nan for n in extrapolate_to},
    }

    if len(pool_sizes) < 3:
        return result

    # Define curve function wrapper for fixed alpha
    def _curve_fn_fixed_alpha(n, k, floor):
        return _curve_fn(n, k, floor, alpha=fixed_alpha)
    
    # Define curve function wrapper for flexible alpha
    def _curve_fn_flexible_alpha(n, k, floor, alpha):
        return _curve_fn(n, k, floor, alpha=alpha)

    try:
        # First, fit with fixed alpha (for baseline R²)
        popt_fixed, _ = curve_fit(
            _curve_fn_fixed_alpha, pool_sizes, means,
            p0=[1.0, means.min()],
            maxfev=5000,
            bounds=([-np.inf, -np.inf], [np.inf, np.inf])
        )
        k_fixed, floor_fixed = popt_fixed
        predicted_fixed = _curve_fn_fixed_alpha(pool_sizes, k_fixed, floor_fixed)
        r2_fixed = _compute_r2(means, predicted_fixed)
        
        if estimate_alpha:
            # Fit with flexible alpha
            try:
                popt_flex, _ = curve_fit(
                    _curve_fn_flexible_alpha, pool_sizes, means,
                    p0=[k_fixed, floor_fixed, fixed_alpha],
                    maxfev=5000,
                    bounds=([-np.inf, -np.inf, alpha_bounds[0]],
                           [np.inf, np.inf, alpha_bounds[1]])
                )
                k, floor, alpha = popt_flex
                predicted = _curve_fn_flexible_alpha(pool_sizes, k, floor, alpha)
                r2 = _compute_r2(means, predicted)
                
                # Calculate R² improvement
                if np.isfinite(r2) and np.isfinite(r2_fixed):
                    r2_improvement = r2 - r2_fixed
                else:
                    r2_improvement = None
                
                # Warn if estimated alpha is far from 0.5
                if np.isfinite(alpha) and abs(alpha - 0.5) > 0.2:
                    logger.warning(
                        f"Estimated alpha ({alpha:.3f}) differs significantly from "
                        f"default 0.5. This may indicate metric convergence differs "
                        f"from CLT rate or potential model misspecification."
                    )
                
                result.update({
                    "k": float(k),
                    "floor": float(floor),
                    "alpha": float(alpha),
                    "alpha_estimated": True,
                    "r2": float(r2) if np.isfinite(r2) else np.nan,
                    "r2_improvement": float(r2_improvement) if r2_improvement is not None else None,
                })
                
            except Exception as e:
                # Fall back to fixed alpha if flexible fitting fails
                logger.debug(f"Alpha estimation failed, using fixed alpha: {e}")
                result.update({
                    "k": float(k_fixed),
                    "floor": float(floor_fixed),
                    "alpha": fixed_alpha,
                    "alpha_estimated": False,
                    "r2": float(r2_fixed) if np.isfinite(r2_fixed) else np.nan,
                    "r2_improvement": None,
                })
        else:
            # Use fixed alpha results
            result.update({
                "k": float(k_fixed),
                "floor": float(floor_fixed),
                "alpha": fixed_alpha,
                "alpha_estimated": False,
                "r2": float(r2_fixed) if np.isfinite(r2_fixed) else np.nan,
                "r2_improvement": None,
            })

        # Non-monotone check: fitted values should be non-increasing as n grows
        alpha_for_check = result["alpha"]
        test_n = np.linspace(pool_sizes.min(), pool_sizes.max(), 50)
        test_vals = _curve_fn(test_n, result["k"], result["floor"], alpha_for_check)
        is_monotone = bool(np.all(np.diff(test_vals) <= 1e-10))

        anomalous = (not np.isfinite(result["r2"])) or (result["r2"] < r2_threshold) or (not is_monotone)

        extrapolations = {
            n: float(_curve_fn(n, result["k"], result["floor"], result["alpha"]))
            for n in extrapolate_to
        }

        result.update({
            "anomalous": anomalous,
            "fit_failed": False,
            "extrapolations": extrapolations,
        })

    except Exception as e:
        logger.debug(f"fit_learning_curve failed, returning default result: {e}")

    return result


def fit_all_curves(
    pool_sizes,
    learning_curves,
    r2_threshold,
    extrapolate_to,
    estimate_alpha: bool = False,
    alpha_bounds: tuple = (0.1, 1.0),
    fixed_alpha: float = 0.5,
) -> dict:
    """Fit learning curve for every metric.
    
    Parameters
    ----------
    pool_sizes : array-like
        Pool sizes used in the learning curve
    learning_curves : dict
        Learning curve data for each metric
    r2_threshold : float
        Minimum R² for a fit to be considered valid
    extrapolate_to : list
        Pool sizes to extrapolate to
    estimate_alpha : bool, default=False
        When True, estimate alpha from data
    alpha_bounds : tuple, default=(0.1, 1.0)
        Bounds for alpha estimation
    fixed_alpha : float, default=0.5
        Alpha value when not estimating
    
    Returns
    -------
    dict
        Fitted curves for each metric
    """
    fitted = {}
    for metric, curve_data in learning_curves.items():
        if curve_data is None or not curve_data.get("means"):
            fitted[metric] = fit_learning_curve(
                [], [], r2_threshold, extrapolate_to,
                estimate_alpha=estimate_alpha,
                alpha_bounds=alpha_bounds,
                fixed_alpha=fixed_alpha,
            )
            continue
        means = [m if m is not None else np.nan for m in curve_data["means"]]
        fitted[metric] = fit_learning_curve(
            pool_sizes, means, r2_threshold, extrapolate_to,
            estimate_alpha=estimate_alpha,
            alpha_bounds=alpha_bounds,
            fixed_alpha=fixed_alpha,
        )
    return fitted


def compute_complexity_score(fitted_curves, metric_weights, has_target):
    """
    Weighted average of floor parameters from valid, non-anomalous fits.
    
    Returns separate scores for target-agnostic metrics (distributional),
    target-dependent metrics, and an overall combined score.
    
    Parameters
    ----------
    fitted_curves : dict
        Dictionary of fitted curve results for each metric.
    metric_weights : dict
        Weights for each metric category with keys 'distributional' and 'target_dependent'.
    has_target : bool
        Whether target data is available for computing target-dependent metrics.
    
    Returns
    -------
    tuple
        (overall_score, per_metric_floors, complexity_scores_dict)
        - overall_score: float - Combined weighted score (backwards compatible)
        - per_metric_floors: dict - Floor values for each metric
        - complexity_scores_dict: dict with keys 'overall', 'target_agnostic', 'target_dependent'
    """
    dist_weights = metric_weights["distributional"]
    td_weights = metric_weights["target_dependent"]

    # Track separate scores by category
    target_agnostic_weight = 0.0
    target_agnostic_sum = 0.0
    target_dependent_weight = 0.0
    target_dependent_sum = 0.0
    
    per_metric_floors = {}

    for metric, fit in fitted_curves.items():
        floor = fit.get("floor", np.nan)
        per_metric_floors[metric] = floor

        if fit.get("fit_failed") or fit.get("anomalous"):
            continue
        if not np.isfinite(floor):
            continue

        # Categorize metric and apply appropriate weight
        if metric in dist_weights:
            w = dist_weights[metric]
            target_agnostic_sum += w * floor
            target_agnostic_weight += w
        elif metric in td_weights and has_target:
            w = td_weights[metric]
            target_dependent_sum += w * floor
            target_dependent_weight += w
        else:
            continue

    # Compute category-specific scores
    target_agnostic_score = (
        target_agnostic_sum / target_agnostic_weight
        if target_agnostic_weight > 0 else np.nan
    )
    target_dependent_score = (
        target_dependent_sum / target_dependent_weight
        if target_dependent_weight > 0 else np.nan
    )
    
    # Compute overall score (combined weighted average)
    total_weight = target_agnostic_weight + target_dependent_weight
    total_sum = target_agnostic_sum + target_dependent_sum
    overall_score = total_sum / total_weight if total_weight > 0 else np.nan
    
    # Build complexity_scores dict
    complexity_scores = {
        "overall": float(overall_score) if np.isfinite(overall_score) else np.nan,
        "target_agnostic": float(target_agnostic_score) if np.isfinite(target_agnostic_score) else np.nan,
        "target_dependent": float(target_dependent_score) if np.isfinite(target_dependent_score) else np.nan,
    }
    
    return overall_score, per_metric_floors, complexity_scores


def get_complexity_score(results: dict, category: str = "overall") -> float:
    """
    Get complexity score by category from fit results.
    
    This accessor function provides a clean interface for retrieving
    complexity scores by category from BootstrapStability.fit() results.
    
    Parameters
    ----------
    results : dict
        Fit results from BootstrapStability.fit() or SHAPStability.fit().
    category : str, default='overall'
        The category of complexity score to retrieve:
        - 'overall': Combined score (weighted average of all metrics)
        - 'target_agnostic': Distributional metrics only (Wasserstein, KS, JS)
        - 'target_dependent': Target-based metrics only (Spearman, IV, Monotonicity)
        
    Returns
    -------
    float
        Complexity score for the specified category. Returns np.nan if:
        - The category is not found
        - No valid metrics were available for that category
        
    Raises
    ------
    ValueError
        If category is not one of 'overall', 'target_agnostic', or 'target_dependent'.
        
    Examples
    --------
    >>> results = analyzer.fit(X, y)
    >>> get_complexity_score(results, 'overall')
    0.45
    >>> get_complexity_score(results, 'target_agnostic')
    0.32
    >>> get_complexity_score(results, 'target_dependent')
    0.58
    """
    valid_categories = {"overall", "target_agnostic", "target_dependent"}
    if category not in valid_categories:
        raise ValueError(
            f"Invalid category '{category}'. Must be one of: {valid_categories}"
        )
    
    # Try new complexity_scores dict first
    if "complexity_scores" in results:
        return results["complexity_scores"].get(category, np.nan)
    
    # Fallback to legacy complexity_score for backwards compatibility
    if category == "overall" and "complexity_score" in results:
        return results["complexity_score"]
    
    # No score available for this category
    return np.nan


def compute_percentile_stability(pool_x_draws, percentiles=(10, 25, 50, 75, 90)) -> dict:
    """Track feature percentiles across pool sizes."""
    result = {f"p{p}": [] for p in percentiles}
    for x_draws in pool_x_draws:
        if x_draws is None or len(x_draws) == 0:
            for p in percentiles:
                result[f"p{p}"].append(np.nan)
        else:
            for p in percentiles:
                result[f"p{p}"].append(float(np.percentile(x_draws, p)))
    return result
