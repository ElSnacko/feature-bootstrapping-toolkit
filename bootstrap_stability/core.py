import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import curve_fit
from scipy.spatial.distance import jensenshannon


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


def _curve_fn(n, k, floor):
    return k / np.sqrt(n) + floor


def fit_learning_curve(pool_sizes, means, r2_threshold=0.70, extrapolate_to=None) -> dict:
    """Fit k/sqrt(n) + floor to the instability curve."""
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
        "r2": np.nan,
        "anomalous": True,
        "fit_failed": True,
        "extrapolations": {n: np.nan for n in extrapolate_to},
    }

    if len(pool_sizes) < 3:
        return result

    try:
        popt, _ = curve_fit(
            _curve_fn, pool_sizes, means,
            p0=[1.0, means.min()],
            maxfev=5000,
            bounds=([-np.inf, -np.inf], [np.inf, np.inf])
        )
        k, floor = popt

        predicted = _curve_fn(pool_sizes, k, floor)
        ss_res = np.sum((means - predicted) ** 2)
        ss_tot = np.sum((means - means.mean()) ** 2)
        r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

        # Non-monotone check: fitted values should be non-increasing as n grows
        test_n = np.linspace(pool_sizes.min(), pool_sizes.max(), 50)
        test_vals = _curve_fn(test_n, k, floor)
        is_monotone = bool(np.all(np.diff(test_vals) <= 1e-10))

        anomalous = (not np.isfinite(r2)) or (r2 < r2_threshold) or (not is_monotone)

        extrapolations = {
            n: float(_curve_fn(n, k, floor)) for n in extrapolate_to
        }

        result.update({
            "k": float(k),
            "floor": float(floor),
            "r2": float(r2) if np.isfinite(r2) else np.nan,
            "anomalous": anomalous,
            "fit_failed": False,
            "extrapolations": extrapolations,
        })

    except Exception:
        pass

    return result


def fit_all_curves(pool_sizes, learning_curves, r2_threshold, extrapolate_to) -> dict:
    """Fit learning curve for every metric."""
    fitted = {}
    for metric, curve_data in learning_curves.items():
        if curve_data is None or not curve_data.get("means"):
            fitted[metric] = fit_learning_curve([], [], r2_threshold, extrapolate_to)
            continue
        means = [m if m is not None else np.nan for m in curve_data["means"]]
        fitted[metric] = fit_learning_curve(pool_sizes, means, r2_threshold, extrapolate_to)
    return fitted


def compute_complexity_score(fitted_curves, metric_weights, has_target):
    """Weighted average of floor parameters from valid, non-anomalous fits."""
    dist_weights = metric_weights["distributional"]
    td_weights = metric_weights["target_dependent"]

    total_weight = 0.0
    weighted_sum = 0.0
    per_metric_floors = {}

    for metric, fit in fitted_curves.items():
        floor = fit.get("floor", np.nan)
        per_metric_floors[metric] = floor

        if fit.get("fit_failed") or fit.get("anomalous"):
            continue
        if not np.isfinite(floor):
            continue

        if metric in dist_weights:
            w = dist_weights[metric]
        elif metric in td_weights and has_target:
            w = td_weights[metric]
        else:
            continue

        weighted_sum += w * floor
        total_weight += w

    score = weighted_sum / total_weight if total_weight > 0 else np.nan
    return score, per_metric_floors


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
