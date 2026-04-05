import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed

from .core import (
    VERSION,
    DEFAULT_WEIGHTS,
    ImbalanceError,
    check_imbalance,
    adjust_min_events_for_imbalance,
    check_truncation,
    detect_feature_type,
    generate_pool_sequence,
    filter_pool_sequence,
    draw_pool,
    run_bootstrap_on_pool,
    MetricRunner,
    aggregate_resample_metrics,
    aggregate_woe_profiles,
    fit_all_curves,
    compute_complexity_score,
    compute_percentile_stability,
    get_complexity_score,
    get_metric_category,
    TARGET_AGNOSTIC_METRICS,
    TARGET_DEPENDENT_METRICS,
    get_metric_runner,
)


class BootstrapStability:
    def __init__(
        self,
        resample_frac=0.8,
        n_resamples=20,
        n_bins=5,
        min_events=20,
        imbalance_threshold=0.05,
        allow_imbalance=False,
        metric_weights=None,
        min_pool=50,
        linear_threshold=1000,
        n_points=25,
        r2_threshold=0.70,
        extrapolate_to=None,
        store_raw=True,
        n_jobs=-1,
        random_state=42,
        version=VERSION,
        estimate_alpha: bool = False,
        alpha_bounds: tuple = (0.1, 1.0),
        fixed_alpha: float = 0.5,
        support_categorical: bool = False,
        categorical_cardinality_threshold: int = 10,
    ):
        self.resample_frac = resample_frac
        self.n_resamples = n_resamples
        self.n_bins = n_bins
        self.min_events = min_events
        self.imbalance_threshold = imbalance_threshold
        self.allow_imbalance = allow_imbalance
        self.metric_weights = metric_weights if metric_weights is not None else DEFAULT_WEIGHTS
        self.min_pool = min_pool
        self.linear_threshold = linear_threshold
        self.n_points = n_points
        self.r2_threshold = r2_threshold
        self.extrapolate_to = extrapolate_to if extrapolate_to is not None else [500, 1000]
        self.store_raw = store_raw
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.version = version
        self.estimate_alpha = estimate_alpha
        self.alpha_bounds = alpha_bounds
        self.fixed_alpha = fixed_alpha
        self.support_categorical = support_categorical
        self.categorical_cardinality_threshold = categorical_cardinality_threshold

    def fit(self, df: pd.DataFrame, feature_col: str, target_col: str = None) -> dict:
        x_full = df[feature_col].dropna().values
        y_full = df[target_col].dropna().values if target_col is not None else None
        has_target = y_full is not None

        if has_target:
            common_idx = df[[feature_col, target_col]].dropna().index
            x_full = df.loc[common_idx, feature_col].values
            y_full = df.loc[common_idx, target_col].values.astype(int)
        else:
            x_full = df[feature_col].dropna().values

        n_obs = len(x_full)

        feature_type = detect_feature_type(x_full, self.categorical_cardinality_threshold)
        if feature_type == "categorical" and not self.support_categorical:
            raise ValueError(
                f"Feature '{feature_col}' is categorical ({feature_type}). "
                "Only binary and continuous features are supported by default. "
                "Set support_categorical=True to enable categorical feature analysis."
            )

        imbalance_result = {"event_rate": np.nan, "imbalance_flag": False, "severity": "none"}
        if has_target:
            imbalance_result = check_imbalance(
                y_full, self.imbalance_threshold, self.allow_imbalance
            )

        censoring_result = check_truncation(x_full)

        effective_min_events = self.min_events
        if has_target:
            effective_min_events = adjust_min_events_for_imbalance(
                self.min_events, imbalance_result
            )

        pool_sequence = generate_pool_sequence(
            n_obs, self.min_pool, self.linear_threshold, self.n_points
        )

        if has_target:
            valid_pools, excluded_pools = filter_pool_sequence(
                pool_sequence, y_full, effective_min_events
            )
        else:
            valid_pools = pool_sequence
            excluded_pools = np.array([])

        n_pools = len(valid_pools)
        print(
            f"Analyzing '{feature_col}' | n={n_obs} | pools={n_pools} | "
            f"resamples={self.n_resamples} per pool"
        )

        metric_runner = get_metric_runner(x_full, y_full, self.n_bins, feature_type)

        def _process_pool(pool_idx, n_pool):
            seed = self.random_state
            x_pool, y_pool = draw_pool(x_full, y_full, n_pool, seed + pool_idx)
            resample_results, degen_count = run_bootstrap_on_pool(
                x_pool, y_pool,
                self.resample_frac,
                self.n_resamples,
                self.random_state,
                pool_idx,
                metric_runner,
            )
            return pool_idx, n_pool, resample_results, degen_count, x_pool

        pool_outputs = Parallel(n_jobs=self.n_jobs, prefer="threads")(
            delayed(_process_pool)(i, n_pool)
            for i, n_pool in enumerate(valid_pools)
        )
        pool_outputs.sort(key=lambda t: t[0])

        all_metric_names = ["wasserstein", "ks", "js", "spearman", "iv", "monotonicity"]
        learning_curves_raw = {m: {"means": [], "stderrs": []} for m in all_metric_names}
        degenerate_rates = {}
        pool_x_draws = []
        raw_bootstrap = {} if self.store_raw else None
        all_woe_results = []

        for pool_idx, n_pool, resample_results, degen_count, x_pool in pool_outputs:
            degen_rate = degen_count / self.n_resamples if self.n_resamples > 0 else 0.0
            degenerate_rates[int(n_pool)] = degen_rate
            pool_x_draws.append(x_pool)

            if not resample_results:
                for m in all_metric_names:
                    learning_curves_raw[m]["means"].append(np.nan)
                    learning_curves_raw[m]["stderrs"].append(np.nan)
                continue

            # Treat monotonicity as a metric: fraction of monotone resamples
            mono_values = [r["monotone"] for r in resample_results if r.get("monotone") is not None]
            metric_agg = aggregate_resample_metrics(
                resample_results,
                ["wasserstein", "ks", "js", "spearman", "iv"]
            )

            if mono_values:
                metric_agg["monotonicity"] = {
                    "mean": float(1.0 - np.mean(mono_values)),  # instability = non-monotone rate
                    "stderr": float(np.std(mono_values, ddof=1) / np.sqrt(len(mono_values))) if len(mono_values) > 1 else 0.0,
                    "values": [1.0 - float(v) for v in mono_values],
                }
            else:
                metric_agg["monotonicity"] = {"mean": np.nan, "stderr": np.nan, "values": []}

            for m in all_metric_names:
                agg = metric_agg.get(m, {})
                learning_curves_raw[m]["means"].append(agg.get("mean", np.nan))
                learning_curves_raw[m]["stderrs"].append(agg.get("stderr", np.nan))

            if self.store_raw:
                raw_bootstrap[int(n_pool)] = {
                    m: metric_agg[m]["values"] for m in all_metric_names
                }

            all_woe_results.extend(resample_results)

        learning_curves = {}
        for m in all_metric_names:
            learning_curves[m] = {
                "means": learning_curves_raw[m]["means"],
                "stderr": learning_curves_raw[m]["stderrs"],
            }

        fitted_curves = fit_all_curves(
            valid_pools,
            learning_curves,
            self.r2_threshold,
            self.extrapolate_to,
            estimate_alpha=self.estimate_alpha,
            alpha_bounds=self.alpha_bounds,
            fixed_alpha=self.fixed_alpha,
        )

        for m in all_metric_names:
            learning_curves[m]["fit"] = fitted_curves.get(m, {})

        complexity_score, per_metric_floors, complexity_scores = compute_complexity_score(
            fitted_curves, self.metric_weights, has_target
        )

        woe_profiles = aggregate_woe_profiles(all_woe_results, self.n_bins)

        percentile_stability = compute_percentile_stability(pool_x_draws)

        # Flag pools where more than 50% of resamples were degenerate
        high_degen_pools = [p for p, r in degenerate_rates.items() if r > 0.5]

        results = {
            "meta": {
                "feature": feature_col,
                "target": target_col,
                "version": self.version,
                "random_state": self.random_state,
                "n_obs": n_obs,
                "event_rate": imbalance_result.get("event_rate", np.nan),
                "imbalance_flag": imbalance_result.get("imbalance_flag", False),
                "imbalance_severity": imbalance_result.get("severity", "none"),
                "censoring_flag": censoring_result["censoring_flag"],
                "censoring_detail": censoring_result["censoring_detail"],
                "feature_type": feature_type,
                "has_target": has_target,
                "run_timestamp": datetime.utcnow().isoformat(),
                "effective_min_events": effective_min_events,
                "high_degenerate_pools": high_degen_pools,
            },
            "pool_sequence": valid_pools.tolist(),
            "excluded_pools": excluded_pools.tolist(),
            "learning_curves": learning_curves,
            "complexity_score": float(complexity_score) if np.isfinite(complexity_score) else np.nan,
            "complexity_scores": complexity_scores,
            "per_metric_floors": per_metric_floors,
            "degenerate_rates": degenerate_rates,
            "woe_profiles": woe_profiles,
            "percentile_stability": percentile_stability,
            "raw_bootstrap": raw_bootstrap,
        }

        return results

    def fit_panel(self, df: pd.DataFrame, target_col: str = None, feature_cols=None) -> dict:
        if feature_cols is None:
            feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            if target_col is not None and target_col in feature_cols:
                feature_cols = [c for c in feature_cols if c != target_col]

        feature_results = {}
        for feat in feature_cols:
            try:
                result = self.fit(df, feat, target_col)
                feature_results[feat] = result
            except Exception as e:
                print(f"Skipping '{feat}': {e}")

        summary_rows = [self._summarize(r) for r in feature_results.values()]
        summary_df = pd.DataFrame(summary_rows)
        if not summary_df.empty and "complexity_score" in summary_df.columns:
            summary_df = summary_df.sort_values("complexity_score", ascending=True).reset_index(drop=True)

        return {"feature_results": feature_results, "summary": summary_df}

    def _summarize(self, results) -> dict:
        meta = results["meta"]
        lc = results["learning_curves"]

        def _floor(metric):
            fit = lc.get(metric, {}).get("fit", {})
            return fit.get("floor", np.nan)

        def _r2(metric):
            fit = lc.get(metric, {}).get("fit", {})
            return fit.get("r2", np.nan)

        def _mean_full(metric):
            means = lc.get(metric, {}).get("means", [])
            return means[-1] if means else np.nan

        return {
            "feature": meta["feature"],
            "n_obs": meta["n_obs"],
            "event_rate": meta["event_rate"],
            "feature_type": meta["feature_type"],
            "complexity_score": results["complexity_score"],
            "censoring_flag": meta["censoring_flag"],
            "imbalance_flag": meta["imbalance_flag"],
            "wasserstein_floor": _floor("wasserstein"),
            "ks_floor": _floor("ks"),
            "js_floor": _floor("js"),
            "spearman_floor": _floor("spearman"),
            "iv_floor": _floor("iv"),
            "monotonicity_floor": _floor("monotonicity"),
            "wasserstein_r2": _r2("wasserstein"),
            "spearman_mean_full": _mean_full("spearman"),
            "wasserstein_mean_full": _mean_full("wasserstein"),
            "version": meta["version"],
            "run_timestamp": meta["run_timestamp"],
        }
