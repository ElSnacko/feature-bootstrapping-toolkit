"""Fix #5 — n_pool_draws: average instability across multiple pool draws.

The analyzer previously drew a single pool per pool size and varied only the
bootstrap resamples within it. At small n the specific pool drawn dominates the
instability estimate. ``n_pool_draws`` (default 1, preserving current behavior)
runs multiple independent pool draws per size and averages them.
"""
import warnings

import numpy as np
import pytest

pytest.importorskip("joblib")

from bootstrap_stability import BootstrapStability  # noqa: E402

# Baseline complexity_score captured from the analyzer BEFORE n_pool_draws was
# introduced (default params, random_state=42, n_resamples=20 on credit_df).
# n_pool_draws=1 MUST reproduce this bit-for-bit.
BASELINE_COMPLEXITY = -0.003675618731021523

FEATURE = "income"
TARGET = "default_flag"


def _fit(credit_df, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bs = BootstrapStability(**kwargs)
        return bs.fit(credit_df, FEATURE, target_col=TARGET)


def test_n_pool_draws_default_is_one():
    bs = BootstrapStability()
    assert bs.n_pool_draws == 1


def test_default_reproduces_baseline(credit_df):
    res = _fit(credit_df, random_state=42, n_resamples=20)
    assert res["complexity_score"] == pytest.approx(BASELINE_COMPLEXITY, rel=0, abs=0)


def test_explicit_one_equals_default(credit_df):
    default = _fit(credit_df, random_state=42, n_resamples=20)
    explicit = _fit(credit_df, random_state=42, n_resamples=20, n_pool_draws=1)
    # Bit-identical: the param must be a true no-op at default.
    assert explicit["complexity_score"] == default["complexity_score"]
    assert (
        explicit["learning_curves"]["wasserstein"]["means"]
        == default["learning_curves"]["wasserstein"]["means"]
    )


def test_multiple_draws_combines_resamples(credit_df):
    single = _fit(credit_df, random_state=42, n_resamples=8, n_pool_draws=1)
    multi = _fit(credit_df, random_state=42, n_resamples=8, n_pool_draws=4)

    # pick a mid-curve pool size that exists in both
    pool = single["pool_sequence"][len(single["pool_sequence"]) // 2]
    n_single = len(single["raw_bootstrap"][pool]["wasserstein"])
    n_multi = len(multi["raw_bootstrap"][pool]["wasserstein"])

    # multi must carry roughly n_pool_draws * n_resamples values (degens skip a few)
    assert n_multi > n_single
    assert n_multi <= 4 * 8
    assert n_multi >= 4 * 8 * 0.7  # tolerate degenerate skips


def test_results_structure_unchanged(credit_df):
    res = _fit(credit_df, random_state=42, n_resamples=10, n_pool_draws=3)
    for key in (
        "meta", "pool_sequence", "excluded_pools", "learning_curves",
        "complexity_score", "complexity_scores", "per_metric_floors",
        "degenerate_rates", "woe_profiles", "percentile_stability", "raw_bootstrap",
    ):
        assert key in res, f"missing key: {key}"
    assert res["meta"]["feature"] == FEATURE


def test_multiple_draws_reduces_cross_run_variance(credit_df):
    """The whole point: averaging pool draws lowers run-to-run instability.

    Asserted on the per-pool instability *means* (what n_pool_draws directly
    stabilizes), not on the fitted complexity_score floor — the floor is a
    nonlinear consequence and converging to the true floor can change its
    spread even as the underlying curve gets smoother.
    """
    seeds = [0, 1, 2, 3, 4, 5, 6]

    def _curve_stds(npd):
        per_seed_means = []
        for s in seeds:
            res = _fit(credit_df, random_state=s, n_resamples=8, n_pool_draws=npd)
            per_seed_means.append(res["learning_curves"]["wasserstein"]["means"])
        # std across seeds at each pool size, averaged over the curve
        arr = np.array(per_seed_means, dtype=float)
        return float(np.nanmean(np.std(arr, axis=0)))

    single_jitter = _curve_stds(1)
    multi_jitter = _curve_stds(5)
    assert multi_jitter < single_jitter
