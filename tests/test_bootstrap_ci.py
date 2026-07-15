"""Fix #4 — general-purpose percentile bootstrap CI helper in core.

A consumer who wants the simplest thing a "bootstrapping toolkit" implies — a
percentile CI on an arbitrary statistic of a 1-D sample — should not have to
reach into MetaBootstrap. These tests pin ``bootstrap_ci``'s contract.
"""
import numpy as np
import pytest

from bootstrap_stability import bootstrap_ci
from bootstrap_stability.core import bootstrap_ci as bootstrap_ci_core


def test_export_matches_core_object():
    assert bootstrap_ci is bootstrap_ci_core


def test_ci_brackets_true_mean(normal_sample, normal_sample_mean):
    res = bootstrap_ci(normal_sample, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42)
    assert res["ci_lower"] < normal_sample_mean < res["ci_upper"]


def test_point_estimate_matches_stat_on_full_sample(normal_sample):
    res = bootstrap_ci(normal_sample, stat_fn=np.mean, n_boot=500, seed=1)
    assert res["point_estimate"] == pytest.approx(np.mean(normal_sample))


def test_ci_ordering_and_width(normal_sample):
    res = bootstrap_ci(normal_sample, stat_fn=np.mean, n_boot=2000, ci=0.95, seed=42)
    assert res["ci_lower"] < res["ci_upper"]
    # wider confidence -> wider interval (same seed/stream length differences aside)
    narrow = bootstrap_ci(normal_sample, stat_fn=np.mean, n_boot=4000, ci=0.90, seed=42)
    wide = bootstrap_ci(normal_sample, stat_fn=np.mean, n_boot=4000, ci=0.99, seed=42)
    assert (wide["ci_upper"] - wide["ci_lower"]) > (narrow["ci_upper"] - narrow["ci_lower"])


def test_custom_stat_fn_median(normal_sample):
    res = bootstrap_ci(normal_sample, stat_fn=np.median, n_boot=1000, seed=3)
    assert res["point_estimate"] == pytest.approx(np.median(normal_sample))
    assert res["ci_lower"] <= np.median(normal_sample) <= res["ci_upper"]


def test_determinism_with_seed(normal_sample):
    a = bootstrap_ci(normal_sample, n_boot=500, seed=99)
    b = bootstrap_ci(normal_sample, n_boot=500, seed=99)
    assert a == b


def test_different_seeds_differ(normal_sample):
    a = bootstrap_ci(normal_sample, n_boot=500, seed=1)
    b = bootstrap_ci(normal_sample, n_boot=500, seed=2)
    assert a != b


def test_output_contract(normal_sample):
    res = bootstrap_ci(normal_sample, n_boot=100, ci=0.95, seed=0)
    for key in ("point_estimate", "ci_lower", "ci_upper", "n_boot", "ci"):
        assert key in res
    assert res["n_boot"] == 100
    assert res["ci"] == pytest.approx(0.95)


def test_nan_values_ignored():
    arr = np.array([1.0, 2.0, np.nan, 3.0, 4.0, 5.0])
    res = bootstrap_ci(arr, n_boot=200, seed=0)
    assert res["point_estimate"] == pytest.approx(3.0)


def test_empty_raises():
    with pytest.raises(ValueError):
        bootstrap_ci(np.array([]), n_boot=100, seed=0)
