"""Shared pytest fixtures for the bootstrap_stability test suite."""
import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def normal_sample():
    """Large normal sample with a known mean for CI coverage tests."""
    rng = np.random.default_rng(42)
    return rng.normal(loc=5.0, scale=2.0, size=4000)


@pytest.fixture
def normal_sample_mean():
    return 5.0


@pytest.fixture
def lognormal_sample():
    """Heavy-tailed sample used to exercise non-trivial statistics."""
    rng = np.random.default_rng(7)
    return rng.lognormal(mean=1.0, sigma=0.8, size=4000)


@pytest.fixture
def credit_df():
    """Credit-risk-style frame: continuous feature + correlated binary target.

    Sized so default ``min_pool``/``n_points`` produce a usable learning curve.
    """
    rng = np.random.default_rng(0)
    n = 800
    income = rng.normal(loc=60000, scale=15000, size=n)
    logits = (income - 60000) / 15000 * -0.8
    p = 1.0 / (1.0 + np.exp(-logits))
    default = (rng.random(n) < p).astype(int)
    return pd.DataFrame({"income": income, "default_flag": default})
