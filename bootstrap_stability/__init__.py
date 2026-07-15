"""bootstrap_stability — feature stability analysis via bootstrap learning curves.

Importing this package only requires the core dependency stack (numpy, scipy,
pandas). Submodules that need optional dependencies (joblib, matplotlib,
scikit-learn, lightgbm, shap) are loaded lazily on first attribute access; if
the required extra is missing, a clear ``ImportError`` with an install hint is
raised.

The reusable pure-math primitives (``fit_learning_curve``, ``generate_pool_sequence``,
``draw_pool``, ``bootstrap_resample``, ``run_bootstrap_on_pool``, ``bootstrap_ci``)
are part of the public surface and always available.
"""
import importlib

# --------------------------------------------------------------------------- #
# Eager imports — these modules need only numpy/scipy/pandas.
# --------------------------------------------------------------------------- #
from .core import (
    VERSION,
    DEFAULT_WEIGHTS,
    ImbalanceError,
    get_metric_category,
    get_complexity_score,
    TARGET_AGNOSTIC_METRICS,
    TARGET_DEPENDENT_METRICS,
    get_metric_runner,
    MetricRunner,
    CategoricalMetricRunner,
    # Reusable pure-math primitives (Integration Notes fix #2)
    fit_learning_curve,
    generate_pool_sequence,
    draw_pool,
    bootstrap_resample,
    run_bootstrap_on_pool,
    bootstrap_ci,
)
from .categorical_support import (
    supports_categorical,
    get_categorical_feature_indices,
    prepare_categorical_for_model,
    MODELS_WITH_NATIVE_CATEGORICAL,
    MODELS_REQUIRING_ENCODING,
)
from .shap_metrics import (
    SHAPMetricRunner,
    SHAPMetricResult,
    compute_rank_stability,
    compute_per_feature_rank_stability,
    compute_shap_wasserstein,
    compute_shap_js_divergence,
    compute_direction_consistency,
    compute_magnitude_cv,
    compute_magnitude_iqr,
    compute_topk_overlap,
    aggregate_shap_metrics,
)
from .train_holdout import (
    TrainHoldoutStability,
    print_holdout_report,
    compute_shap_rank_correlation,
    compute_direction_flip_rate,
    compute_magnitude_drift,
    compute_topk_overlap as compute_holdout_topk_overlap,
    compute_per_feature_drift,
    compute_overall_drift_score,
    get_drift_grade,
)
from .reliability import (
    ReliabilityConfig,
    ReliabilityResult,
    ReliabilityScorer,
    compute_reliability_score,
    DEFAULT_RELIABILITY_CONFIG,
)
from .validation import MarginalVsSHAPValidator, plot_marginal_vs_shap

__version__ = VERSION

# --------------------------------------------------------------------------- #
# Lazy imports — modules requiring optional extras (Integration Notes fix #1).
# --------------------------------------------------------------------------- #
# Maps an optional submodule (relative name) to the install extra that provides
# its hard dependency.
_LAZY_MODULES = {
    "analyzer": "parallel",            # joblib
    "output": "viz",                   # matplotlib
    "meta_bootstrap": "meta",          # scikit-learn (+ joblib)
    "permutation_baseline": "parallel",  # joblib
    "shap_stability": "shap",          # joblib (+ lightgbm/shap at use time)
    "synthetic_validation": "parallel",  # imports analyzer + permutation_baseline
}

# Maps each public lazy attribute to the submodule that defines it.
_LAZY_ATTRS = {
    # analyzer
    "BootstrapStability": "analyzer",
    # output
    "plot_results": "output",
    "plot_panel": "output",
    "print_report": "output",
    "to_csv": "output",
    "panel_to_csv": "output",
    # meta_bootstrap
    "SplitStrategy": "meta_bootstrap",
    "MetaBootstrapResult": "meta_bootstrap",
    "MetaBootstrap": "meta_bootstrap",
    # permutation_baseline
    "PermutationBaseline": "permutation_baseline",
    # shap_stability
    "SHAPStability": "shap_stability",
    "DEFAULT_SHAP_WEIGHTS": "shap_stability",
    "SHAP_METRIC_NAMES": "shap_stability",
    "SHAP_TARGET_AGNOSTIC_METRICS": "shap_stability",
    "SHAP_TARGET_DEPENDENT_METRICS": "shap_stability",
    # synthetic_validation
    "InstabilityType": "synthetic_validation",
    "TestResult": "synthetic_validation",
    "SyntheticValidation": "synthetic_validation",
    "print_synthetic_report": "synthetic_validation",
}

# Maps a *missing* third-party module name (from ModuleNotFoundError.name) to
# the install extra that provides it. Falls back to the submodule's declared
# extra when the missing module is unrecognized.
_EXTRA_BY_MODULE = {
    "joblib": "parallel",
    "matplotlib": "viz",
    "sklearn": "meta",
    "lightgbm": "shap",
    "shap": "shap",
}

_LAZY_CACHE = {}


def _load_lazy(submod_name):
    if submod_name in _LAZY_CACHE:
        return _LAZY_CACHE[submod_name]

    extra = _LAZY_MODULES[submod_name]
    full = f"bootstrap_stability.{submod_name}"
    try:
        module = importlib.import_module(full)
    except ModuleNotFoundError as exc:
        missing = exc.name
        hint_extra = _EXTRA_BY_MODULE.get(missing, extra)
        raise ImportError(
            f"optional dependency '{missing}' is required by '{full}'. "
            f"Install it with: pip install 'bootstrap-stability[{hint_extra}]'"
        ) from exc
    _LAZY_CACHE[submod_name] = module
    return module


def __getattr__(name):
    submod_name = _LAZY_ATTRS.get(name)
    if submod_name is not None:
        module = _load_lazy(submod_name)
        try:
            return getattr(module, name)
        except AttributeError:
            pass
    raise AttributeError(f"module 'bootstrap_stability' has no attribute {name!r}")


def __dir__():
    return sorted(set(globals()) | set(_LAZY_ATTRS))


__all__ = [
    # Core
    "BootstrapStability",
    "plot_results", "plot_panel", "print_report", "to_csv", "panel_to_csv",
    "VERSION", "DEFAULT_WEIGHTS", "ImbalanceError",

    # Metric Categorization
    "get_metric_category", "get_complexity_score",
    "TARGET_AGNOSTIC_METRICS", "TARGET_DEPENDENT_METRICS",

    # Reusable pure-math primitives (Integration Notes fix #2)
    "fit_learning_curve", "generate_pool_sequence", "draw_pool",
    "bootstrap_resample", "run_bootstrap_on_pool", "bootstrap_ci",

    # SHAP Stability
    "SHAPStability", "DEFAULT_SHAP_WEIGHTS", "SHAP_METRIC_NAMES",
    "SHAP_TARGET_AGNOSTIC_METRICS", "SHAP_TARGET_DEPENDENT_METRICS",
    "SHAPMetricRunner", "SHAPMetricResult",
    "compute_rank_stability", "compute_per_feature_rank_stability",
    "compute_shap_wasserstein", "compute_shap_js_divergence",
    "compute_direction_consistency", "compute_magnitude_cv", "compute_magnitude_iqr",
    "compute_topk_overlap", "aggregate_shap_metrics",

    # Train/holdout Stability
    "TrainHoldoutStability", "print_holdout_report",
    "compute_shap_rank_correlation", "compute_direction_flip_rate",
    "compute_magnitude_drift", "compute_holdout_topk_overlap",
    "compute_per_feature_drift", "compute_overall_drift_score", "get_drift_grade",

    # Reliability Scoring
    "ReliabilityConfig", "ReliabilityResult", "ReliabilityScorer",
    "compute_reliability_score", "DEFAULT_RELIABILITY_CONFIG",

    # Meta-Bootstrap for Confidence Intervals
    "SplitStrategy", "MetaBootstrapResult", "MetaBootstrap",

    # Permutation Baseline
    "PermutationBaseline",

    # Validation
    "MarginalVsSHAPValidator", "plot_marginal_vs_shap",

    # Synthetic Validation Suite
    "InstabilityType", "TestResult", "SyntheticValidation", "print_synthetic_report",
]
