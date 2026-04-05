from .analyzer import BootstrapStability
from .output import plot_results, plot_panel, print_report, to_csv, panel_to_csv
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
)

# Categorical feature support
from .categorical_support import (
    supports_categorical,
    get_categorical_feature_indices,
    prepare_categorical_for_model,
    MODELS_WITH_NATIVE_CATEGORICAL,
    MODELS_REQUIRING_ENCODING,
)

# SHAP stability modules
from .shap_stability import (
    SHAPStability,
    DEFAULT_SHAP_WEIGHTS,
    SHAP_METRIC_NAMES,
    SHAP_TARGET_AGNOSTIC_METRICS,
    SHAP_TARGET_DEPENDENT_METRICS,
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
from .train_oot import (
    TrainOOTStability,
    print_oot_report,
    compute_shap_rank_correlation,
    compute_direction_flip_rate,
    compute_magnitude_drift,
    compute_topk_overlap as compute_oot_topk_overlap,
    compute_per_feature_drift,
    compute_overall_drift_score,
    get_drift_grade,
)

# Reliability Scoring
from .reliability import (
    ReliabilityConfig,
    ReliabilityResult,
    ReliabilityScorer,
    compute_reliability_score,
    DEFAULT_RELIABILITY_CONFIG,
)

# Meta-Bootstrap for Confidence Intervals
from .meta_bootstrap import (
    SplitStrategy,
    MetaBootstrapResult,
    MetaBootstrap,
)

# Synthetic Validation Suite
from .synthetic_validation import (
    InstabilityType,
    TestResult,
    SyntheticValidation,
    print_synthetic_report,
)

__version__ = VERSION
__all__ = [
    # Core
    "BootstrapStability",
    "plot_results", "plot_panel", "print_report", "to_csv", "panel_to_csv",
    "VERSION", "DEFAULT_WEIGHTS", "ImbalanceError",
    
    # Metric Categorization
    "get_metric_category", "get_complexity_score",
    "TARGET_AGNOSTIC_METRICS", "TARGET_DEPENDENT_METRICS",
    
    # SHAP Stability
    "SHAPStability", "DEFAULT_SHAP_WEIGHTS", "SHAP_METRIC_NAMES",
    "SHAP_TARGET_AGNOSTIC_METRICS", "SHAP_TARGET_DEPENDENT_METRICS",
    "SHAPMetricRunner", "SHAPMetricResult",
    "compute_rank_stability", "compute_per_feature_rank_stability",
    "compute_shap_wasserstein", "compute_shap_js_divergence",
    "compute_direction_consistency", "compute_magnitude_cv", "compute_magnitude_iqr",
    "compute_topk_overlap", "aggregate_shap_metrics",
    
    # Train/OOT Stability
    "TrainOOTStability", "print_oot_report",
    "compute_shap_rank_correlation", "compute_direction_flip_rate",
    "compute_magnitude_drift", "compute_oot_topk_overlap",
    "compute_per_feature_drift", "compute_overall_drift_score", "get_drift_grade",
    
    # Reliability Scoring
    "ReliabilityConfig", "ReliabilityResult", "ReliabilityScorer",
    "compute_reliability_score", "DEFAULT_RELIABILITY_CONFIG",
    
    # Meta-Bootstrap for Confidence Intervals
    "SplitStrategy", "MetaBootstrapResult", "MetaBootstrap",
    
    # Synthetic Validation Suite
    "InstabilityType", "TestResult", "SyntheticValidation", "print_synthetic_report",
]
