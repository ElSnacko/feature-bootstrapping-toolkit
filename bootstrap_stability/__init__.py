from .analyzer import BootstrapStability
from .output import plot_results, plot_panel, print_report, to_csv, panel_to_csv
from .core import VERSION, DEFAULT_WEIGHTS, ImbalanceError

# SHAP stability modules
from .shap_stability import SHAPStability, DEFAULT_SHAP_WEIGHTS, SHAP_METRIC_NAMES
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

__version__ = VERSION
__all__ = [
    # Core
    "BootstrapStability",
    "plot_results", "plot_panel", "print_report", "to_csv", "panel_to_csv",
    "VERSION", "DEFAULT_WEIGHTS", "ImbalanceError",
    
    # SHAP Stability
    "SHAPStability", "DEFAULT_SHAP_WEIGHTS", "SHAP_METRIC_NAMES",
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
]
