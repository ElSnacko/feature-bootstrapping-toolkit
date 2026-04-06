"""
SHAP Stability Analyzer Module

This module provides the main SHAPStability class for measuring stability
of SHAP-based feature contributions across bootstrap resamples.

The approach mirrors BootstrapStability but operates on SHAP values rather
than marginal feature distributions.
"""

import numpy as np
import pandas as pd
from datetime import datetime
from joblib import Parallel, delayed
from typing import Callable, Dict, List, Optional, Union, Any

from .core import (
    generate_pool_sequence,
    filter_pool_sequence,
    fit_learning_curve,
    fit_all_curves,
    VERSION,
)
from .shap_metrics import (
    SHAPMetricRunner,
    SHAPMetricResult,
    aggregate_shap_metrics,
)


# Default weights for SHAP complexity score
DEFAULT_SHAP_WEIGHTS = {
    "primary": {
        "rank_stability": 0.30,
        "direction_consistency": 0.30,
    },
    "secondary": {
        "wasserstein": 0.15,
        "magnitude_cv": 0.15,
    },
    "tertiary": {
        "js_divergence": 0.10,
    },
}

# All SHAP metric names
SHAP_METRIC_NAMES = [
    "rank_stability_global",
    "rank_stability",
    "wasserstein",
    "js_divergence",
    "direction_consistency",
    "magnitude_cv",
    "magnitude_iqr",
    "topk_overlap",
]

# SHAP metrics are all target-dependent (require model trained on target)
SHAP_TARGET_DEPENDENT_METRICS = {
    "rank_stability_global",
    "rank_stability",
    "wasserstein",
    "js_divergence",
    "direction_consistency",
    "magnitude_cv",
    "magnitude_iqr",
    "topk_overlap",
}
SHAP_TARGET_AGNOSTIC_METRICS = set()  # SHAP metrics all require a trained model


def _get_explainer(model, explainer_type: str, explainer_kwargs: dict = None):
    """
    Get SHAP explainer for the model.
    
    Parameters
    ----------
    model : Any
        Fitted model.
    explainer_type : str
        Type of explainer: 'tree', 'kernel', 'linear', 'deep', or 'auto'.
    explainer_kwargs : dict, optional
        Additional arguments for explainer.
    
    Returns
    -------
    shap.Explainer
        SHAP explainer instance.
    """
    try:
        import shap
    except ImportError:
        raise ImportError("shap is required. Install with: pip install shap")
    
    kwargs = explainer_kwargs or {}
    
    if explainer_type == "tree":
        return shap.TreeExplainer(model, **kwargs)
    elif explainer_type == "kernel":
        return shap.KernelExplainer(model.predict, **kwargs)
    elif explainer_type == "linear":
        return shap.LinearExplainer(model, **kwargs)
    elif explainer_type == "deep":
        return shap.DeepExplainer(model, **kwargs)
    elif explainer_type == "auto":
        return shap.Explainer(model, **kwargs)
    else:
        raise ValueError(f"Unknown explainer type: {explainer_type}")


def _compute_shap_values(
    model,
    X: np.ndarray,
    explainer_type: str,
    explainer_kwargs: dict = None,
    subsample: int = None,
    random_state: int = None,
) -> np.ndarray:
    """
    Compute SHAP values for a model and dataset.
    
    Parameters
    ----------
    model : Any
        Fitted model.
    X : np.ndarray
        Feature matrix.
    explainer_type : str
        Type of SHAP explainer.
    explainer_kwargs : dict, optional
        Additional explainer arguments.
    subsample : int, optional
        If set, subsample X for SHAP computation.
    random_state : int, optional
        Random state for subsampling.
    
    Returns
    -------
    np.ndarray
        SHAP values with shape (n_samples, n_features).
    """
    # Subsample if requested
    if subsample is not None and len(X) > subsample:
        rng = np.random.default_rng(random_state)
        idx = rng.choice(len(X), size=subsample, replace=False)
        X_eval = X[idx]
    else:
        X_eval = X
    
    # Get explainer and compute SHAP values
    explainer = _get_explainer(model, explainer_type, explainer_kwargs)
    shap_values = explainer.shap_values(X_eval)
    
    # Handle list output (binary classification)
    if isinstance(shap_values, list):
        # Use positive class SHAP values
        shap_values = shap_values[1]
    
    return np.array(shap_values)


def _train_and_compute_shap(
    model_factory: Callable,
    X_boot: np.ndarray,
    y_boot: np.ndarray,
    X_eval: np.ndarray,
    explainer_type: str,
    explainer_kwargs: dict = None,
    shap_subsample: int = None,
    random_state: int = None,
    categorical_feature_indices: list = None,
) -> tuple:
    """
    Train model and compute SHAP values.
    
    Parameters
    ----------
    model_factory : Callable
        Function that returns an unfitted model.
    X_boot : np.ndarray
        Training features.
    y_boot : np.ndarray
        Training target.
    X_eval : np.ndarray
        Evaluation features for SHAP computation.
    explainer_type : str
        Type of SHAP explainer.
    explainer_kwargs : dict, optional
        Additional explainer arguments.
    shap_subsample : int, optional
        Subsample for SHAP computation.
    random_state : int, optional
        Random state.
    categorical_feature_indices : list, optional
        Indices of categorical features for native handling.
    
    Returns
    -------
    tuple
        (model, shap_values)
    """
    # Train model
    model = model_factory()
    
    # Handle categorical features for models with native support
    if categorical_feature_indices is not None:
        model_class_name = model.__class__.__name__
        module_name = model.__class__.__module__.split(".")[0]
        
        # LightGBM native categorical support
        if module_name == "lightgbm" or "lightgbm" in str(type(model)).lower():
            model.fit(
                X_boot, y_boot,
                categorical_feature=categorical_feature_indices
            )
        # CatBoost native categorical support
        elif module_name == "catboost" or "catboost" in str(type(model)).lower():
            model.fit(
                X_boot, y_boot,
                cat_features=categorical_feature_indices
            )
        else:
            # Standard fit for other models
            model.fit(X_boot, y_boot)
    else:
        model.fit(X_boot, y_boot)
    
    # Compute SHAP values
    shap_values = _compute_shap_values(
        model, X_eval, explainer_type, explainer_kwargs,
        subsample=shap_subsample, random_state=random_state
    )
    
    return model, shap_values


class SHAPStability:
    """
    Measure stability of SHAP-based feature contributions across bootstrap resamples.
    
    Unlike marginal stability which measures if feature distributions stabilize,
    this measures if feature contributions to model decisions stabilize.
    
    Parameters
    ----------
    model_factory : Callable
        Function that returns an unfitted model instance.
        Example: `lambda: LGBMClassifier(n_estimators=100, max_depth=6)`
    explainer_type : str, default='tree'
        Type of SHAP explainer: 'tree', 'kernel', 'linear', 'deep', 'auto'.
    explainer_kwargs : dict, optional
        Additional arguments passed to SHAP explainer.
    
    # Bootstrap parameters (mirror BootstrapStability)
    resample_frac : float, default=0.8
        Fraction of pool to resample with replacement.
    n_resamples : int, default=20
        Number of bootstrap resamples per pool size.
    min_pool : int, default=100
        Minimum pool size.
    linear_threshold : int, default=1000
        Threshold for switching from linear to log spacing.
    n_points : int, default=15
        Number of pool sizes to evaluate.
    
    # SHAP-specific parameters
    eval_set_strategy : str, default='holdout'
        Strategy for evaluation set:
        - 'holdout': Use fixed held-out set (recommended)
        - 'pool': Evaluate on current pool
        - 'full': Evaluate on full dataset
    eval_set_size : float, default=0.2
        Fraction of data to hold out for evaluation (if strategy='holdout').
    compute_interactions : bool, default=False
        Whether to compute SHAP interaction values (expensive).
    shap_subsample : int, optional
        If set, subsample evaluation set for SHAP computation.
    
    # Model training
    retrain_per_bootstrap : bool, default=False
        If True, retrain model per bootstrap resample (Option B).
        If False, train once on full data (Option A, default).
    
    # Fitting parameters
    r2_threshold : float, default=0.70
        R² threshold below which fit is flagged as anomalous.
    extrapolate_to : list, optional
        Pool sizes to extrapolate to. Default: [500, 1000].
    metric_weights : dict, optional
        Weights for computing complexity score. Default: DEFAULT_SHAP_WEIGHTS.
    
    # Computation
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed.
    store_raw : bool, default=True
        Whether to store raw SHAP values.
    verbose : int, default=1
        Verbosity level: 0=silent, 1=progress, 2=debug.
    
    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> from bootstrap_stability import SHAPStability
    >>> 
    >>> def create_model():
    ...     return LGBMClassifier(n_estimators=100, max_depth=6)
    >>> 
    >>> shap_stab = SHAPStability(model_factory=create_model)
    >>> results = shap_stab.fit(X, y)
    """
    
    def __init__(
        self,
        model_factory: Callable[[], Any],
        explainer_type: str = 'tree',
        explainer_kwargs: dict = None,
        
        # Bootstrap parameters
        resample_frac: float = 0.8,
        n_resamples: int = 20,
        min_pool: int = 100,
        linear_threshold: int = 1000,
        n_points: int = 15,
        
        # SHAP-specific parameters
        eval_set_strategy: str = 'holdout',
        eval_set_size: float = 0.2,
        compute_interactions: bool = False,
        shap_subsample: int = None,
        
        # Model training
        retrain_per_bootstrap: bool = False,
        
        # Fitting parameters
        r2_threshold: float = 0.70,
        extrapolate_to: list = None,
        metric_weights: dict = None,
        
        # Alpha parameters for flexible learning curve
        estimate_alpha: bool = False,
        alpha_bounds: tuple = (0.1, 1.0),
        fixed_alpha: float = 0.5,
        
        # Categorical feature support
        support_categorical: bool = False,
        categorical_feature_indices: list = None,
        
        # Computation
        n_jobs: int = -1,
        random_state: int = 42,
        store_raw: bool = True,
        verbose: int = 1,
    ):
        self.model_factory = model_factory
        self.explainer_type = explainer_type
        self.explainer_kwargs = explainer_kwargs or {}
        
        self.resample_frac = resample_frac
        self.n_resamples = n_resamples
        self.min_pool = min_pool
        self.linear_threshold = linear_threshold
        self.n_points = n_points
        
        self.eval_set_strategy = eval_set_strategy
        self.eval_set_size = eval_set_size
        self.compute_interactions = compute_interactions
        self.shap_subsample = shap_subsample
        
        self.retrain_per_bootstrap = retrain_per_bootstrap
        
        self.r2_threshold = r2_threshold
        self.extrapolate_to = extrapolate_to if extrapolate_to is not None else [500, 1000]
        self.metric_weights = metric_weights if metric_weights is not None else DEFAULT_SHAP_WEIGHTS
        
        self.estimate_alpha = estimate_alpha
        self.alpha_bounds = alpha_bounds
        self.fixed_alpha = fixed_alpha
        
        self.support_categorical = support_categorical
        self.categorical_feature_indices = categorical_feature_indices
        
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.store_raw = store_raw
        self.verbose = verbose
    
    def _print(self, msg: str, level: int = 1):
        """Print message if verbosity level is sufficient."""
        if self.verbose >= level:
            print(msg)
    
    def _train_model_with_categorical(self, model, X_boot: np.ndarray, y_boot: np.ndarray) -> Any:
        """Train model with proper handling of categorical features.
        
        For models with native categorical support (LightGBM, CatBoost),
        passes categorical_feature parameter to fit method.
        
        Parameters
        ----------
        model : Any
            Unfitted model instance.
        X_boot : np.ndarray
            Training features.
        y_boot : np.ndarray
            Training target.
        
        Returns
        -------
        Any
            Fitted model.
        """
        if not self.support_categorical or not self.categorical_feature_indices:
            # Standard training without categorical handling
            model.fit(X_boot, y_boot)
            return model
        
        # Check if model supports categorical features natively
        model_class_name = model.__class__.__name__
        module_name = model.__class__.__module__.split(".")[0]
        
        # LightGBM native categorical support
        if module_name == "lightgbm" or "lightgbm" in str(type(model)).lower():
            model.fit(
                X_boot, y_boot,
                categorical_feature=self.categorical_feature_indices
            )
            return model
        
        # CatBoost native categorical support
        if module_name == "catboost" or "catboost" in str(type(model)).lower():
            model.fit(
                X_boot, y_boot,
                cat_features=self.categorical_feature_indices
            )
            return model
        
        # For other models, try standard fit (may require pre-encoded data)
        model.fit(X_boot, y_boot)
        return model
    
    def _prepare_data(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        feature_names: list = None,
    ) -> tuple:
        """Prepare data for analysis."""
        # Convert to numpy
        if isinstance(X, pd.DataFrame):
            if feature_names is None:
                feature_names = list(X.columns)
            X = X.values
        elif isinstance(X, np.ndarray):
            if feature_names is None:
                feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if y is not None:
            if isinstance(y, pd.Series):
                y = y.values
        
        return X, y, feature_names
    
    def _create_eval_set(
        self,
        X: np.ndarray,
        y: np.ndarray = None,
    ) -> tuple:
        """Create evaluation set based on strategy."""
        n = len(X)
        
        if self.eval_set_strategy == 'holdout':
            # Hold out a fraction for evaluation
            rng = np.random.default_rng(self.random_state)
            holdout_idx = rng.choice(n, size=int(n * self.eval_set_size), replace=False)
            train_idx = np.setdiff1d(np.arange(n), holdout_idx)
            
            X_train = X[train_idx]
            y_train = y[train_idx] if y is not None else None
            X_eval = X[holdout_idx]
            y_eval = y[holdout_idx] if y is not None else None
            
            return X_train, y_train, X_eval, y_eval, train_idx, holdout_idx
        
        elif self.eval_set_strategy == 'full':
            # Evaluate on full dataset
            return X, y, X, y, np.arange(n), np.arange(n)
        
        elif self.eval_set_strategy == 'pool':
            # Will be handled per pool
            return X, y, None, None, np.arange(n), None
        
        else:
            raise ValueError(f"Unknown eval_set_strategy: {self.eval_set_strategy}")
    
    def _draw_pool(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_pool: int,
        seed: int,
        train_idx: np.ndarray = None,
    ) -> tuple:
        """Draw a pool of size n_pool."""
        rng = np.random.default_rng(seed)
        
        if train_idx is not None:
            # Only sample from training indices
            n_available = len(train_idx)
            idx = rng.choice(train_idx, size=min(n_pool, n_available), replace=False)
        else:
            n_available = len(X)
            idx = rng.choice(n_available, size=min(n_pool, n_available), replace=False)
        
        X_pool = X[idx]
        y_pool = y[idx] if y is not None else None
        
        return X_pool, y_pool, idx
    
    def _bootstrap_resample(
        self,
        X_pool: np.ndarray,
        y_pool: np.ndarray,
        seed: int,
    ) -> tuple:
        """Resample with replacement from pool."""
        rng = np.random.default_rng(seed)
        n = len(X_pool)
        n_boot = max(1, int(n * self.resample_frac))
        idx = rng.choice(n, size=n_boot, replace=True)
        
        X_boot = X_pool[idx]
        y_boot = y_pool[idx] if y_pool is not None else None
        
        return X_boot, y_boot
    
    def _process_pool_option_a(
        self,
        pool_idx: int,
        n_pool: int,
        X: np.ndarray,
        y: np.ndarray,
        reference_model,
        X_eval: np.ndarray,
        train_idx: np.ndarray,
        feature_names: List[str],
    ) -> tuple:
        """
        Process a single pool with Option A (single model).
        
        Train model once on full data, compute SHAP on bootstrap samples.
        """
        rng = np.random.default_rng(self.random_state + pool_idx)
        
        shap_values_list = []
        degen_count = 0
        
        for r in range(self.n_resamples):
            seed = self.random_state + pool_idx * 1000 + r
            
            # Draw pool
            X_pool, y_pool, _ = self._draw_pool(X, y, n_pool, seed, train_idx)
            
            # Bootstrap resample
            X_boot, y_boot = self._bootstrap_resample(X_pool, y_pool, seed + 10000)
            
            # Check for degenerate resamples
            if y_boot is not None and len(np.unique(y_boot)) < 2:
                degen_count += 1
                continue
            
            # Compute SHAP on bootstrap sample using reference model
            shap_vals = _compute_shap_values(
                reference_model,
                X_boot,
                self.explainer_type,
                self.explainer_kwargs,
                subsample=self.shap_subsample,
                random_state=seed + 20000,
            )
            shap_values_list.append(shap_vals)
        
        return pool_idx, n_pool, shap_values_list, degen_count
    
    def _process_pool_option_b(
        self,
        pool_idx: int,
        n_pool: int,
        X: np.ndarray,
        y: np.ndarray,
        X_eval: np.ndarray,
        train_idx: np.ndarray,
        feature_names: List[str],
    ) -> tuple:
        """
        Process a single pool with Option B (retrain per bootstrap).
        
        Retrain model per bootstrap, compute SHAP on fixed evaluation set.
        """
        shap_values_list = []
        degen_count = 0
        
        for r in range(self.n_resamples):
            seed = self.random_state + pool_idx * 1000 + r
            
            # Draw pool
            X_pool, y_pool, _ = self._draw_pool(X, y, n_pool, seed, train_idx)
            
            # Bootstrap resample
            X_boot, y_boot = self._bootstrap_resample(X_pool, y_pool, seed + 10000)
            
            # Check for degenerate resamples
            if y_boot is not None and len(np.unique(y_boot)) < 2:
                degen_count += 1
                continue
            
            # Determine evaluation set for this resample
            if self.eval_set_strategy == 'pool':
                X_eval_r = X_boot
            else:
                X_eval_r = X_eval
            
            try:
                # Train model and compute SHAP
                _, shap_vals = _train_and_compute_shap(
                    self.model_factory,
                    X_boot,
                    y_boot,
                    X_eval_r,
                    self.explainer_type,
                    self.explainer_kwargs,
                    shap_subsample=self.shap_subsample,
                    random_state=seed + 20000,
                    categorical_feature_indices=self.categorical_feature_indices if self.support_categorical else None,
                )
                shap_values_list.append(shap_vals)
            except Exception as e:
                self._print(f"Warning: Resample failed: {e}", level=2)
                degen_count += 1
        
        return pool_idx, n_pool, shap_values_list, degen_count
    
    def fit(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        feature_names: list = None,
    ) -> dict:
        """
        Compute SHAP stability metrics for all features.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray, optional
            Target variable. Required for supervised models.
        feature_names : list, optional
            Feature names. Inferred from X if DataFrame.
        
        Returns
        -------
        dict
            Results dictionary with keys:
            - 'meta': Metadata about the run
            - 'pool_sequence': Pool sizes used
            - 'learning_curves': Per-metric learning curves with fits
            - 'feature_results': Per-feature SHAP stability metrics
            - 'complexity_score': Weighted average of SHAP floors
            - 'raw_shap': Raw SHAP values if store_raw=True
        """
        # Prepare data
        X, y, feature_names = self._prepare_data(X, y, feature_names)
        n_obs = len(X)
        n_features = len(feature_names)
        
        if y is None:
            raise ValueError("Target y is required for SHAP stability analysis.")
        
        # Create evaluation set
        X_train, y_train, X_eval, y_eval, train_idx, eval_idx = self._create_eval_set(X, y)
        
        if self.eval_set_strategy == 'holdout':
            n_train = len(X_train)
            n_eval = len(X_eval)
            self._print(f"Train set: {n_train} samples, Eval set: {n_eval} samples")
        else:
            n_train = len(X_train)
            self._print(f"Using {self.eval_set_strategy} evaluation strategy")
        
        # Generate pool sequence
        pool_sequence = generate_pool_sequence(
            n_train, self.min_pool, self.linear_threshold, self.n_points
        )
        
        # Filter pool sequence based on minimum events
        event_rate = np.mean(y_train)
        min_events = max(20, int(20 / max(event_rate, 0.01)))  # Adjust for imbalance
        valid_pools, excluded_pools = filter_pool_sequence(
            pool_sequence, y_train, min_events
        )
        
        n_pools = len(valid_pools)
        self._print(f"Analyzing SHAP stability | n={n_train} | pools={n_pools} | resamples={self.n_resamples}")
        
        # Train reference model (for Option A or as reference for metrics)
        self._print("Training reference model on full data...")
        try:
            reference_model = self.model_factory()
            reference_model.fit(X_train, y_train)
            self._print("Reference model trained successfully", level=1)
        except Exception as e:
            self._print(f"ERROR: Failed to train reference model: {e}", level=1)
            raise
        
        # Compute reference SHAP values
        self._print("Computing reference SHAP values...", level=1)
        try:
            reference_shap = _compute_shap_values(
                reference_model,
                X_eval if X_eval is not None else X_train,
                self.explainer_type,
                self.explainer_kwargs,
                subsample=self.shap_subsample,
                random_state=self.random_state,
            )
            
            # Validate SHAP values
            if reference_shap is None or len(reference_shap) == 0:
                self._print("ERROR: Reference SHAP values are empty", level=1)
                raise ValueError("Reference SHAP values are empty")
            
            # Check for degenerate SHAP values (all zeros or constant)
            shap_std = np.std(reference_shap, axis=0)
            n_constant = np.sum(shap_std < 1e-10)
            if n_constant == n_features:
                self._print("WARNING: All SHAP values are constant (zero variance)", level=1)
                self._print("  This may indicate a model that doesn't use the features", level=1)
            elif n_constant > 0:
                self._print(f"WARNING: {n_constant}/{n_features} features have constant SHAP values", level=1)
            
            self._print(f"Reference SHAP computed: shape={reference_shap.shape}", level=1)
            
        except Exception as e:
            self._print(f"ERROR: Failed to compute reference SHAP values: {e}", level=1)
            raise
        
        # Initialize metric runner
        metric_runner = SHAPMetricRunner(
            reference_shap=reference_shap,
            feature_names=feature_names,
        )
        
        # Process pools
        if self.retrain_per_bootstrap:
            self._print("Using Option B: Retraining model per bootstrap resample")
        else:
            self._print("Using Option A: Single model, bootstrap SHAP data")
        
        # Process pools sequentially to avoid pickling issues with bound methods
        pool_outputs = []
        for i, n_pool in enumerate(valid_pools):
            if self.retrain_per_bootstrap:
                result = self._process_pool_option_b(
                    i, n_pool, X, y, X_eval, train_idx, feature_names
                )
            else:
                result = self._process_pool_option_a(
                    i, n_pool, X, y, reference_model, X_eval, train_idx, feature_names
                )
            pool_outputs.append(result)
        pool_outputs.sort(key=lambda t: t[0])
        
        # Aggregate results
        all_pool_metrics = []
        degenerate_rates = {}
        raw_shap = {} if self.store_raw else None
        
        for pool_idx, n_pool, shap_values_list, degen_count in pool_outputs:
            degen_rate = degen_count / self.n_resamples if self.n_resamples > 0 else 0.0
            degenerate_rates[int(n_pool)] = degen_rate
            
            if not shap_values_list:
                all_pool_metrics.append(None)
                continue
            
            # Compute metrics
            metrics = metric_runner.compute_all_metrics(shap_values_list)
            all_pool_metrics.append(metrics)
            
            # Store raw SHAP values
            if self.store_raw:
                raw_shap[int(n_pool)] = shap_values_list
        
        # Aggregate metrics across pools
        self._print("Aggregating metrics across pools...", level=1)
        aggregated = aggregate_shap_metrics(all_pool_metrics, SHAP_METRIC_NAMES)
        
        # Log aggregation summary
        n_valid_pools = sum(1 for m in all_pool_metrics if m is not None)
        self._print(f"Valid pool metrics: {n_valid_pools}/{len(all_pool_metrics)}", level=1)
        
        # Stability metrics increase with pool size (higher = more stable).
        # The k/n^alpha + floor model expects decreasing curves (instability).
        # Invert stability metrics to instability (1 - value) before fitting
        # so the model can fit them properly.
        _STABILITY_METRICS = {
            "rank_stability_global", "rank_stability",
            "direction_consistency", "topk_overlap",
        }

        # Build learning curves
        learning_curves = {}
        for metric in SHAP_METRIC_NAMES:
            means = list(aggregated[metric]["means"])
            stderrs = list(aggregated[metric]["stderrs"])
            if metric in _STABILITY_METRICS:
                means = [1.0 - m if not np.isnan(m) else m for m in means]
            learning_curves[metric] = {
                "means": means,
                "stderr": stderrs,
            }
            # Log learning curve summary
            if means and len(means) > 0:
                valid_means = [m for m in means if not np.isnan(m)]
                if valid_means:
                    self._print(f"  {metric}: mean range [{min(valid_means):.4f}, {max(valid_means):.4f}]"
                                + (" (inverted)" if metric in _STABILITY_METRICS else ""), level=2)

        # Fit learning curves
        self._print(f"Fitting learning curves (r2_threshold={self.r2_threshold})...", level=1)
        fitted_curves = fit_all_curves(
            valid_pools,
            learning_curves,
            self.r2_threshold,
            self.extrapolate_to,
            estimate_alpha=self.estimate_alpha,
            alpha_bounds=self.alpha_bounds,
            fixed_alpha=self.fixed_alpha,
        )
        
        # Log fitting results
        n_successful_fits = sum(1 for f in fitted_curves.values() if not f.get("fit_failed", True))
        self._print(f"Successful curve fits: {n_successful_fits}/{len(SHAP_METRIC_NAMES)}", level=1)
        
        for metric in SHAP_METRIC_NAMES:
            learning_curves[metric]["fit"] = fitted_curves.get(metric, {})
        
        # Compute complexity score
        self._print("Computing SHAP complexity score from fitted curves...", level=1)
        complexity_score, per_metric_floors, complexity_scores = self._compute_shap_complexity_score(
            fitted_curves, aggregated_metrics=aggregated
        )
        
        # Validate complexity score
        if not np.isfinite(complexity_score):
            self._print("", level=1)
            self._print("=" * 60, level=1)
            self._print("WARNING: SHAP complexity score is NaN", level=1)
            self._print("=" * 60, level=1)
            self._print("This may indicate:", level=1)
            self._print("  1. Insufficient data for curve fitting (try more samples)", level=1)
            self._print("  2. All curve fits were anomalous (check r2_threshold)", level=1)
            self._print("  3. No valid metric weights configured", level=1)
            self._print("  4. SHAP values are constant/zero for all features", level=1)
            self._print(f"Per-metric floors: {per_metric_floors}", level=1)
            self._print("=" * 60, level=1)
            self._print("", level=1)
        else:
            self._print(f"SHAP complexity score computed: {complexity_score:.4f}", level=1)
        
        # Build feature results
        feature_results = self._build_feature_results(
            all_pool_metrics, feature_names, valid_pools
        )
        
        # Build results dict
        results = {
            "meta": {
                "n_obs": n_obs,
                "n_train": n_train if self.eval_set_strategy == 'holdout' else n_obs,
                "n_eval": n_eval if self.eval_set_strategy == 'holdout' else n_obs,
                "n_features": n_features,
                "feature_names": feature_names,
                "eval_set_strategy": self.eval_set_strategy,
                "retrain_per_bootstrap": self.retrain_per_bootstrap,
                "explainer_type": self.explainer_type,
                "version": VERSION,
                "random_state": self.random_state,
                "run_timestamp": datetime.utcnow().isoformat(),
            },
            "pool_sequence": valid_pools.tolist(),
            "excluded_pools": excluded_pools.tolist(),
            "learning_curves": learning_curves,
            "complexity_score": float(complexity_score) if np.isfinite(complexity_score) else np.nan,
            "complexity_scores": complexity_scores,
            "per_metric_floors": per_metric_floors,
            "feature_results": feature_results,
            "degenerate_rates": degenerate_rates,
            "raw_shap": raw_shap,
        }
        
        return results
    
    def _compute_shap_complexity_score(
        self,
        fitted_curves: dict,
        aggregated_metrics: dict = None,
    ) -> tuple:
        """
        Compute weighted average of SHAP floor parameters.
        
        For SHAP metrics, we want lower values for most metrics
        (except direction_consistency and rank_stability where higher is better).
        
        All SHAP metrics are target-dependent since they require a model trained on the target.
        
        Enhanced with validation and diagnostic logging for debugging NaN issues.
        
        Returns
        -------
        tuple
            (overall_score, per_metric_floors, complexity_scores_dict)
            - overall_score: float - Combined weighted score (backwards compatible)
            - per_metric_floors: dict - Floor values for each metric
            - complexity_scores_dict: dict with keys 'overall', 'target_agnostic', 'target_dependent'
              Note: For SHAP, 'target_agnostic' will always be NaN since all SHAP metrics
              require a model trained on the target.
        """
        import warnings
        
        # Flatten weights
        all_weights = {}
        for tier, weights in self.metric_weights.items():
            all_weights.update(weights)
        
        # Diagnostic: Check if we have any fitted curves
        if not fitted_curves:
            self._print("WARNING: No fitted curves provided to _compute_shap_complexity_score", level=1)
            complexity_scores = {
                "overall": np.nan,
                "target_agnostic": np.nan,
                "target_dependent": np.nan,
            }
            return np.nan, {}, complexity_scores
        
        total_weight = 0.0
        weighted_sum = 0.0
        per_metric_floors = {}
        
        # Track diagnostics
        n_total = len(fitted_curves)
        n_failed = 0
        n_anomalous = 0
        n_nan_floor = 0
        n_zero_weight = 0
        n_valid = 0
        
        for metric, fit in fitted_curves.items():
            floor = fit.get("floor", np.nan)
            per_metric_floors[metric] = floor
            
            # Track why metrics are excluded
            if fit.get("fit_failed"):
                n_failed += 1
                self._print(f"  DEBUG: {metric} - fit failed", level=2)
                continue
            if fit.get("anomalous"):
                n_anomalous += 1
                self._print(f"  DEBUG: {metric} - anomalous (floor={floor:.4f})", level=2)
                continue
            if not np.isfinite(floor):
                n_nan_floor += 1
                self._print(f"  DEBUG: {metric} - NaN floor", level=2)
                continue
            
            # Get weight
            w = all_weights.get(metric, 0.0)
            if w == 0:
                n_zero_weight += 1
                self._print(f"  DEBUG: {metric} - zero weight in metric_weights", level=2)
                continue
            
            # For metrics where higher is better (stability metrics),
            # we need to invert: instability = 1 - stability
            if metric in ["rank_stability", "rank_stability_global", "direction_consistency", "topk_overlap"]:
                # These are stability metrics (higher = more stable)
                # Convert to instability: floor_instability = 1 - floor_stability
                floor_instability = 1 - floor
                weighted_sum += w * floor_instability
                self._print(f"  DEBUG: {metric} - valid (floor={floor:.4f}, instability={floor_instability:.4f}, weight={w})", level=2)
            else:
                # These are instability metrics (lower = more stable)
                weighted_sum += w * floor
                self._print(f"  DEBUG: {metric} - valid (floor={floor:.4f}, weight={w})", level=2)
            
            total_weight += w
            n_valid += 1
        
        # Diagnostic summary
        self._print(f"SHAP complexity score diagnostics:", level=1)
        self._print(f"  Total metrics: {n_total}", level=1)
        self._print(f"  Valid contributions: {n_valid}", level=1)
        self._print(f"  Fit failures: {n_failed}", level=1)
        self._print(f"  Anomalous fits: {n_anomalous}", level=1)
        self._print(f"  NaN floors: {n_nan_floor}", level=1)
        self._print(f"  Zero weight: {n_zero_weight}", level=1)
        self._print(f"  Total weight: {total_weight:.4f}", level=1)
        
        # Fallback with warning if no valid fits
        if total_weight == 0:
            warnings.warn(
                "No valid curve fits found for SHAP complexity score. "
                f"Diagnostics: total={n_total}, failed={n_failed}, anomalous={n_anomalous}, "
                f"nan_floor={n_nan_floor}, zero_weight={n_zero_weight}. "
                "Consider increasing n_resamples or adjusting r2_threshold."
            )
            self._print("WARNING: No valid curve fits found for SHAP complexity score", level=1)
            
            # Fallback: try to compute a proxy from raw metric values
            fallback_score = self._compute_fallback_complexity_score(aggregated_metrics or {}, all_weights)
            if np.isfinite(fallback_score):
                self._print(f"  Using fallback complexity score: {fallback_score:.4f}", level=1)
                complexity_scores = {
                    "overall": float(fallback_score),
                    "target_agnostic": np.nan,  # SHAP metrics are all target-dependent
                    "target_dependent": float(fallback_score),
                }
                return fallback_score, per_metric_floors, complexity_scores
            else:
                complexity_scores = {
                    "overall": np.nan,
                    "target_agnostic": np.nan,
                    "target_dependent": np.nan,
                }
                return np.nan, per_metric_floors, complexity_scores
        
        score = weighted_sum / total_weight
        self._print(f"  Final complexity score: {score:.4f}", level=1)
        
        # Build complexity_scores dict
        # Note: All SHAP metrics are target-dependent, so target_agnostic is always NaN
        complexity_scores = {
            "overall": float(score) if np.isfinite(score) else np.nan,
            "target_agnostic": np.nan,  # SHAP metrics require a model trained on target
            "target_dependent": float(score) if np.isfinite(score) else np.nan,
        }
        
        return score, per_metric_floors, complexity_scores
    
    def _compute_fallback_complexity_score(
        self,
        aggregated_metrics: dict,
        all_weights: dict,
    ) -> float:
        """
        Compute a fallback complexity score when curve fitting fails.
        
        Uses the raw metric values at the largest pool size as a proxy
        for the floor parameter.
        
        Parameters
        ----------
        aggregated_metrics : dict
            Aggregated metrics from aggregate_shap_metrics().
        all_weights : dict
            Flattened metric weights.
        
        Returns
        -------
        float
            Fallback complexity score, or NaN if not computable.
        """
        # Get the last (largest pool) values
        weighted_sum = 0.0
        total_weight = 0.0
        
        for metric, weight in all_weights.items():
            if weight == 0:
                continue
            if metric not in aggregated_metrics:
                continue
            
            means = aggregated_metrics[metric].get("means", [])
            if not means:
                continue
            
            # Use the last value (largest pool)
            last_value = means[-1]
            if not np.isfinite(last_value):
                continue
            
            # Handle stability vs instability metrics
            if metric in ["rank_stability", "rank_stability_global", "direction_consistency", "topk_overlap"]:
                # Stability metric: convert to instability
                instability = 1 - last_value
                weighted_sum += weight * instability
            else:
                # Instability metric: use directly
                weighted_sum += weight * last_value
            
            total_weight += weight
            self._print(f"  Fallback: {metric} = {last_value:.4f} (weight={weight})", level=2)
        
        if total_weight > 0:
            return weighted_sum / total_weight
        else:
            return np.nan
    
    def _get_fallback_complexity_score(self, aggregated_metrics: dict) -> float:
        """
        Compute a fallback complexity score when curve fitting fails.
        
        Uses the raw metric values at the largest pool size as a proxy.
        
        Parameters
        ----------
        aggregated_metrics : dict
            Aggregated metrics from aggregate_shap_metrics().
        
        Returns
        -------
        float
            Fallback complexity score, or NaN if not computable.
        """
        self._print("Attempting fallback complexity score computation...", level=1)
        
        # Flatten weights
        all_weights = {}
        for tier, weights in self.metric_weights.items():
            all_weights.update(weights)
        
        # Get the last (largest pool) values
        total_weight = 0.0
        weighted_sum = 0.0
        
        for metric, weight in all_weights.items():
            if weight == 0:
                continue
            if metric not in aggregated_metrics:
                continue
            
            means = aggregated_metrics[metric].get("means", [])
            if not means:
                continue
            
            # Use the last value (largest pool)
            last_value = means[-1]
            if not np.isfinite(last_value):
                continue
            
            # Handle stability vs instability metrics
            if metric in ["rank_stability", "rank_stability_global", "direction_consistency", "topk_overlap"]:
                # Stability metric: convert to instability
                instability = 1 - last_value
                weighted_sum += weight * instability
            else:
                # Instability metric: use directly
                weighted_sum += weight * last_value
            
            total_weight += weight
            self._print(f"  Fallback: {metric} = {last_value:.4f} (weight={weight})", level=2)
        
        if total_weight > 0:
            fallback_score = weighted_sum / total_weight
            self._print(f"  Fallback complexity score: {fallback_score:.4f}", level=1)
            return fallback_score
        else:
            self._print("  Fallback failed: no valid metrics", level=1)
            return np.nan
    
    def _build_feature_results(
        self,
        all_pool_metrics: List[Dict],
        feature_names: List[str],
        pool_sizes: np.ndarray,
    ) -> List[Dict]:
        """Build per-feature results summary."""
        n_features = len(feature_names)
        
        # Initialize accumulators
        feature_metrics = {fname: {} for fname in feature_names}
        
        for pool_idx, metrics in enumerate(all_pool_metrics):
            if metrics is None:
                continue
            
            for metric_name, metric_result in metrics.items():
                if len(metric_result.values) == n_features:
                    for f, fname in enumerate(feature_names):
                        if metric_name not in feature_metrics[fname]:
                            feature_metrics[fname][metric_name] = []
                        feature_metrics[fname][metric_name].append(metric_result.values[f])
        
        # Aggregate per feature
        results = []
        for fname in feature_names:
            feat_result = {"feature": fname}
            for metric_name, values in feature_metrics[fname].items():
                if values:
                    feat_result[f"{metric_name}_mean"] = float(np.mean(values))
                    feat_result[f"{metric_name}_std"] = float(np.std(values, ddof=1))
            results.append(feat_result)
        
        return results
    
    def fit_panel(
        self,
        X: Union[pd.DataFrame, np.ndarray],
        y: Union[pd.Series, np.ndarray] = None,
        feature_names: list = None,
    ) -> dict:
        """
        Compute SHAP stability for panel of features.
        
        Note: Unlike marginal stability, SHAP stability is computed jointly
        for all features (since SHAP is computed for all features at once).
        This method exists for API consistency and returns the same results
        as fit() but with a summary DataFrame.
        
        Parameters
        ----------
        X : pd.DataFrame or np.ndarray
            Feature matrix.
        y : pd.Series or np.ndarray
            Target variable.
        feature_names : list, optional
            Feature names.
        
        Returns
        -------
        dict
            Results with 'feature_results' and 'summary' DataFrame.
        """
        results = self.fit(X, y, feature_names)
        
        # Get the overall complexity score from fit results
        overall_complexity = results.get("complexity_score", np.nan)
        
        # Flatten metric weights for per-feature complexity computation
        all_weights = {
            metric: w
            for tier_weights in self.metric_weights.values()
            for metric, w in tier_weights.items()
        }
        stability_metrics = {"rank_stability", "direction_consistency", "topk_overlap", "rank_stability_global"}

        # Build summary DataFrame with per-feature complexity scores
        summary_rows = []
        for feat_result in results["feature_results"]:
            # Compute a per-feature complexity proxy from per-feature stability means.
            # Uses the same weighting scheme as _compute_shap_complexity_score but
            # substitutes per-feature metric means for learning-curve floor parameters.
            total_w = 0.0
            weighted_instab = 0.0
            for metric, weight in all_weights.items():
                if weight == 0:
                    continue
                val = feat_result.get(f"{metric}_mean", np.nan)
                if not np.isfinite(val):
                    continue
                instab = (1.0 - val) if metric in stability_metrics else val
                weighted_instab += weight * instab
                total_w += weight
            feat_complexity = weighted_instab / total_w if total_w > 0 else overall_complexity

            row = {
                "feature": feat_result["feature"],
                "complexity_score": feat_complexity,
                "direction_consistency": feat_result.get("direction_consistency_mean", np.nan),
                "rank_stability": feat_result.get("rank_stability_mean", np.nan),
                "wasserstein": feat_result.get("wasserstein_mean", np.nan),
                "magnitude_cv": feat_result.get("magnitude_cv_mean", np.nan),
            }
            summary_rows.append(row)
        
        summary_df = pd.DataFrame(summary_rows)
        
        # Sort by direction consistency (descending - higher is better)
        if "direction_consistency" in summary_df.columns:
            summary_df = summary_df.sort_values(
                "direction_consistency", ascending=False
            ).reset_index(drop=True)
        
        # Log summary statistics
        if not np.isnan(overall_complexity):
            self._print(f"fit_panel() returning complexity_score: {overall_complexity:.4f}", level=1)
        else:
            self._print("WARNING: fit_panel() returning NaN complexity_score", level=1)
        
        return {
            "feature_results": results["feature_results"],
            "summary": summary_df,
            "full_results": results,
        }
