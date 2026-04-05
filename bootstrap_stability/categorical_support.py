"""
Categorical Feature Support Module

This module provides utilities for detecting and handling categorical features
in machine learning models for the bootstrap stability analysis.
"""

from typing import Any, Dict, List, Optional, Set

# Models with native categorical feature support
# These models can handle categorical features directly without encoding
MODELS_WITH_NATIVE_CATEGORICAL: Dict[str, Set[str]] = {
    # LightGBM has native categorical support via categorical_feature parameter
    "lightgbm": {"LGBMClassifier", "LGBMRegressor", "Booster"},
    # CatBoost has native categorical support
    "catboost": {"CatBoostClassifier", "CatBoostRegressor"},
    # XGBoost has experimental categorical support (from version 1.6+)
    "xgboost": {"XGBClassifier", "XGBRegressor"},
    # H2O models
    "h2o": {"H2OGradientBoostingEstimator", "H2ORandomForestEstimator", "H2ODeepLearningEstimator"},
}

# Models that require encoding for categorical features
MODELS_REQUIRING_ENCODING: Set[str] = {
    "RandomForestClassifier",
    "RandomForestRegressor",
    "GradientBoostingClassifier",
    "GradientBoostingRegressor",
    "DecisionTreeClassifier",
    "DecisionTreeRegressor",
    "ExtraTreesClassifier",
    "ExtraTreesRegressor",
    "LogisticRegression",
    "LinearRegression",
    "Ridge",
    "Lasso",
    "ElasticNet",
    "SVC",
    "SVR",
    "KNeighborsClassifier",
    "KNeighborsRegressor",
    "MLPClassifier",
    "MLPRegressor",
}


def supports_categorical(model: Any) -> bool:
    """Check if a model supports categorical features natively.
    
    Parameters
    ----------
    model : Any
        A fitted or unfitted model instance.
    
    Returns
    -------
    bool
        True if the model has native categorical feature support.
    
    Examples
    --------
    >>> from lightgbm import LGBMClassifier
    >>> model = LGBMClassifier()
    >>> supports_categorical(model)
    True
    
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> model = RandomForestClassifier()
    >>> supports_categorical(model)
    False
    """
    if model is None:
        return False
    
    model_class_name = model.__class__.__name__
    module_name = model.__class__.__module__.split(".")[0]
    
    # Check if the model is in the native categorical support list
    if module_name in MODELS_WITH_NATIVE_CATEGORICAL:
        if model_class_name in MODELS_WITH_NATIVE_CATEGORICAL[module_name]:
            return True
    
    # Special handling for LightGBM - check for categorical_feature parameter
    if module_name == "lightgbm" or "lightgbm" in str(type(model)).lower():
        return True
    
    # Special handling for CatBoost
    if module_name == "catboost" or "catboost" in str(type(model)).lower():
        return True
    
    # Check for models that have explicit categorical feature handling
    # via fit parameters or attributes
    if hasattr(model, "fit"):
        try:
            import inspect
            sig = inspect.signature(model.fit)
            if "categorical_feature" in sig.parameters:
                return True
            if "cat_features" in sig.parameters:
                return True
        except (ValueError, TypeError):
            pass
    
    # Check for cat_features attribute (CatBoost style)
    if hasattr(model, "cat_features") or hasattr(model, "categorical_feature"):
        return True
    
    return False


def get_categorical_feature_indices(
    X,
    categorical_features: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> Optional[List[int]]:
    """Get indices of categorical features.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    categorical_features : list of str, optional
        List of categorical feature names.
    feature_names : list of str, optional
        List of all feature names. If None, will try to extract from X.
    
    Returns
    -------
    list of int or None
        Indices of categorical features, or None if not specified.
    """
    if categorical_features is None:
        return None
    
    if feature_names is None:
        if hasattr(X, "columns"):
            feature_names = list(X.columns)
        else:
            # Cannot determine feature names from numpy array
            return None
    
    indices = []
    for cat_feat in categorical_features:
        if isinstance(cat_feat, int):
            indices.append(cat_feat)
        elif cat_feat in feature_names:
            indices.append(feature_names.index(cat_feat))
    
    return indices if indices else None


def prepare_categorical_for_model(
    model: Any,
    X,
    categorical_features: Optional[List[str]] = None,
    feature_names: Optional[List[str]] = None,
) -> tuple:
    """Prepare categorical features for model training.
    
    For models with native categorical support (LightGBM, CatBoost),
    returns the data as-is with categorical indices.
    
    For models requiring encoding, returns a note that encoding is needed.
    
    Parameters
    ----------
    model : Any
        The model to prepare data for.
    X : array-like
        Feature matrix.
    categorical_features : list of str, optional
        List of categorical feature names.
    feature_names : list of str, optional
        List of all feature names.
    
    Returns
    -------
    tuple
        (X, categorical_indices, needs_encoding)
        - X: The feature matrix (possibly modified)
        - categorical_indices: Indices of categorical features
        - needs_encoding: Whether encoding is required before training
    """
    cat_indices = get_categorical_feature_indices(X, categorical_features, feature_names)
    
    if supports_categorical(model):
        return X, cat_indices, False
    else:
        # Model requires encoding
        return X, cat_indices, True
