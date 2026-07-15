"""Fix #1 (lazy imports) + Fix #2 (export core primitives).

The package's pure-math core (core.py) only needs numpy/scipy/pandas, but the
old __init__ eagerly imported every submodule, so ``import bootstrap_stability``
failed whenever joblib/matplotlib/sklearn were absent. These tests pin the
contract that the package imports with only the core dependency stack and that
the reusable primitives are part of the public surface.
"""
import sys

import pytest


def _purge_package():
    """Remove the package and any cached submodules from sys.modules."""
    for mod in list(sys.modules):
        if mod == "bootstrap_stability" or mod.startswith("bootstrap_stability."):
            del sys.modules[mod]


def _purge_optional():
    """Drop optional dependency modules so import-side effects are observable."""
    for name in list(sys.modules):
        top = name.split(".")[0]
        if top in {"joblib", "matplotlib", "sklearn", "shap", "lightgbm"}:
            del sys.modules[name]


# --------------------------------------------------------------------------- #
# Fix #1: package imports without the optional stack
# --------------------------------------------------------------------------- #
def test_import_succeeds_with_only_core_deps():
    _purge_package()
    _purge_optional()
    import bootstrap_stability

    assert bootstrap_stability is not None
    assert bootstrap_stability.__version__


def test_import_does_not_pull_optional_deps():
    _purge_package()
    _purge_optional()
    before = set(sys.modules)
    import bootstrap_stability  # noqa: F401

    added = set(sys.modules) - before
    for forbidden in ("joblib", "matplotlib", "sklearn", "shap", "lightgbm"):
        leaked = [m for m in added if m == forbidden or m.startswith(forbidden + ".")]
        assert not leaked, (
            f"importing bootstrap_stability pulled in optional dep '{forbidden}': {leaked}"
        )


def test_core_submodule_imports_with_only_core_deps():
    _purge_package()
    _purge_optional()
    from bootstrap_stability import core

    assert hasattr(core, "fit_learning_curve")


def test_optional_component_without_extra_raises_clear_error(monkeypatch):
    _purge_package()
    _purge_optional()
    import bootstrap_stability as bs

    # Ensure analyzer (joblib-backed) is not yet loaded, then make joblib
    # unresolvable so the lazy loader has to surface a helpful error.
    monkeypatch.setitem(sys.modules, "joblib", None)
    with pytest.raises(ImportError) as excinfo:
        _ = bs.BootstrapStability
    msg = str(excinfo.value).lower()
    assert "pip install" in msg or "bootstrap-stability[" in msg


# --------------------------------------------------------------------------- #
# Fix #2: core primitives are exported on the public surface
# --------------------------------------------------------------------------- #
PRIMITIVES = [
    "fit_learning_curve",
    "generate_pool_sequence",
    "draw_pool",
    "bootstrap_resample",
    "run_bootstrap_on_pool",
    "bootstrap_ci",
]


def test_core_primitives_importable_from_top_level():
    _purge_package()
    _purge_optional()
    import bootstrap_stability as bs

    for name in PRIMITIVES:
        assert hasattr(bs, name), f"{name} is not exported from bootstrap_stability"


def test_core_primitives_are_identical_to_core_module():
    _purge_package()
    _purge_optional()
    import bootstrap_stability as bs
    from bootstrap_stability import core

    for name in PRIMITIVES:
        exported = getattr(bs, name)
        direct = getattr(core, name)
        assert exported is direct, f"{name} export is not the core object"


def test_primitives_listed_in_all():
    _purge_package()
    _purge_optional()
    import bootstrap_stability as bs

    for name in PRIMITIVES:
        assert name in bs.__all__, f"{name} missing from __all__"
