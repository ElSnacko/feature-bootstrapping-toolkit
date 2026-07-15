"""Fix #3 — make the toolkit pip-installable with split extras.

A minimal pyproject.toml with core deps (numpy/scipy/pandas) and extras for the
optional stack. The distribution name ``bootstrap-stability`` matches the
install hints surfaced by the lazy-import errors.
"""
import re
import sys
import zipfile
from pathlib import Path

import pytest

if sys.version_info >= (3, 11):
    import tomllib
else:
    tomllib = pytest.importorskip("tomli")

REPO_ROOT = Path(__file__).resolve().parent.parent
PYPROJECT = REPO_ROOT / "pyproject.toml"


def _load_pyproject():
    assert PYPROJECT.exists(), f"pyproject.toml not found at {PYPROJECT}"
    return tomllib.loads(PYPROJECT.read_text())


def _dep_name(raw):
    """Normalize a PEP 508 dependency spec to its bare distribution name."""
    return re.split(r"[<>=!~;\s\[]", raw.strip(), 1)[0].lower()


def test_pyproject_parses():
    data = _load_pyproject()
    assert "project" in data
    proj = data["project"]
    assert proj["name"]
    assert proj["version"]


def test_distribution_name_matches_install_hints():
    # The friendly ImportError messages say: pip install 'bootstrap-stability[<extra>]'
    proj = _load_pyproject()["project"]
    assert proj["name"] == "bootstrap-stability"


def test_core_deps_are_minimal():
    proj = _load_pyproject()["project"]
    deps = {_dep_name(d) for d in proj["dependencies"]}
    assert deps == {"numpy", "scipy", "pandas"}


def test_core_deps_exclude_optional_stack():
    proj = _load_pyproject()["project"]
    deps = {_dep_name(d) for d in proj["dependencies"]}
    for forbidden in ("joblib", "matplotlib", "scikit-learn", "lightgbm", "shap"):
        assert forbidden not in deps, f"{forbidden} must be an extra, not a core dep"


def test_extras_split_matches_lazy_modules():
    extras = _load_pyproject()["project"]["optional-dependencies"]
    assert _dep_name("joblib") in {_dep_name(d) for d in extras["parallel"]}
    assert _dep_name("matplotlib") in {_dep_name(d) for d in extras["viz"]}
    assert _dep_name("scikit-learn") in {_dep_name(d) for d in extras["meta"]}
    shap_deps = {_dep_name(d) for d in extras["shap"]}
    assert "lightgbm" in shap_deps and "shap" in shap_deps


def test_version_matches_core_version():
    from bootstrap_stability.core import VERSION

    proj = _load_pyproject()["project"]
    assert proj["version"] == VERSION


def test_lazy_import_extras_resolve_to_installable_extras():
    """Every install hint the lazy loader raises at users must be a real extra.

    The ImportError messages reference extras like ``bootstrap-stability[parallel]``;
    if one of those names drifts out of pyproject.toml, users would be told to
    install an extra that doesn't exist.
    """
    import bootstrap_stability as bs

    declared = set(_load_pyproject()["project"]["optional-dependencies"])
    referenced = set(bs._LAZY_MODULES.values()) | set(bs._EXTRA_BY_MODULE.values())
    missing = referenced - declared
    assert not missing, (
        f"lazy loader hints reference undeclared extras (not in pyproject.toml): {missing}"
    )


@pytest.mark.integration
def test_package_builds_into_wheel(tmp_path):
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation",
         "-w", str(tmp_path), str(REPO_ROOT)],
        check=True,
    )
    wheels = list(tmp_path.glob("bootstrap_stability-*.whl"))
    assert wheels, "no wheel produced"
    with zipfile.ZipFile(wheels[0]) as zf:
        meta = next(n for n in zf.namelist() if n.endswith("METADATA"))
        text = zf.read(meta).decode()
    assert "Name: bootstrap-stability" in text
    from bootstrap_stability.core import VERSION

    assert f"Version: {VERSION}" in text
    assert "Requires-Dist: numpy" in text
    # optional deps must only appear behind an extra marker, never as hard requires
    hard_requires = [
        line for line in text.splitlines()
        if line.startswith("Requires-Dist:") and "extra ==" not in line
    ]
    hard_names = {_dep_name(line.split(":", 1)[1]) for line in hard_requires}
    for forbidden in ("joblib", "matplotlib", "scikit-learn", "lightgbm", "shap"):
        assert forbidden not in hard_names, f"{forbidden} is a hard requirement"
