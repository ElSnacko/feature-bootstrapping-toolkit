"""Permutation baseline for calibrating bootstrap complexity scores.

Shuffles feature values to break feature-target association while preserving
the marginal distribution. Running the bootstrap analysis on permuted data
gives a null distribution of complexity scores that represents pure noise.
"""

import numpy as np
import pandas as pd
from typing import Optional, List

from .analyzer import BootstrapStability
from .core import get_complexity_score


class PermutationBaseline:
    """Build a null distribution of complexity scores via feature permutation.

    Parameters
    ----------
    n_permutations : int
        Number of permutations to run (default 25).
    analyzer_kwargs : dict, optional
        Keyword arguments passed to BootstrapStability for null runs.
        Defaults to reduced settings for speed (n_resamples=10).
    alpha : float
        Significance level for the p-value threshold (default 0.05).
    random_state : int, optional
        Random seed for reproducibility.
    verbose : int
        Verbosity level (0=silent, 1=progress).
    """

    def __init__(
        self,
        n_permutations: int = 25,
        analyzer_kwargs: dict = None,
        alpha: float = 0.05,
        random_state: int = 42,
        verbose: int = 1,
    ):
        self.n_permutations = n_permutations
        self.alpha = alpha
        self.random_state = random_state
        self.verbose = verbose

        # Default to fast settings for null runs
        defaults = dict(
            n_resamples=10,
            estimate_alpha=False,
            fixed_alpha=0.5,
            random_state=random_state,
            n_jobs=-1,
        )
        if analyzer_kwargs:
            defaults.update(analyzer_kwargs)
        self.analyzer_kwargs = defaults

    def fit(
        self,
        df: pd.DataFrame,
        feature_col: str,
        target_col: str,
        category: str = "overall",
    ) -> dict:
        """Run permutation baseline for a single feature.

        Parameters
        ----------
        df : DataFrame
            Full dataset.
        feature_col : str
            Feature column name.
        target_col : str
            Target column name.
        category : str
            Complexity score category: "overall", "target_agnostic", or
            "target_dependent".

        Returns
        -------
        dict with keys:
            observed : float — real complexity score
            null_scores : list[float] — permutation null distribution
            null_mean : float
            null_std : float
            p_value : float — fraction of null >= observed
            z_score : float — (observed - null_mean) / null_std
            significant : bool — p_value < alpha
        """
        rng = np.random.RandomState(self.random_state)

        # Observed score
        bs = BootstrapStability(**self.analyzer_kwargs)
        observed_result = bs.fit(df, feature_col=feature_col, target_col=target_col)
        observed = get_complexity_score(observed_result, category)

        # Null distribution
        null_scores = []
        for i in range(self.n_permutations):
            df_perm = df.copy()
            df_perm[feature_col] = rng.permutation(df_perm[feature_col].values)

            bs_null = BootstrapStability(**{
                **self.analyzer_kwargs,
                "random_state": self.random_state + i + 1,
            })
            null_result = bs_null.fit(df_perm, feature_col=feature_col, target_col=target_col)
            score = get_complexity_score(null_result, category)
            null_scores.append(score)

            if self.verbose >= 1:
                print(f"  {feature_col} permutation {i+1}/{self.n_permutations}: {score:.6f}")

        null_scores = [s for s in null_scores if np.isfinite(s)]
        null_mean = float(np.mean(null_scores)) if null_scores else np.nan
        null_std = float(np.std(null_scores, ddof=1)) if len(null_scores) > 1 else np.nan

        if null_scores and np.isfinite(observed):
            p_value = float(np.mean([s >= observed for s in null_scores]))
            z_score = (observed - null_mean) / null_std if null_std > 0 else np.nan
        else:
            p_value = np.nan
            z_score = np.nan

        return {
            "feature": feature_col,
            "observed": float(observed) if np.isfinite(observed) else np.nan,
            "null_scores": null_scores,
            "null_mean": null_mean,
            "null_std": null_std,
            "p_value": p_value,
            "z_score": z_score,
            "significant": p_value < self.alpha if np.isfinite(p_value) else False,
            "n_permutations": len(null_scores),
            "category": category,
        }

    def fit_panel(
        self,
        df: pd.DataFrame,
        target_col: str,
        feature_cols: Optional[List[str]] = None,
        category: str = "overall",
    ) -> dict:
        """Run permutation baseline for all features.

        Returns dict with 'results' (list of per-feature dicts) and
        'summary' (DataFrame).
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if c != target_col]

        results = []
        for feat in feature_cols:
            if self.verbose >= 1:
                print(f"Permutation baseline: {feat}")
            r = self.fit(df, feature_col=feat, target_col=target_col, category=category)
            results.append(r)

        summary = pd.DataFrame([
            {
                "feature": r["feature"],
                "observed": r["observed"],
                "null_mean": r["null_mean"],
                "null_std": r["null_std"],
                "p_value": r["p_value"],
                "z_score": r["z_score"],
                "significant": r["significant"],
            }
            for r in results
        ])

        return {"results": results, "summary": summary}
