"""Marginal-vs-SHAP validation: compare distributional and model-decision stability.

Produces a scatter plot of marginal complexity vs SHAP complexity per feature,
classifies features into quadrants (concordant stable/unstable, false alarm,
missed risk), and computes rank correlation.
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Optional


class MarginalVsSHAPValidator:
    """Compare marginal and SHAP stability scores per feature.

    Parameters
    ----------
    marginal_threshold : float or None
        Threshold for marginal complexity above which a feature is "unstable".
        If None and permutation results are supplied, uses the 95th percentile
        of the null distribution; otherwise falls back to the median of
        observed scores.
    shap_threshold : float or None
        Threshold for SHAP complexity above which a feature is "unstable".
        If None, uses the median of observed scores.
    null_percentile : float
        Percentile of the permutation null distribution used as the stability
        threshold when ``marginal_threshold`` is None and permutation results
        are available (default 95).
    """

    def __init__(
        self,
        marginal_threshold: Optional[float] = None,
        shap_threshold: Optional[float] = None,
        null_percentile: float = 95.0,
    ):
        self.marginal_threshold = marginal_threshold
        self.shap_threshold = shap_threshold
        self.null_percentile = null_percentile

    def compare(
        self,
        marginal_panel: dict,
        shap_panel: dict,
        permutation_results: Optional[dict] = None,
    ) -> dict:
        """Compare marginal and SHAP per-feature complexity.

        Parameters
        ----------
        marginal_panel : dict
            Output of BootstrapStability.fit_panel(). Must contain 'summary'
            DataFrame with 'feature' and 'complexity_score' columns.
        shap_panel : dict
            Output of SHAPStability.fit_panel(). Must contain 'summary'
            DataFrame with 'feature' and 'complexity_score' columns.
        permutation_results : dict, optional
            Output of PermutationBaseline.fit_panel(). When supplied, the
            marginal threshold is set to the ``null_percentile``-th percentile
            of pooled null scores across all features, making the "unstable"
            label mean "significantly above noise" rather than "above the
            median of this feature set".

        Returns
        -------
        dict with:
            comparison : DataFrame with per-feature marginal/SHAP scores and quadrant
            rank_correlation : float (Spearman rho)
            rank_pvalue : float
            quadrant_counts : dict mapping quadrant name to count
            marginal_threshold : float — threshold applied on the marginal axis
            shap_threshold : float — threshold applied on the SHAP axis
            threshold_source : str — "permutation_calibrated", "user_supplied", or "median"
        """
        m_summary = marginal_panel["summary"][["feature", "complexity_score"]].copy()
        m_summary = m_summary.rename(columns={"complexity_score": "marginal_complexity"})

        s_summary = shap_panel["summary"][["feature", "complexity_score"]].copy()
        s_summary = s_summary.rename(columns={"complexity_score": "shap_complexity"})

        merged = m_summary.merge(s_summary, on="feature", how="inner")
        merged = merged.dropna(subset=["marginal_complexity", "shap_complexity"])

        if len(merged) < 3:
            return {
                "comparison": merged,
                "rank_correlation": np.nan,
                "rank_pvalue": np.nan,
                "quadrant_counts": {},
            }

        # Rank correlation
        rho, pval = stats.spearmanr(
            merged["marginal_complexity"], merged["shap_complexity"]
        )

        # Marginal threshold: permutation-calibrated > user-supplied > median fallback
        m_thresh = self.marginal_threshold
        threshold_source = "user_supplied"
        if m_thresh is None:
            if permutation_results is not None:
                # Pool all null scores across features and take the requested percentile.
                all_null = []
                for r in permutation_results.get("results", []):
                    all_null.extend(r.get("null_scores", []))
                all_null = [s for s in all_null if np.isfinite(s)]
                if all_null:
                    m_thresh = float(np.percentile(all_null, self.null_percentile))
                    threshold_source = "permutation_calibrated"
            if m_thresh is None:
                m_thresh = float(merged["marginal_complexity"].median())
                threshold_source = "median"

        s_thresh = self.shap_threshold
        if s_thresh is None:
            s_thresh = float(merged["shap_complexity"].median())

        # Classify into quadrants
        def _quadrant(row):
            m_high = row["marginal_complexity"] > m_thresh
            s_high = row["shap_complexity"] > s_thresh
            if m_high and s_high:
                return "concordant_unstable"
            elif m_high and not s_high:
                return "false_alarm"
            elif not m_high and s_high:
                return "missed_risk"
            else:
                return "concordant_stable"

        merged["quadrant"] = merged.apply(_quadrant, axis=1)
        quadrant_counts = merged["quadrant"].value_counts().to_dict()

        return {
            "comparison": merged,
            "rank_correlation": float(rho),
            "rank_pvalue": float(pval),
            "marginal_threshold": m_thresh,
            "shap_threshold": s_thresh,
            "quadrant_counts": quadrant_counts,
            "threshold_source": threshold_source,
        }


def plot_marginal_vs_shap(comparison_result: dict, save_path: str = None):
    """Scatter plot of marginal vs SHAP complexity with quadrant labels.

    Parameters
    ----------
    comparison_result : dict
        Output of MarginalVsSHAPValidator.compare().
    save_path : str, optional
        Path to save the figure.

    Returns
    -------
    matplotlib Figure
    """
    import matplotlib.pyplot as plt

    df = comparison_result["comparison"]
    rho = comparison_result["rank_correlation"]
    m_thresh = comparison_result["marginal_threshold"]
    s_thresh = comparison_result["shap_threshold"]
    threshold_source = comparison_result.get("threshold_source", "median")

    quadrant_colors = {
        "concordant_unstable": "#D85A30",
        "false_alarm": "#F5A623",
        "missed_risk": "#9B59B6",
        "concordant_stable": "#378ADD",
    }
    quadrant_labels = {
        "concordant_unstable": "Concordant unstable",
        "false_alarm": "False alarm",
        "missed_risk": "Missed risk",
        "concordant_stable": "Concordant stable",
    }

    fig, ax = plt.subplots(figsize=(8, 7), dpi=100)

    for quad, color in quadrant_colors.items():
        mask = df["quadrant"] == quad
        subset = df[mask]
        if len(subset) > 0:
            ax.scatter(
                subset["marginal_complexity"],
                subset["shap_complexity"],
                c=color, label=quadrant_labels[quad],
                s=60, alpha=0.85, edgecolors="white", linewidth=0.5,
            )
            for _, row in subset.iterrows():
                ax.annotate(
                    row["feature"], (row["marginal_complexity"], row["shap_complexity"]),
                    fontsize=7, ha="left", va="bottom", xytext=(3, 3),
                    textcoords="offset points",
                )

    # Threshold lines
    ax.axvline(m_thresh, color="gray", ls="--", lw=0.8, alpha=0.6)
    ax.axhline(s_thresh, color="gray", ls="--", lw=0.8, alpha=0.6)

    source_label = {
        "permutation_calibrated": "permutation-calibrated",
        "user_supplied": "user-supplied",
        "median": "median",
    }.get(threshold_source, threshold_source)
    ax.set_xlabel(f"Marginal Complexity Score  [threshold: {source_label}]")
    ax.set_ylabel("SHAP Complexity Score")
    ax.set_title(f"Marginal vs SHAP Stability (Spearman ρ = {rho:.3f})")
    ax.legend(loc="best", fontsize=8)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig
