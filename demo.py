"""
Bootstrap stability demo using sklearn breast cancer dataset as a credit risk proxy.
Malignant = event (1), benign = non-event (0).
"""
import pandas as pd
from sklearn.datasets import load_breast_cancer

from bootstrap_stability import (
    BootstrapStability,
    plot_results,
    plot_panel,
    print_report,
    to_csv,
    panel_to_csv,
)


def main():
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    # Malignant=0 in sklearn, so flip to make it the event
    df["target"] = (data.target == 0).astype(int)

    bs = BootstrapStability(n_resamples=20, random_state=42)

    print("\n" + "=" * 60)
    print("1. Deep dive: 'mean radius' (strong stable predictor)")
    print("=" * 60)
    results_radius = bs.fit(df, "mean radius", target_col="target")
    print_report(results_radius)
    fig1 = plot_results(results_radius, save_path="mean_radius_stability.png")
    fig1.clf()
    to_csv(results_radius, "mean_radius_stability.csv")
    print("Saved: mean_radius_stability.png, mean_radius_stability.csv")

    print("\n" + "=" * 60)
    print("2. Deep dive: 'symmetry error' (weak, noisy)")
    print("=" * 60)
    results_sym = bs.fit(df, "symmetry error", target_col="target")
    print_report(results_sym)
    fig2 = plot_results(results_sym, save_path="symmetry_error_stability.png")
    fig2.clf()
    print("Saved: symmetry_error_stability.png")

    print("\n" + "=" * 60)
    print("3. No-target mode: 'mean radius' (distributional only)")
    print("=" * 60)
    results_notarget = bs.fit(df, "mean radius", target_col=None)
    print_report(results_notarget)
    fig3 = plot_results(results_notarget, save_path="mean_radius_notarget.png")
    fig3.clf()
    print("Saved: mean_radius_notarget.png")

    print("\n" + "=" * 60)
    print("4. Panel analysis (n_resamples=10 for speed)")
    print("=" * 60)
    bs_panel = BootstrapStability(n_resamples=10, random_state=42)
    panel = bs_panel.fit_panel(df, target_col="target")
    print("\nPanel Summary:")
    print(panel["summary"][["feature", "complexity_score", "censoring_flag", "wasserstein_floor"]].to_string(index=False))
    fig4 = plot_panel(panel, save_path="panel_complexity.png")
    fig4.clf()
    panel_to_csv(panel, "panel_summary.csv")
    print("\nSaved: panel_complexity.png, panel_summary.csv")

    print("\n" + "=" * 60)
    print("Validation checks:")
    radius_score = results_radius["complexity_score"]
    sym_score = results_sym["complexity_score"]
    print(f"  'mean radius' complexity : {radius_score:.4f}")
    print(f"  'symmetry error' complexity: {sym_score:.4f}")
    if radius_score < sym_score:
        print("  PASS: mean radius < symmetry error complexity (as expected)")
    else:
        print("  NOTE: scores are close or reversed — check curve fits")

    notarget_lc = results_notarget["learning_curves"]
    has_dist = all(
        len(notarget_lc[m]["means"]) > 0
        for m in ["wasserstein", "ks", "js"]
    )
    td_means = [
        v for m in ["spearman", "iv"]
        for v in notarget_lc[m]["means"]
    ]
    print(f"  No-target distributional metrics present: {has_dist}")
    print(f"  No-target target metrics all None/empty: {all(v != v for v in td_means) or not td_means}")

    n_panel_features = len(panel["feature_results"])
    n_expected = len(df.select_dtypes(include="number").columns) - 1  # minus target
    print(f"  Panel ran on {n_panel_features}/{n_expected} features (skips = {n_expected - n_panel_features})")


if __name__ == "__main__":
    main()
