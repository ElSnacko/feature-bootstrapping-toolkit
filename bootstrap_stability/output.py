import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


COLORS = {
    "wasserstein": "#378ADD",
    "ks":          "#639922",
    "js":          "#D85A30",
    "spearman":    "#7F77DD",
    "iv":          "#BA7517",
    "monotonicity":"#1D9E75",
    "p10": "#D85A30",
    "p25": "#BA7517",
    "p50": "#2C2C2A",
    "p75": "#378ADD",
    "p90": "#7F77DD",
}

METRIC_LABELS = {
    "wasserstein": "Wasserstein",
    "ks":          "KS",
    "js":          "JS divergence",
    "spearman":    "Spearman ρ",
    "iv":          "IV",
    "monotonicity":"Monotonicity",
}

PERCENTILE_LABELS = {
    "p10": "10th", "p25": "25th", "p50": "50th",
    "p75": "75th", "p90": "90th",
}


def plot_results(results, save_path=None, figsize=(15, 11), dpi=150):
    meta = results["meta"]
    lc = results["learning_curves"]
    pool_sizes = results["pool_sequence"]
    has_target = meta["has_target"]
    feature = meta["feature"]
    complexity = results["complexity_score"]
    version = meta["version"]

    fig = plt.figure(figsize=figsize, dpi=dpi)
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.38, wspace=0.32)

    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1])

    # Panel 1: learning curves
    dist_metrics = ["wasserstein", "ks", "js"]
    td_metrics = ["spearman"] if has_target else []
    plot_metrics = dist_metrics + td_metrics

    r2_annotations = []
    for metric in plot_metrics:
        curve = lc.get(metric, {})
        means = np.array(curve.get("means", []), dtype=float)
        stderr = np.array(curve.get("stderr", []), dtype=float)
        fit = curve.get("fit", {})
        color = COLORS[metric]
        label = METRIC_LABELS[metric]

        valid = ~np.isnan(means)
        if not np.any(valid):
            continue

        ps = np.array(pool_sizes)[valid]
        m = means[valid]
        se = stderr[valid]

        ax1.plot(ps, m, color=color, label=label, linewidth=1.8)
        ax1.fill_between(ps, m - se, m + se, color=color, alpha=0.15)

        if not fit.get("fit_failed") and np.isfinite(fit.get("k", np.nan)):
            n_smooth = np.linspace(ps.min(), ps.max(), 200)
            fitted_vals = fit["k"] / np.sqrt(n_smooth) + fit["floor"]
            ax1.plot(n_smooth, fitted_vals, color=color, linestyle="--", linewidth=1.0, alpha=0.7)
            if np.isfinite(fit.get("floor", np.nan)):
                ax1.axhline(fit["floor"], color=color, linestyle=":", linewidth=0.8, alpha=0.5)

        r2_val = fit.get("r2", np.nan)
        r2_str = f"{r2_val:.3f}" if np.isfinite(r2_val) else "n/a"
        r2_annotations.append(f"{label}: R²={r2_str}")

    ax1.set_xlabel("Pool size")
    ax1.set_ylabel("Instability")
    ax1.set_title("Learning Curves")
    ax1.legend(fontsize=8)
    if r2_annotations:
        ax1.text(
            0.02, 0.04, "\n".join(r2_annotations),
            transform=ax1.transAxes, fontsize=7,
            verticalalignment="bottom", family="monospace",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7)
        )

    # Panel 2: floor decomposition
    floor_metrics = ["wasserstein", "ks", "js"]
    if has_target:
        floor_metrics += ["spearman", "iv", "monotonicity"]

    floors = []
    bar_labels = []
    bar_colors = []
    for metric in floor_metrics:
        fit = lc.get(metric, {}).get("fit", {})
        floor = fit.get("floor", np.nan)
        if np.isfinite(floor):
            floors.append(floor)
            bar_labels.append(METRIC_LABELS[metric])
            bar_colors.append(COLORS[metric])

    if floors:
        y_pos = range(len(floors))
        bars = ax2.barh(y_pos, floors, color=bar_colors, alpha=0.85)
        ax2.set_yticks(list(y_pos))
        ax2.set_yticklabels(bar_labels)
        ax2.set_xlabel("Floor value")
        for bar, val in zip(bars, floors):
            ax2.text(
                max(bar.get_width(), 0) + ax2.get_xlim()[1] * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}", va="center", fontsize=8
            )

    score_str = f"{complexity:.4f}" if np.isfinite(complexity) else "n/a"
    ax2.set_title(f"Floor Decomposition\nComplexity score: {score_str}")

    # Panel 3: WOE profile stability
    woe_profiles = results.get("woe_profiles", {})
    if has_target and woe_profiles:
        bins = sorted(woe_profiles.keys())
        bin_labels = [b.replace("_", " ").title() for b in bins]
        mean_woes = [woe_profiles[b]["mean_woe"] for b in bins]
        sd_woes = [woe_profiles[b]["sd_woe"] for b in bins]
        flip_rates = [woe_profiles[b]["sign_flip_rate"] for b in bins]

        x_pos = np.arange(len(bins))
        ax3.bar(x_pos, mean_woes, yerr=sd_woes, color="#378ADD", alpha=0.75,
                error_kw={"capsize": 4, "elinewidth": 1.2})
        ax3.axhline(0, color="black", linewidth=0.8)

        for i, flip in enumerate(flip_rates):
            ax3.text(i, max(mean_woes[i], 0) + max(sd_woes[i], 0) * 0.1 + 0.01,
                     f"{flip:.0%}", ha="center", fontsize=7, color="#D85A30")

        ax3.set_xticks(x_pos)
        ax3.set_xticklabels(bin_labels, fontsize=8)
        ax3.set_ylabel("Mean WOE")
        ax3.set_title("WOE Profile Stability\n(flip rate above bars)")
    else:
        ax3.text(0.5, 0.5, "No target", transform=ax3.transAxes,
                 ha="center", va="center", fontsize=14, color="gray")
        ax3.set_title("WOE Profile Stability")

    # Panel 4: percentile stability
    pct_stability = results.get("percentile_stability", {})
    pct_keys = ["p10", "p25", "p50", "p75", "p90"]
    for pct in pct_keys:
        vals = pct_stability.get(pct, [])
        if not vals:
            continue
        lw = 2.0 if pct == "p50" else 1.2
        ax4.plot(pool_sizes[:len(vals)], vals,
                 color=COLORS[pct], label=PERCENTILE_LABELS[pct],
                 linewidth=lw)

    ax4.set_xlabel("Pool size")
    ax4.set_ylabel("Feature value")
    ax4.set_title("Percentile Stability")
    ax4.legend(fontsize=8)

    # Main title
    title_parts = [f"'{feature}'"]
    score_str = f"{complexity:.4f}" if np.isfinite(complexity) else "n/a"
    title_parts.append(f"complexity={score_str}")
    title_parts.append(f"v{version}")
    warnings = []
    if meta.get("censoring_flag"):
        warnings.append("CENSORING")
    if meta.get("imbalance_flag"):
        warnings.append("IMBALANCE")

    title = "Bootstrap Stability — " + " | ".join(title_parts)
    if warnings:
        fig.suptitle(title, fontsize=12, y=1.01)
        fig.text(0.5, 0.995, "  ".join(warnings), ha="center",
                 fontsize=10, color="red", fontweight="bold")
    else:
        fig.suptitle(title, fontsize=12)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def plot_panel(panel_results, save_path=None, top_n=30, figsize=(12, 8), dpi=150):
    summary = panel_results.get("summary", pd.DataFrame())
    if summary.empty:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        ax.text(0.5, 0.5, "No results", ha="center", va="center")
        return fig

    df = summary.dropna(subset=["complexity_score"]).head(top_n)
    features = df["feature"].tolist()
    scores = df["complexity_score"].tolist()
    censoring = df["censoring_flag"].tolist() if "censoring_flag" in df.columns else [False] * len(df)

    bar_colors = ["#D85A30" if c else "#378ADD" for c in censoring]

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    y_pos = np.arange(len(features))
    ax.barh(y_pos, scores, color=bar_colors, alpha=0.85)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9)
    ax.set_xlabel("Complexity Score")
    ax.set_title("Feature Complexity Scores (ascending)\nBlue = clean, Red = censoring detected")

    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="#378ADD", alpha=0.85, label="No censoring"),
        Patch(facecolor="#D85A30", alpha=0.85, label="Censoring detected"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    if save_path:
        fig.savefig(save_path, bbox_inches="tight")

    return fig


def print_report(results) -> None:
    meta = results["meta"]
    lc = results["learning_curves"]
    has_target = meta["has_target"]
    woe_profiles = results.get("woe_profiles", {})

    W = 60
    print("=" * W)
    print(f"Bootstrap Stability Report — '{meta['feature']}'")
    print(f"Version: {meta['version']}  |  {meta['run_timestamp']}")
    print("=" * W)

    print(f"{'Observations':<20}: {meta['n_obs']}")
    print(f"{'Feature type':<20}: {meta['feature_type']}")
    if has_target:
        er = meta.get("event_rate", np.nan)
        er_str = f"{er:.4f}" if np.isfinite(er) else "n/a"
        print(f"{'Event rate':<20}: {er_str}")
    imb_str = "Yes" if meta.get("imbalance_flag") else "No"
    cen_str = "Yes" if meta.get("censoring_flag") else "No"
    print(f"{'Imbalance flag':<20}: {imb_str}")
    print(f"{'Censoring flag':<20}: {cen_str}")
    excl = results.get("excluded_pools", [])
    if excl:
        print(f"{'Excluded pools':<20}: {excl}")

    print()
    cs = results["complexity_score"]
    cs_str = f"{cs:.4f}" if np.isfinite(cs) else "n/a"
    print(f"Complexity score : {cs_str}")
    print()

    weight_tiers = {
        "wasserstein": "[primary]   ",
        "ks":          "[secondary] ",
        "js":          "[secondary] ",
        "spearman":    "[primary]   ",
        "iv":          "[secondary] ",
        "monotonicity":"[secondary] ",
    }

    print(f"{'Metric':<28} {'Floor':>10} {'R²':>8}   {'Anomalous':>9}")
    print("-" * 62)

    metric_order = ["wasserstein", "ks", "js"]
    if has_target:
        metric_order += ["spearman", "iv", "monotonicity"]

    for metric in metric_order:
        curve = lc.get(metric, {})
        fit = curve.get("fit", {})
        floor = fit.get("floor", np.nan)
        r2 = fit.get("r2", np.nan)
        anomalous = fit.get("anomalous", True)
        failed = fit.get("fit_failed", True)

        label = f"{METRIC_LABELS[metric]:<14}{weight_tiers.get(metric, '')}"
        floor_str = f"{floor:.4f}" if np.isfinite(floor) else "  n/a  "
        r2_str = f"{r2:.3f}" if np.isfinite(r2) else "  n/a"
        anom_str = "yes" if anomalous else "no"
        if failed:
            anom_str = "failed"

        print(f"{label:<28} {floor_str:>10} {r2_str:>8}   {anom_str:>9}")

    # Extrapolation block
    extrap_metrics = ["wasserstein"]
    if has_target:
        extrap_metrics.append("spearman")

    print()
    print("Extrapolations:")
    for metric in extrap_metrics:
        fit = lc.get(metric, {}).get("fit", {})
        extraps = fit.get("extrapolations", {})
        if not extraps:
            continue
        parts = [f"n={n}: {v:.4f}" if np.isfinite(v) else f"n={n}: n/a" for n, v in sorted(extraps.items())]
        print(f"  {METRIC_LABELS[metric]:<20}: {', '.join(parts)}")

    if has_target and woe_profiles:
        print()
        print(f"{'Bin':<10} {'Mean WOE':>10} {'SD WOE':>10} {'Flip %':>8}  Status")
        print("-" * 50)
        for bin_name in sorted(woe_profiles.keys()):
            bp = woe_profiles[bin_name]
            mean_w = bp["mean_woe"]
            sd_w = bp["sd_woe"]
            flip = bp["sign_flip_rate"]

            if sd_w < 0.15 and flip < 0.10:
                status = "stable"
            elif flip > 0.30:
                status = "unstable"
            else:
                status = "noisy"

            print(
                f"{bin_name:<10} {mean_w:>10.4f} {sd_w:>10.4f} {flip:>7.1%}  {status}"
            )

    if meta.get("censoring_flag"):
        print()
        print("WARNING: Censoring detected — " + meta.get("censoring_detail", ""))
        print("  Feature stability may be inflated by policy truncation.")

    print("=" * W)


def to_csv(results, save_path) -> pd.DataFrame:
    meta = results["meta"]
    lc = results["learning_curves"]
    pool_sizes = results["pool_sequence"]
    degenerate_rates = results.get("degenerate_rates", {})
    extrapolate_to = []

    for metric_data in lc.values():
        fit = metric_data.get("fit", {})
        extraps = fit.get("extrapolations", {})
        if extraps:
            extrapolate_to = sorted(extraps.keys())
            break

    all_metrics = list(lc.keys())

    rows = []
    for i, ps in enumerate(pool_sizes):
        row = {"pool_size": ps, "degenerate_rate": degenerate_rates.get(ps, np.nan)}
        for metric in all_metrics:
            curve = lc.get(metric, {})
            means = curve.get("means", [])
            stderrs = curve.get("stderr", [])
            fit = curve.get("fit", {})

            row[f"{metric}_mean"] = means[i] if i < len(means) else np.nan
            row[f"{metric}_stderr"] = stderrs[i] if i < len(stderrs) else np.nan
            row[f"{metric}_floor"] = fit.get("floor", np.nan)
            row[f"{metric}_k"] = fit.get("k", np.nan)
            row[f"{metric}_r2"] = fit.get("r2", np.nan)
            row[f"{metric}_anomalous"] = fit.get("anomalous", True)
            for n in extrapolate_to:
                extrap_val = fit.get("extrapolations", {}).get(n, np.nan)
                row[f"{metric}_extrap_{n}"] = extrap_val
        rows.append(row)

    df = pd.DataFrame(rows)

    meta_comments = [
        f"# feature: {meta['feature']}",
        f"# target: {meta['target']}",
        f"# version: {meta['version']}",
        f"# n_obs: {meta['n_obs']}",
        f"# event_rate: {meta.get('event_rate', 'n/a')}",
        f"# imbalance_flag: {meta.get('imbalance_flag', False)}",
        f"# censoring_flag: {meta.get('censoring_flag', False)}",
        f"# run_timestamp: {meta.get('run_timestamp', '')}",
        f"# complexity_score: {results.get('complexity_score', np.nan)}",
    ]

    with open(save_path, "w") as f:
        for line in meta_comments:
            f.write(line + "\n")
        df.to_csv(f, index=False)

    return df


def panel_to_csv(panel_results, save_path) -> pd.DataFrame:
    summary = panel_results.get("summary", pd.DataFrame())
    summary.to_csv(save_path, index=False)
    return summary
