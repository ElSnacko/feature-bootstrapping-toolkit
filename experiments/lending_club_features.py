#!/usr/bin/env python3
"""
Lending Club Feature Engineering — Fintech Behavioral Metrics
=============================================================

Builds features from the Accepted 2007-2018 Lending Club dataset that
mirror what a fintech bank's credit risk team would actually construct.
Heavy emphasis on *behavioral* signals: credit-seeking velocity, utilisation
dynamics, payment discipline patterns, and balance-management behaviour.

Features are grouped into six themes:

    1. Credit-seeking velocity    — how aggressively the borrower is shopping
    2. Utilisation dynamics       — revolving & instalment balance management
    3. Payment discipline         — delinquency recency/severity patterns
    4. Account lifecycle          — seasoning, diversification, turnover
    5. Capacity stress            — leverage, debt-to-available signals
    6. Loan-level risk            — the loan request itself relative to profile

Target: binary default (Charged Off = 1, Fully Paid = 0).
"""

import os
import numpy as np
import pandas as pd


# ── target definition ─────────────────────────────────────────────────
def define_target(df):
    """Binary default: keep only terminal statuses."""
    mask = df["loan_status"].isin(["Fully Paid", "Charged Off"])
    df = df[mask].copy()
    df["default"] = (df["loan_status"] == "Charged Off").astype(int)
    return df


# ── helpers ───────────────────────────────────────────────────────────
def _safe_div(a, b, fill=0.0):
    """Element-wise a/b, returning fill where b==0 or result is inf/nan."""
    with np.errstate(divide="ignore", invalid="ignore"):
        r = a / b
    r = r.replace([np.inf, -np.inf], np.nan).fillna(fill)
    return r


def _clip_outliers(s, lo=0.01, hi=0.99):
    """Winsorise to [lo, hi] percentiles."""
    low, high = s.quantile([lo, hi])
    return s.clip(low, high)


def _parse_emp_length(s):
    """Convert emp_length string to numeric years."""
    s = s.astype(str)
    s = s.str.replace(r"\+ years?", "", regex=True)
    s = s.str.replace(r" years?", "", regex=True)
    s = s.str.replace("< 1", "0", regex=False)
    return pd.to_numeric(s, errors="coerce")


# ── feature builders (one per theme) ─────────────────────────────────

def _credit_seeking_velocity(df):
    """How aggressively is the borrower shopping for credit?"""
    out = pd.DataFrame(index=df.index)

    # Recent inquiry intensity
    out["inq_last_6m"] = df["inq_last_6mths"].fillna(0)
    out["inq_last_12m"] = df["inq_last_12m"].fillna(0)

    # Inquiry acceleration: ratio of 6m to 12m inquiries
    # High ratio = credit seeking is *accelerating*
    out["inq_acceleration"] = _safe_div(
        out["inq_last_6m"], out["inq_last_12m"].clip(lower=1)
    )

    # New accounts opened in last 24m relative to total accounts
    # High = rapid portfolio expansion
    out["acct_expansion_rate_24m"] = _safe_div(
        df["acc_open_past_24mths"].fillna(0),
        df["total_acc"].clip(lower=1),
    )

    # Months since most recent inquiry (lower = more active shopping)
    out["mths_since_recent_inq"] = df["mths_since_recent_inq"].fillna(24)

    # New revolving accounts in last 12m
    out["new_rev_12m"] = df["open_rv_12m"].fillna(0)

    # New instalment accounts in last 12m
    out["new_il_12m"] = df["open_il_12m"].fillna(0)

    # Inquiry-to-new-account conversion: inquiries that led to opened accounts
    total_new = out["new_rev_12m"] + out["new_il_12m"]
    out["inq_to_acct_ratio"] = _safe_div(
        out["inq_last_12m"], total_new.clip(lower=1)
    )

    return out


def _utilisation_dynamics(df):
    """Revolving & instalment balance management behaviour."""
    out = pd.DataFrame(index=df.index)

    # Overall revolving utilisation
    out["revol_util"] = pd.to_numeric(
        df["revol_util"].astype(str).str.replace("%", ""),
        errors="coerce",
    ).fillna(50.0)

    # BC (bankcard) utilisation
    out["bc_util"] = df["bc_util"].fillna(df["bc_util"].median())

    # Fraction of bankcard lines above 75% utilisation
    out["pct_bc_gt75"] = df["percent_bc_gt_75"].fillna(0)

    # Available-to-balance ratio on bank cards (liquidity buffer)
    out["bc_liquidity_buffer"] = _safe_div(
        df["bc_open_to_buy"].fillna(0),
        df["total_bc_limit"].clip(lower=1),
    )

    # Instalment utilisation
    out["il_util"] = df["il_util"].fillna(df["il_util"].median())

    # All-product utilisation
    out["all_util"] = df["all_util"].fillna(df["all_util"].median())

    # Balance concentration: revolving balance as fraction of total
    total_bal = df["tot_cur_bal"].clip(lower=1)
    out["revol_bal_concentration"] = _safe_div(df["revol_bal"].fillna(0), total_bal)

    # Instalment balance concentration
    out["il_bal_concentration"] = _safe_div(
        df["total_bal_il"].fillna(0), total_bal
    )

    # Utilisation gap: difference between all-product util and BC util
    # Positive = revolving cards are better managed than rest of portfolio
    out["util_gap_bc_vs_all"] = out["all_util"] - out["bc_util"]

    return out


def _payment_discipline(df):
    """Delinquency recency, severity, and consistency patterns."""
    out = pd.DataFrame(index=df.index)

    # Delinquency count in last 2 years
    out["delinq_2yrs"] = df["delinq_2yrs"].fillna(0)

    # Recency of last delinquency (missing = never delinquent → large value)
    out["mths_since_last_delinq"] = df["mths_since_last_delinq"].fillna(999)

    # Recency of last public record
    out["mths_since_last_record"] = df["mths_since_last_record"].fillna(999)

    # Recency of last major derogatory
    out["mths_since_last_major_derog"] = df["mths_since_last_major_derog"].fillna(999)

    # Ever 120+ days past due (count)
    out["num_accts_ever_120pd"] = df["num_accts_ever_120_pd"].fillna(0)

    # Current 30/90 DPD counts
    out["num_tl_30dpd"] = df["num_tl_30dpd"].fillna(0)
    out["num_tl_90g_dpd_24m"] = df["num_tl_90g_dpd_24m"].fillna(0)

    # Percent of tradelines never delinquent
    out["pct_tl_nvr_dlq"] = df["pct_tl_nvr_dlq"].fillna(100.0)

    # Public records + bankruptcies (severity)
    out["pub_rec"] = df["pub_rec"].fillna(0)
    out["pub_rec_bankruptcies"] = df["pub_rec_bankruptcies"].fillna(0)

    # Collections in last 12 months
    out["collections_12m"] = df["collections_12_mths_ex_med"].fillna(0)

    # Delinquency intensity: delinq_2yrs normalised by number of accounts
    out["delinq_intensity"] = _safe_div(
        out["delinq_2yrs"], df["open_acc"].clip(lower=1)
    )

    # Clean payment track: binary — zero delinquencies AND 100% never-dlq
    out["clean_payment_track"] = (
        (out["delinq_2yrs"] == 0)
        & (out["pct_tl_nvr_dlq"] >= 99.0)
        & (out["pub_rec"] == 0)
    ).astype(int)

    return out


def _account_lifecycle(df):
    """Seasoning, diversification, and account turnover."""
    out = pd.DataFrame(index=df.index)

    # Age of oldest instalment account (months)
    out["mo_sin_old_il"] = df["mo_sin_old_il_acct"].fillna(0)

    # Age of oldest revolving tradeline
    out["mo_sin_old_rev"] = df["mo_sin_old_rev_tl_op"].fillna(0)

    # Months since most recent account opening
    out["mo_sin_rcnt_tl"] = df["mo_sin_rcnt_tl"].fillna(0)

    # Months since most recent revolving tradeline opening
    out["mo_sin_rcnt_rev"] = df["mo_sin_rcnt_rev_tl_op"].fillna(0)

    # Months since most recent bankcard opening
    out["mths_since_recent_bc"] = df["mths_since_recent_bc"].fillna(0)

    # Portfolio diversification: number of account types
    out["num_il_tl"] = df["num_il_tl"].fillna(0)
    out["num_rev_accts"] = df["num_rev_accts"].fillna(0)

    # Active revolving accounts as fraction of total revolving
    out["rev_active_ratio"] = _safe_div(
        df["num_actv_rev_tl"].fillna(0),
        df["num_rev_accts"].clip(lower=1),
    )

    # Active bankcard lines as fraction of total BC lines
    out["bc_active_ratio"] = _safe_div(
        df["num_actv_bc_tl"].fillna(0),
        df["num_bc_tl"].clip(lower=1),
    )

    # Account churn: new accounts (24m) vs total
    out["acct_churn_24m"] = _safe_div(
        df["acc_open_past_24mths"].fillna(0),
        df["total_acc"].clip(lower=1),
    )

    # Mortgage indicator (has mortgage accounts)
    out["has_mortgage"] = (df["mort_acc"].fillna(0) > 0).astype(int)
    out["mort_acc"] = df["mort_acc"].fillna(0)

    # Seasoning gap: oldest account age minus newest account age
    out["seasoning_gap"] = out["mo_sin_old_rev"] - out["mo_sin_rcnt_tl"]

    return out


def _capacity_stress(df):
    """Leverage and debt-to-available-credit stress signals."""
    out = pd.DataFrame(index=df.index)

    # DTI (already computed by LC)
    out["dti"] = df["dti"].fillna(df["dti"].median())

    # Total current balance relative to income
    out["bal_to_income"] = _safe_div(
        df["tot_cur_bal"].fillna(0),
        df["annual_inc"].clip(lower=1),
    )

    # Total debt (excl mortgage) relative to income
    out["debt_ex_mort_to_income"] = _safe_div(
        df["total_bal_ex_mort"].fillna(0),
        df["annual_inc"].clip(lower=1),
    )

    # Revolving balance relative to income
    out["revol_to_income"] = _safe_div(
        df["revol_bal"].fillna(0),
        df["annual_inc"].clip(lower=1),
    )

    # Total credit limit relative to income (credit access)
    out["credit_limit_to_income"] = _safe_div(
        df["tot_hi_cred_lim"].fillna(0),
        df["annual_inc"].clip(lower=1),
    )

    # Instalment debt relative to instalment credit limit
    out["il_leverage"] = _safe_div(
        df["total_bal_il"].fillna(0),
        df["total_il_high_credit_limit"].clip(lower=1),
    )

    # Average balance per account
    out["avg_cur_bal"] = df["avg_cur_bal"].fillna(0)

    # Total collections amount (prior defaults)
    out["tot_coll_amt"] = df["tot_coll_amt"].fillna(0)

    # Installment-to-income ratio for the LC loan itself
    out["installment_to_income"] = _safe_div(
        df["installment"] * 12,
        df["annual_inc"].clip(lower=1),
    )

    return out


def _loan_level_risk(df):
    """The loan request itself relative to the borrower's profile."""
    out = pd.DataFrame(index=df.index)

    # Loan amount relative to income
    out["loan_to_income"] = _safe_div(
        df["loan_amnt"],
        df["annual_inc"].clip(lower=1),
    )

    # Interest rate (numeric)
    out["int_rate"] = pd.to_numeric(
        df["int_rate"].astype(str).str.replace("%", ""),
        errors="coerce",
    ).fillna(df["int_rate"].median() if df["int_rate"].dtype == float else 13.0)

    # Term (binary: 60-month = higher risk)
    out["term_60m"] = df["term"].astype(str).str.contains("60").astype(int)

    # Employment length (numeric years)
    out["emp_length_yrs"] = _parse_emp_length(df["emp_length"])

    # FICO midpoint
    out["fico_mid"] = (
        df["fico_range_low"].fillna(660) + df["fico_range_high"].fillna(664)
    ) / 2

    # Loan amount relative to revolving credit limit
    # High = the borrower is seeking a big chunk vs existing capacity
    out["loan_to_revol_limit"] = _safe_div(
        df["loan_amnt"],
        df["total_rev_hi_lim"].clip(lower=1),
    )

    # Loan amount relative to total credit
    out["loan_to_total_credit"] = _safe_div(
        df["loan_amnt"],
        df["tot_hi_cred_lim"].clip(lower=1),
    )

    return out


# ── main builder ──────────────────────────────────────────────────────

FEATURE_GROUPS = {
    "credit_seeking_velocity": _credit_seeking_velocity,
    "utilisation_dynamics":    _utilisation_dynamics,
    "payment_discipline":      _payment_discipline,
    "account_lifecycle":       _account_lifecycle,
    "capacity_stress":         _capacity_stress,
    "loan_level_risk":         _loan_level_risk,
}


def build_features(df, sample_n=None, random_state=42):
    """
    Load raw LC data, define target, engineer all features.

    Parameters
    ----------
    df : DataFrame  — raw accepted loans (or already loaded)
    sample_n : int or None — subsample for speed (None = use all)
    random_state : int

    Returns
    -------
    features : DataFrame  — engineered features (numeric, no target)
    target : Series       — binary default indicator
    feature_groups : dict  — {group_name: [col_names]}
    """
    # Define target (terminal statuses only)
    df = define_target(df)

    if sample_n is not None and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=random_state).reset_index(drop=True)

    # Build each group
    parts = {}
    feature_groups = {}
    for name, builder in FEATURE_GROUPS.items():
        part = builder(df)
        parts[name] = part
        feature_groups[name] = list(part.columns)

    features = pd.concat(parts.values(), axis=1)
    target = df["default"]

    # Drop any remaining all-null columns
    null_pct = features.isna().mean()
    drop_cols = null_pct[null_pct > 0.5].index.tolist()
    if drop_cols:
        print(f"  Dropping {len(drop_cols)} cols with >50% missing: {drop_cols}")
        features = features.drop(columns=drop_cols)
        for grp, cols in feature_groups.items():
            feature_groups[grp] = [c for c in cols if c not in drop_cols]

    # Fill remaining NaN with median
    for col in features.columns:
        if features[col].isna().any():
            features[col] = features[col].fillna(features[col].median())

    print(f"  Built {features.shape[1]} features across {len(feature_groups)} groups")
    print(f"  Samples: {len(features):,}  Event rate: {target.mean():.4f}")
    for grp, cols in feature_groups.items():
        print(f"    {grp}: {len(cols)} features")

    return features, target, feature_groups


# Columns needed from the raw CSV
NEEDED_COLS = [
    "loan_status", "loan_amnt", "funded_amnt", "term", "int_rate",
    "installment", "emp_length", "annual_inc", "dti",
    "delinq_2yrs", "fico_range_low", "fico_range_high",
    "inq_last_6mths", "mths_since_last_delinq", "mths_since_last_record",
    "open_acc", "pub_rec", "revol_bal", "revol_util", "total_acc",
    "collections_12_mths_ex_med", "mths_since_last_major_derog",
    "acc_open_past_24mths", "avg_cur_bal", "bc_open_to_buy", "bc_util",
    "mo_sin_old_il_acct", "mo_sin_old_rev_tl_op", "mo_sin_rcnt_rev_tl_op",
    "mo_sin_rcnt_tl", "mort_acc", "mths_since_recent_bc",
    "mths_since_recent_inq", "num_accts_ever_120_pd",
    "num_actv_bc_tl", "num_actv_rev_tl", "num_bc_tl",
    "num_il_tl", "num_rev_accts", "num_tl_30dpd", "num_tl_90g_dpd_24m",
    "pct_tl_nvr_dlq", "percent_bc_gt_75", "pub_rec_bankruptcies",
    "tot_coll_amt", "tot_cur_bal", "tot_hi_cred_lim",
    "total_bal_ex_mort", "total_bal_il", "total_bc_limit",
    "total_il_high_credit_limit", "total_rev_hi_lim",
    "all_util", "il_util", "inq_last_12m",
    "open_rv_12m", "open_il_12m", "num_rev_tl_bal_gt_0",
    "mths_since_recent_bc_dlq", "mths_since_recent_revol_delinq",
    "num_tl_120dpd_2m",
    "issue_d",
]


def load_lending_club(path, sample_n=None, random_state=42):
    """Convenience: load CSV and build features in one call."""
    print(f"  Loading {path} ...")
    df = pd.read_csv(path, usecols=NEEDED_COLS, low_memory=False)
    return build_features(df, sample_n=sample_n, random_state=random_state)


def load_lending_club_temporal(
    path,
    dev_end="Dec-2016",
    holdout_start="Jan-2017",
    sample_n=None,
    random_state=42,
):
    """Load LC data with a temporal split for out-of-time validation.

    Parameters
    ----------
    path : str
        Path to accepted loans CSV.
    dev_end : str
        Last month in development period (format "Mon-YYYY").
    holdout_start : str
        First month in holdout period (format "Mon-YYYY").
    sample_n : int or None
        Subsample per period (applied independently).
    random_state : int

    Returns
    -------
    dev_features, dev_target, ho_features, ho_target, feature_groups
    """
    print(f"  Loading {path} (temporal split) ...")
    df = pd.read_csv(path, usecols=NEEDED_COLS, low_memory=False)

    # Parse issue date
    # Filter to terminal loan statuses before splitting
    df = define_target(df)

    # Parse issue date and split temporally
    df["issue_dt"] = pd.to_datetime(df["issue_d"], format="%b-%Y", errors="coerce")
    n_before = len(df)
    df = df.dropna(subset=["issue_dt"])
    if len(df) < n_before:
        print(f"  Dropped {n_before - len(df)} rows with unparseable issue_d")

    dev_cutoff = pd.to_datetime(dev_end, format="%b-%Y")
    ho_cutoff = pd.to_datetime(holdout_start, format="%b-%Y")

    dev_df = df[df["issue_dt"] <= dev_cutoff].copy()
    ho_df = df[df["issue_dt"] >= ho_cutoff].copy()

    print(f"  Dev period: {dev_df['issue_dt'].min():%Y-%m} to {dev_df['issue_dt'].max():%Y-%m}  "
          f"({len(dev_df):,} loans)")
    print(f"  Holdout period: {ho_df['issue_dt'].min():%Y-%m} to {ho_df['issue_dt'].max():%Y-%m}  "
          f"({len(ho_df):,} loans)")

    # Build features independently per period to avoid leakage.
    # define_target already ran above, so skip it inside build_features
    # by passing pre-filtered data that still has the 'default' column.
    print("  Building dev features ...")
    dev_features, dev_target, feature_groups = build_features(
        dev_df, sample_n=sample_n, random_state=random_state,
    )
    print("  Building holdout features ...")
    ho_features, ho_target, _ = build_features(
        ho_df, sample_n=sample_n, random_state=random_state + 1,
    )

    return dev_features, dev_target, ho_features, ho_target, feature_groups


if __name__ == "__main__":
    # Quick sanity check
    DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "..", "accepted_2007_to_2018Q4.csv", "accepted_2007_to_2018Q4.csv")
    features, target, groups = load_lending_club(DATA_PATH, sample_n=10000)
    print(f"\nFeature matrix: {features.shape}")
    print(f"Target distribution:\n{target.value_counts()}")
    print(f"\nFirst few features:\n{features.head()}")
