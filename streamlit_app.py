# streamlit_app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from scipy.optimize import minimize


# -----------------------------
# Loss aversion model (Route A)
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable logistic sigmoid."""
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def prepare_y(decision: pd.Series) -> np.ndarray:
    """Map accept/reject to 1/0."""
    y = decision.map({"accept": 1, "reject": 0})
    if y.isna().any():
        bad = decision[y.isna()].unique()
        raise ValueError(f"Unexpected decision values: {bad}. Expected only 'accept'/'reject'.")
    return y.to_numpy(dtype=float)


def neg_log_likelihood(params: np.ndarray, win: np.ndarray, lose: np.ndarray, y: np.ndarray) -> float:
    """
    Parameters are optimized in log-space to enforce positivity:
      params = [log_lambda, log_beta]

    With fixed p = 0.5:
      V = 0.5 * win - 0.5 * lambda * lose
      P(accept) = sigmoid(beta * V)
    """
    log_lambda, log_beta = params
    lam = float(np.exp(log_lambda))
    beta = float(np.exp(log_beta))

    V = 0.5 * win - 0.5 * lam * lose
    p = sigmoid(beta * V)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)

    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -float(ll)


def fit_participant(
    g: pd.DataFrame,
    *,
    lambda_min: float,
    lambda_max: float,
    beta_min: float,
    beta_max: float,
    boundary_atol_log: float = 1e-3,
) -> dict:
    """
    Fit (lambda, beta) via bounded MLE for one participant.
    Adds identifiability flags to handle degenerate solutions (e.g., beta -> 0, lambda -> huge).
    """
    win = g["win"].to_numpy(dtype=float)
    lose = g["lose"].to_numpy(dtype=float)
    y = prepare_y(g["decision"])

    # Initial guesses: lambda ~ 2, beta ~ 1
    x0 = np.log(np.array([2.0, 1.0], dtype=float))

    # Bounds in log-space
    bounds = [
        (np.log(lambda_min), np.log(lambda_max)),  # log_lambda
        (np.log(beta_min), np.log(beta_max)),      # log_beta
    ]

    opt = minimize(
        neg_log_likelihood,
        x0=x0,
        args=(win, lose, y),
        method="L-BFGS-B",
        bounds=bounds,
    )

    if opt.success:
        lam_hat = float(np.exp(opt.x[0]))
        beta_hat = float(np.exp(opt.x[1]))
    else:
        lam_hat = np.nan
        beta_hat = np.nan

    # Boundary flags (in log space)
    hit_lambda_upper = bool(opt.success and np.isclose(opt.x[0], bounds[0][1], atol=boundary_atol_log))
    hit_lambda_lower = bool(opt.success and np.isclose(opt.x[0], bounds[0][0], atol=boundary_atol_log))
    hit_beta_upper = bool(opt.success and np.isclose(opt.x[1], bounds[1][1], atol=boundary_atol_log))
    hit_beta_lower = bool(opt.success and np.isclose(opt.x[1], bounds[1][0], atol=boundary_atol_log))

    # Identifiability rule:
    # - must converge
    # - must NOT be pushed to the boundaries in suspicious ways
    #   (especially beta at lower bound, or lambda at upper bound)
    identifiable = bool(opt.success) and (not hit_beta_lower) and (not hit_lambda_upper)

    return {
        "n_trials": int(len(g)),
        "lambda_hat": lam_hat,
        "beta_hat": beta_hat,
        "success": bool(opt.success),
        "identifiable": identifiable,
        "hit_lambda_lower": hit_lambda_lower,
        "hit_lambda_upper": hit_lambda_upper,
        "hit_beta_lower": hit_beta_lower,
        "hit_beta_upper": hit_beta_upper,
        "neg_ll": float(opt.fun) if opt.fun is not None else np.nan,
        "message": str(opt.message),
    }


def estimate_per_participant(
    df: pd.DataFrame,
    *,
    min_trials: int,
    lambda_min: float,
    lambda_max: float,
    beta_min: float,
    beta_max: float,
) -> pd.DataFrame:
    """Estimate lambda and beta per participant_id, including identifiability flags."""
    required = {"participant_id", "group", "win", "lose", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["win"] = pd.to_numeric(df["win"], errors="raise")
    df["lose"] = pd.to_numeric(df["lose"], errors="raise")

    rows = []
    for pid, g in df.groupby("participant_id", sort=False):
        group_label = str(g["group"].iloc[0])

        if len(g) < min_trials:
            rows.append({
                "participant_id": str(pid),
                "group": group_label,
                "n_trials": int(len(g)),
                "lambda_hat": np.nan,
                "beta_hat": np.nan,
                "success": False,
                "identifiable": False,
                "hit_lambda_lower": False,
                "hit_lambda_upper": False,
                "hit_beta_lower": False,
                "hit_beta_upper": False,
                "neg_ll": np.nan,
                "message": f"Too few trials (<{min_trials})",
            })
            continue

        r = fit_participant(
            g,
            lambda_min=lambda_min,
            lambda_max=lambda_max,
            beta_min=beta_min,
            beta_max=beta_max,
        )
        rows.append({"participant_id": str(pid), "group": group_label, **r})

    return pd.DataFrame(rows)


# -----------------------------
# Streamlit app (English UI)
# -----------------------------

st.set_page_config(page_title="Loss Aversion (λ) Estimator", layout="wide")
st.title("Loss Aversion (λ) Estimation — Per Participant")

st.write(
    "Upload multiple CSV files and estimate **λ** (loss aversion) and **β** (choice sensitivity) per participant "
    "using a bounded maximum-likelihood logistic choice model with fixed probability **p = 0.5**."
)

with st.expander("Expected CSV columns", expanded=False):
    st.markdown(
        "- `participant_id`\n"
        "- `group` (e.g., `control` / `training`)\n"
        "- `win` (numeric)\n"
        "- `lose` (numeric, positive magnitude)\n"
        "- `decision` exactly `accept` or `reject`\n"
    )

# Static tables reduce frontend JS usage and help avoid "module script" errors
use_static_tables = st.toggle(
    "Use static tables (recommended)",
    value=True,
    help="Static tables reduce frontend JS usage and are more stable on Streamlit Cloud."
)

def show_table(df: pd.DataFrame, *, title: str | None = None, max_rows: int = 200) -> None:
    """Render a table in the most compatible way for Streamlit Cloud."""
    if title:
        st.subheader(title)

    display_df = df.copy()
    if len(display_df) > max_rows:
        st.caption(f"Showing first {max_rows} rows out of {len(display_df)}.")
        display_df = display_df.head(max_rows)

    if use_static_tables:
        st.table(display_df)
    else:
        st.dataframe(display_df, use_container_width=True)


files = st.file_uploader("Upload CSV files", type=["csv"], accept_multiple_files=True)

col1, col2, col3, col4 = st.columns(4)
with col1:
    min_trials = st.number_input("Minimum trials per participant", min_value=1, max_value=5000, value=10, step=1)
with col2:
    show_failed = st.checkbox("Show failed estimates in results table", value=True)
with col3:
    show_non_identifiable = st.checkbox("Show non-identifiable estimates in results table", value=True)
with col4:
    st.caption("Non-identifiable = converged but parameter information is weak (e.g., boundary solutions).")

st.markdown("### Parameter bounds (recommended to avoid degenerate solutions)")
b1, b2, b3, b4 = st.columns(4)
with b1:
    lambda_min = st.number_input("lambda_min", min_value=1e-6, value=0.01, format="%.6f")
with b2:
    lambda_max = st.number_input("lambda_max", min_value=0.1, value=50.0, format="%.3f")
with b3:
    beta_min = st.number_input("beta_min", min_value=1e-6, value=0.01, format="%.6f")
with b4:
    beta_max = st.number_input("beta_max", min_value=0.1, value=50.0, format="%.3f")

if not files:
    st.stop()

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f.name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

show_table(data.head(30), title="Data preview", max_rows=30)

st.subheader("Quick validation")
needed = ["participant_id", "group", "win", "lose", "decision"]
missing_cols = [c for c in needed if c not in data.columns]
if missing_cols:
    st.error(f"Missing required columns: {missing_cols}")
    st.stop()

rows_per_group = data["group"].value_counts().rename_axis("group").to_frame("rows")
show_table(rows_per_group, title="Rows per group")

participants_per_group = data.groupby("group")["participant_id"].nunique().rename("n_participants").to_frame()
show_table(participants_per_group, title="Participants per group")

if st.button("Estimate λ per participant", type="primary"):
    with st.spinner("Estimating parameters per participant..."):
        res = estimate_per_participant(
            data,
            min_trials=int(min_trials),
            lambda_min=float(lambda_min),
            lambda_max=float(lambda_max),
            beta_min=float(beta_min),
            beta_max=float(beta_max),
        )

    st.success("Estimation completed.")

    # View filtering (table only)
    view = res.copy()

    if not show_failed:
        view = view[view["success"]].copy()

    if not show_non_identifiable:
        view = view[view["identifiable"]].copy()

    show_table(
        view.sort_values(["group", "participant_id"]),
        title="Results table (per participant)",
        max_rows=500
    )

    st.subheader("Download")
    # Always export ALL rows for transparency
    out_csv = res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download results CSV (λ per participant)",
        data=out_csv,
        file_name="lambda_per_participant.csv",
        mime="text/csv",
    )

    # --- Analysis set for plots: only identifiable estimates ---
    ok = res[(res["success"]) & (res["identifiable"]) & (res["lambda_hat"].notna())].copy()

    st.subheader("Charts")
    if ok.empty:
        st.warning(
            "No identifiable estimates to plot. Consider adjusting bounds or inspect the input data."
        )
        # Still show diagnostics
        diag = res.groupby(["group", "success", "identifiable"]).size().reset_index(name="n")
        show_table(diag, title="Convergence / identifiability diagnostics", max_rows=200)
        st.stop()

    st.markdown("### Overall distribution of λ (identifiable estimates)")
    fig1, ax1 = plt.subplots()
    ax1.hist(ok["lambda_hat"].values, bins=40)
    ax1.set_xlabel("lambda_hat (λ)")
    ax1.set_ylabel("Number of participants")
    st.pyplot(fig1)

    st.markdown("### Distribution of λ by group (identifiable estimates)")
    groups = sorted(ok["group"].unique().tolist())
    for gname in groups:
        sub = ok[ok["group"] == gname]
        figg, axg = plt.subplots()
        axg.hist(sub["lambda_hat"].values, bins=40)
        axg.set_xlabel("lambda_hat (λ)")
        axg.set_ylabel("Number of participants")
        axg.set_title(f"Group: {gname}")
        st.pyplot(figg)

    st.markdown("### Boxplot of λ by group (identifiable estimates)")
    fig2, ax2 = plt.subplots()
    data_for_box = [ok.loc[ok["group"] == g, "lambda_hat"].values for g in groups]
    ax2.boxplot(data_for_box, labels=groups, showfliers=True)
    ax2.set_xlabel("Group")
    ax2.set_ylabel("lambda_hat (λ)")
    st.pyplot(fig2)

    # Diagnostics table
    diag = res.groupby(["group", "success", "identifiable"]).size().reset_index(name="n")
    show_table(diag, title="Convergence / identifiability diagnostics", max_rows=200)

    # Optional: show how many hit bounds
    bounds_diag = res.groupby(["group"])[["hit_lambda_lower", "hit_lambda_upper", "hit_beta_lower", "hit_beta_upper"]].sum()
    show_table(bounds_diag, title="Boundary hits (counts)", max_rows=50)


