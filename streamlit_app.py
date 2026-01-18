# app.py
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

from scipy.optimize import minimize

# -----------------------------
# Model (Ruta A): estimate lambda and beta per participant
# p is fixed at 0.5
# -----------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)  # numerical stability
    return 1.0 / (1.0 + np.exp(-x))

def prepare_y(decision: pd.Series) -> np.ndarray:
    y = decision.map({"accept": 1, "reject": 0})
    if y.isna().any():
        bad = decision[y.isna()].unique()
        raise ValueError(f"Unexpected decision values found: {bad}. Expected only accept/reject.")
    return y.to_numpy(dtype=float)

def neg_log_likelihood(params: np.ndarray, win: np.ndarray, lose: np.ndarray, y: np.ndarray) -> float:
    """
    params = [log_lambda, log_beta] to enforce lambda>0, beta>0
    V = 0.5*win - 0.5*lambda*lose
    P(accept) = sigmoid(beta*V)
    """
    log_lam, log_beta = params
    lam = float(np.exp(log_lam))
    beta = float(np.exp(log_beta))

    V = 0.5 * win - 0.5 * lam * lose
    p = sigmoid(beta * V)

    eps = 1e-12
    p = np.clip(p, eps, 1 - eps)
    ll = np.sum(y * np.log(p) + (1 - y) * np.log(1 - p))
    return -float(ll)

def fit_participant(g: pd.DataFrame) -> dict:
    win = g["win"].to_numpy(dtype=float)
    lose = g["lose"].to_numpy(dtype=float)
    y = prepare_y(g["decision"])

    # Initial values (log-space): lambda ~ 2, beta ~ 1
    x0 = np.log(np.array([2.0, 1.0], dtype=float))

    opt = minimize(
        neg_log_likelihood,
        x0=x0,
        args=(win, lose, y),
        method="L-BFGS-B",
    )

    if opt.success:
        lam_hat = float(np.exp(opt.x[0]))
        beta_hat = float(np.exp(opt.x[1]))
    else:
        lam_hat = np.nan
        beta_hat = np.nan

    return {
        "n_trials": len(g),
        "lambda_hat": lam_hat,
        "beta_hat": beta_hat,
        "success": bool(opt.success),
        "neg_ll": float(opt.fun) if opt.fun is not None else np.nan,
        "message": str(opt.message),
    }

def estimate_per_participant(df: pd.DataFrame, min_trials: int = 10) -> pd.DataFrame:
    required = {"participant_id", "group", "win", "lose", "decision"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    # Ensure numeric
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
                "n_trials": len(g),
                "lambda_hat": np.nan,
                "beta_hat": np.nan,
                "success": False,
                "neg_ll": np.nan,
                "message": f"Too few trials (<{min_trials})",
            })
            continue

        r = fit_participant(g)
        rows.append({
            "participant_id": str(pid),
            "group": group_label,
            **r
        })

    return pd.DataFrame(rows)

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Loss Aversion (λ) Estimator", layout="wide")
st.title("Estimación de Loss Aversion (λ) por participante")

st.write(
    "Sube múltiples CSV y estima **λ** (y **β**) por participante usando un modelo logit con p=0.5. "
    "Luego descarga un CSV con resultados y visualiza distribuciones (global y por grupo)."
)

with st.expander("Formato esperado de columnas", expanded=False):
    st.markdown(
        "- `participant_id`\n"
        "- `group` (por ejemplo: `control` / `training`)\n"
        "- `win` (número)\n"
        "- `lose` (número, magnitud positiva)\n"
        "- `decision` exactamente `accept` o `reject`\n"
    )

files = st.file_uploader("Sube tus CSV", type=["csv"], accept_multiple_files=True)

colA, colB, colC = st.columns(3)
with colA:
    min_trials = st.number_input("Mínimo de trials por participante", min_value=1, max_value=5000, value=10, step=1)
with colB:
    show_failed = st.checkbox("Mostrar también estimaciones fallidas", value=True)
with colC:
    st.caption("Tip: si algún participante no converge, revisa su n_trials o si siempre acepta/rechaza.")

if not files:
    st.stop()

dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = f.name
    dfs.append(df)

data = pd.concat(dfs, ignore_index=True)

st.subheader("Preview de datos (primeras filas)")
st.dataframe(data.head(30), use_container_width=True)

# Quick validation info
st.subheader("Validación rápida")
needed = ["participant_id", "group", "win", "lose", "decision"]
missing_cols = [c for c in needed if c not in data.columns]
if missing_cols:
    st.error(f"Faltan columnas requeridas: {missing_cols}")
    st.stop()

st.write("Conteo de filas por grupo:")
st.dataframe(data["group"].value_counts().rename_axis("group").to_frame("rows"), use_container_width=True)

st.write("Conteo de participantes por grupo:")
st.dataframe(
    data.groupby("group")["participant_id"].nunique().rename("n_participants").to_frame(),
    use_container_width=True
)

if st.button("Estimar λ por participante", type="primary"):
    with st.spinner("Estimando parámetros por participante..."):
        res = estimate_per_participant(data, min_trials=int(min_trials))

    st.success("Estimación terminada.")

    # Optionally filter
    view = res.copy()
    if not show_failed:
        view = view[view["success"] & view["lambda_hat"].notna()].copy()

    st.subheader("Resultados (tabla por participante)")
    st.dataframe(view.sort_values(["group", "participant_id"]), use_container_width=True)

    # Summary stats
    st.subheader("Resumen por grupo (solo éxitos)")
    ok = res[res["success"] & res["lambda_hat"].notna()].copy()

    if ok.empty:
        st.warning("No hay estimaciones exitosas para graficar. Revisa el mínimo de trials o los datos.")
        st.stop()

    summary = ok.groupby("group")["lambda_hat"].describe()
    st.dataframe(summary, use_container_width=True)

    # Download
    st.subheader("Descarga")
    out_csv = res.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Descargar CSV con λ por participante",
        data=out_csv,
        file_name="lambda_per_participant.csv",
        mime="text/csv",
    )

    # -----------------------------
    # Charts
    # -----------------------------
    st.subheader("Gráficos")

    # Overall histogram
    st.markdown("### Distribución general de λ (estimaciones exitosas)")
    hist_all = (
        alt.Chart(ok)
        .mark_bar()
        .encode(
            x=alt.X("lambda_hat:Q", bin=alt.Bin(maxbins=40), title="lambda_hat"),
            y=alt.Y("count():Q", title="N participantes"),
            tooltip=[alt.Tooltip("count():Q", title="N")]
        )
        .properties(height=280)
    )
    st.altair_chart(hist_all, use_container_width=True)

    # Histogram by group
    st.markdown("### Distribución de λ por grupo")
    hist_by_group = (
        alt.Chart(ok)
        .mark_bar()
        .encode(
            x=alt.X("lambda_hat:Q", bin=alt.Bin(maxbins=40), title="lambda_hat"),
            y=alt.Y("count():Q", title="N participantes"),
            column=alt.Column("group:N", title="Grupo"),
            tooltip=[alt.Tooltip("count():Q", title="N")]
        )
        .properties(height=250)
    )
    st.altair_chart(hist_by_group, use_container_width=True)

    # Boxplot by group
    st.markdown("### Boxplot de λ por grupo")
    box = (
        alt.Chart(ok)
        .mark_boxplot()
        .encode(
            x=alt.X("group:N", title="Grupo"),
            y=alt.Y("lambda_hat:Q", title="lambda_hat"),
            tooltip=["group:N"]
        )
        .properties(height=320)
    )
    st.altair_chart(box, use_container_width=True)

    # Convergence diagnostics
    st.markdown("### Diagnóstico de convergencia")
    diag = res.groupby(["group", "success"]).size().reset_index(name="n")
    st.dataframe(diag, use_container_width=True)

    st.caption(
        "Nota: si ves muchos `success=False`, suele ser por pocos trials o por decisiones casi determinísticas "
        "(por ejemplo, siempre accept o siempre reject)."
    )
