import streamlit as st
import pandas as pd
import os
import numpy as np

# =====================================================
# CRITICAL FIX: DISABLE PYARROW STRING STORAGE
# =====================================================
pd.options.mode.string_storage = "python"

# =====================================================
# PAGE CONFIG
# =====================================================
st.set_page_config(
    page_title="Cross-Border Fraud Risk Decision Tool",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================
BASE_PATH = "OUTPUTS/RISK_SCORE_TXNS"
MONTHS = ["2025_01", "2025_02", "2025_03"]

@st.cache_data
def load_data():
    dfs = []
    for m in MONTHS:
        path = os.path.join(BASE_PATH, f"risk_scored_transactions_{m}.csv")
        df = pd.read_csv(path, parse_dates=["transaction_timestamp"])
        df["month"] = m
        dfs.append(df)

    df_all = pd.concat(dfs, ignore_index=True)

    # Force all string columns to pure Python strings
    for col in df_all.select_dtypes(include=["object", "string"]).columns:
        df_all[col] = df_all[col].astype(str)

    return df_all

df = load_data()

# =====================================================
# TITLE
# =====================================================
st.title("Cross-Border Fraud Risk Decision Tool")

st.markdown(
    """
    This application demonstrates a **fraud risk decision system**
    for **outbound cross-border transactions originating from India**.

    It combines **rules**, **ML risk scores**, and **customer trust scores**
    to decide whether a transaction should be:
    **ALLOW**, **REVIEW**, or **BLOCK**.
    """
)

# =====================================================
# TABS
# =====================================================
tab1, tab2, tab3, tab4 = st.tabs(
    ["Overview", "Decision Simulator", "Risk Analysis", "Transaction Explorer"]
)

# =====================================================
# TAB 1 — OVERVIEW
# =====================================================
with tab1:
    st.subheader("System Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    c3.metric("Months Covered", df["month"].nunique())
    c4.metric("Destination Countries", df["destination_country"].nunique())

    decision_month = (
        df.groupby(["month", "decision"])
        .size()
        .reset_index(name="count")
    )

    decision_month["total"] = decision_month.groupby("month")["count"].transform("sum")
    decision_month["percentage"] = decision_month["count"] / decision_month["total"] * 100

    pivot = decision_month.pivot(
        index="month",
        columns="decision",
        values="percentage"
    ).fillna(0)

    st.markdown("### Monthly Decision Split (%)")
    st.bar_chart(pivot)

# =====================================================
# TAB 2 — DECISION SIMULATOR
# =====================================================
with tab2:
    st.subheader("Decision Simulator")

    block_threshold = st.slider("Block if ML risk ≥", 0.5, 1.0, 0.9, 0.05)
    review_threshold = st.slider("Review if ML risk ≥", 0.1, 0.9, 0.6, 0.05)
    trust_override = st.slider("Auto-allow if Trust score ≥", 0, 100, 70, 5)

    sim_df = df.copy()

    def simulate(row):
        if row["ml_risk_score"] >= block_threshold:
            d = "BLOCK"
        elif row["ml_risk_score"] >= review_threshold:
            d = "REVIEW"
        else:
            d = "ALLOW"

        if row["trust_score"] >= trust_override and row["ml_risk_score"] < block_threshold:
            d = "ALLOW"

        return d

    sim_df["simulated_decision"] = sim_df.apply(simulate, axis=1)

    sim_dist = sim_df["simulated_decision"].value_counts(normalize=True) * 100
    orig_dist = df["decision"].value_counts(normalize=True) * 100

    c1, c2, c3 = st.columns(3)
    c1.metric("ALLOW %", f"{sim_dist.get('ALLOW',0):.2f}", f"{sim_dist.get('ALLOW',0)-orig_dist.get('ALLOW',0):+.2f}")
    c2.metric("REVIEW %", f"{sim_dist.get('REVIEW',0):.2f}", f"{sim_dist.get('REVIEW',0)-orig_dist.get('REVIEW',0):+.2f}")
    c3.metric("BLOCK %", f"{sim_dist.get('BLOCK',0):.2f}", f"{sim_dist.get('BLOCK',0)-orig_dist.get('BLOCK',0):+.2f}")

# =====================================================
# TAB 3 — RISK ANALYSIS
# =====================================================
with tab3:
    st.subheader("Risk Analysis")

    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(df["ml_risk_score"], bins=bins)

    risk_df = pd.DataFrame({
        "Risk Band": [f"{edges[i]:.2f}-{edges[i+1]:.2f}" for i in range(len(hist))],
        "Transactions": hist
    })

    st.bar_chart(risk_df.set_index("Risk Band"))

    corridor = (
        df.groupby("destination_country")["ml_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
        .reset_index()
    )

    st.markdown("### Highest Risk Destination Corridors")
    st.dataframe(corridor, use_container_width=True)

# =====================================================
# TAB 4 — TRANSACTION EXPLORER
# =====================================================
with tab4:
    st.subheader("Transaction Explorer")

    months = sorted(df["month"].unique().tolist())
    decisions = sorted(df["decision"].unique().tolist())

    m_filter = st.multiselect("Month", months, months)
    d_filter = st.multiselect("Decision", decisions, decisions)
    min_risk = st.slider("Minimum ML risk score", 0.0, 1.0, 0.0, 0.05)

    view = df[
        (df["month"].isin(m_filter)) &
        (df["decision"].isin(d_filter)) &
        (df["ml_risk_score"] >= min_risk)
    ]

    st.dataframe(
        view[
            [
                "transaction_id",
                "month",
                "transaction_amount",
                "ml_risk_score",
                "trust_score",
                "decision",
                "reason_codes_str"
            ]
        ]
        .sort_values("ml_risk_score", ascending=False)
        .head(500),
        use_container_width=True
    )
