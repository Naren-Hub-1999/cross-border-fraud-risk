import streamlit as st
import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np

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
    return pd.concat(dfs, ignore_index=True)

df = load_data()

# =====================================================
# TITLE
# =====================================================
st.title("Cross-Border Fraud Risk Decision Tool")

st.markdown(
    """
    This application demonstrates a **fraud risk decision system**
    for **outbound cross-border transactions originating from India**.

    The system combines **rule signals**, **ML risk score**, and
    **customer trust score** to decide whether a transaction is:
    **Allowed**, **Reviewed**, or **Blocked**.
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

    st.markdown(
        """
        **Scope**
        - Outbound transactions from India to multiple destination countries
        - Synthetic data generated for learning and demonstration

        **Decision Inputs**
        - Rule indicators (device change, corridor risk)
        - ML transaction risk score
        - Customer trust score (evolves monthly)

        **Decision Outputs**
        - ALLOW
        - REVIEW
        - BLOCK
        """
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Transactions", f"{len(df):,}")
    c2.metric("Unique Customers", f"{df['customer_id'].nunique():,}")
    c3.metric("Months Covered", len(df["month"].unique()))
    c4.metric("Destination Countries", df["destination_country"].nunique())

    st.markdown("### Monthly Decision Split (%)")

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

    st.bar_chart(pivot)

# =====================================================
# TAB 2 — DECISION SIMULATOR
# =====================================================
with tab2:
    st.subheader("Decision Simulator")

    st.markdown(
        """
        This section simulates **policy tuning**.
        The model is not retrained — only decision thresholds are adjusted.
        """
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        block_threshold = st.slider(
            "Block if ML risk ≥",
            0.5, 1.0, 0.9, 0.05
        )

    with col2:
        review_threshold = st.slider(
            "Review if ML risk ≥",
            0.1, 0.9, 0.6, 0.05
        )

    with col3:
        trust_override = st.slider(
            "Auto-allow if Trust score ≥",
            0, 100, 70, 5
        )

    sim_df = df.copy()

    def simulate_decision(row):
        if row["ml_risk_score"] >= block_threshold:
            decision = "BLOCK"
        elif row["ml_risk_score"] >= review_threshold:
            decision = "REVIEW"
        else:
            decision = "ALLOW"

        if row["trust_score"] >= trust_override and row["ml_risk_score"] < block_threshold:
            decision = "ALLOW"

        return decision

    sim_df["simulated_decision"] = sim_df.apply(simulate_decision, axis=1)

    st.markdown("### Decision Outcome Comparison")

    sim_dist = sim_df["simulated_decision"].value_counts(normalize=True) * 100
    orig_dist = df["decision"].value_counts(normalize=True) * 100

    d1, d2, d3 = st.columns(3)

    d1.metric(
        "ALLOW %",
        f"{sim_dist.get('ALLOW', 0):.2f}",
        f"{sim_dist.get('ALLOW', 0) - orig_dist.get('ALLOW', 0):+.2f}"
    )
    d2.metric(
        "REVIEW %",
        f"{sim_dist.get('REVIEW', 0):.2f}",
        f"{sim_dist.get('REVIEW', 0) - orig_dist.get('REVIEW', 0):+.2f}"
    )
    d3.metric(
        "BLOCK %",
        f"{sim_dist.get('BLOCK', 0):.2f}",
        f"{sim_dist.get('BLOCK', 0) - orig_dist.get('BLOCK', 0):+.2f}"
    )

# =====================================================
# TAB 3 — RISK ANALYSIS
# =====================================================
with tab3:
    st.subheader("Risk Analysis")

    st.markdown("### ML Risk Score Distribution")

    bins = np.linspace(0, 1, 21)
    hist, edges = np.histogram(df["ml_risk_score"], bins=bins)

    risk_dist = pd.DataFrame({
        "Risk Score Range": [f"{edges[i]:.2f}–{edges[i+1]:.2f}" for i in range(len(hist))],
        "Transactions": hist
    })

    st.bar_chart(risk_dist.set_index("Risk Score Range"))

    st.markdown("### Highest Risk Destination Corridors (from India)")

    corridor_risk = (
        df.groupby(["destination_country"])
        ["ml_risk_score"]
        .mean()
        .sort_values(ascending=False)
        .head(5)
    )

    st.dataframe(corridor_risk.reset_index())

# =====================================================
# TAB 4 — TRANSACTION EXPLORER
# =====================================================
with tab4:
    st.subheader("Transaction Explorer")

    m_filter = st.multiselect(
        "Month",
        df["month"].unique(),
        df["month"].unique()
    )

    d_filter = st.multiselect(
        "Decision",
        df["decision"].unique(),
        df["decision"].unique()
    )

    min_risk = st.slider(
        "Minimum ML risk score",
        0.0, 1.0, 0.0, 0.05
    )

    view_df = df[
        (df["month"].isin(m_filter)) &
        (df["decision"].isin(d_filter)) &
        (df["ml_risk_score"] >= min_risk)
    ]

    st.dataframe(
        view_df[
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
