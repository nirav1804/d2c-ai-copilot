import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="AI Returns Intelligence Copilot (CMO)",
    layout="wide"
)

st.title("📦 AI Returns Intelligence Copilot")
st.caption("CMO-grade decisions for Returns, Revenue & Growth")

# =========================
# FILE UPLOADERS
# =========================
st.subheader("📤 Upload Data")

reviews_file = st.file_uploader("Customer Reviews CSV", type="csv")
sku_file = st.file_uploader("New SKU CSV (Pre-Launch)", type="csv")

# =========================
# HELPERS
# =========================
def normalize(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

# =========================
# ML ISSUE CLUSTERING
# =========================
def cluster_issues(df, text_col, n_clusters=5):
    tfidf = TfidfVectorizer(
        stop_words="english",
        max_features=800
    )
    X = tfidf.fit_transform(df[text_col].astype(str))

    model = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10
    )
    df["cluster"] = model.fit_predict(X)

    terms = tfidf.get_feature_names_out()
    labels = {}
    for i in range(n_clusters):
        top = model.cluster_centers_[i].argsort()[-3:]
        labels[i] = ", ".join([terms[t] for t in top])

    df["issue"] = df["cluster"].map(labels)
    return df

# =========================
# MAIN PROCESSING
# =========================
issue_baseline = None
sku_model = None
le = LabelEncoder()

if reviews_file:

    reviews = normalize(pd.read_csv(reviews_file))

    if not {"review_text", "rating"}.issubset(reviews.columns):
        st.error("Reviews file must contain: review_text, rating")
        st.stop()

    reviews = cluster_issues(reviews, "review_text")
    reviews["is_negative"] = (reviews["rating"] <= 2).astype(int)

    issue_baseline = (
        reviews.groupby("issue")
        .agg(
            total_reviews=("rating", "count"),
            negative_reviews=("is_negative", "sum")
        )
        .reset_index()
    )

    issue_baseline["negative_rate"] = (
        issue_baseline["negative_reviews"] /
        issue_baseline["total_reviews"]
    )

    # =========================
    # RISK DASHBOARD
    # =========================
    st.subheader("📊 AI-Detected Return Risk Drivers")

    col1, col2 = st.columns(2)
    with col1:
        st.dataframe(
            issue_baseline.sort_values(
                "negative_rate", ascending=False
            )
        )
    with col2:
        fig, ax = plt.subplots()
        ax.barh(
            issue_baseline["issue"],
            issue_baseline["negative_rate"]
        )
        ax.set_xlabel("Negative Review Rate")
        st.pyplot(fig)

    # =========================
    # BUSINESS INPUTS
    # =========================
    st.subheader("💼 Business Assumptions")

    col1, col2 = st.columns(2)
    with col1:
        aov = st.number_input(
            "Average Order Value (₹)",
            500, 10000, 1500
        )
    with col2:
        monthly_orders = st.number_input(
            "Monthly Orders",
            1000, 500000, 10000
        )

    issue_baseline["monthly_loss"] = (
        issue_baseline["negative_rate"] *
        monthly_orders *
        aov
    )

    # =========================
    # SCENARIO PLANNER
    # =========================
    st.markdown("---")
    st.subheader("🧪 Scenario Planner — Budget Allocation")

    st.caption(
        "Split improvement effort across Marketing, Product & Operations"
    )

    # Effectiveness multipliers
    EFFECTIVENESS = {
        "marketing": 0.4,
        "product": 0.8,
        "ops": 0.6
    }

    scenario = issue_baseline.copy()
    results = []

    for _, row in scenario.iterrows():

        st.markdown(f"### 🔧 Issue: {row['issue']}")

        col1, col2, col3 = st.columns(3)
        with col1:
            mkt = st.slider(
                "Marketing %",
                0, 50, 0, step=5,
                key=f"m_{row['issue']}"
            )
        with col2:
            prod = st.slider(
                "Product %",
                0, 50, 0, step=5,
                key=f"p_{row['issue']}"
            )
        with col3:
            ops = st.slider(
                "Operations %",
                0, 50, 0, step=5,
                key=f"o_{row['issue']}"
            )

        effective_reduction = (
            mkt * EFFECTIVENESS["marketing"] +
            prod * EFFECTIVENESS["product"] +
            ops * EFFECTIVENESS["ops"]
        ) / 100

        optimized_rate = max(
            0,
            row["negative_rate"] * (1 - effective_reduction)
        )

        optimized_loss = (
            optimized_rate *
            monthly_orders *
            aov
        )

        results.append({
            "issue": row["issue"],
            "current_risk": row["negative_rate"],
            "optimized_risk": optimized_rate,
            "current_loss": row["monthly_loss"],
            "optimized_loss": optimized_loss,
            "revenue_saved": row["monthly_loss"] - optimized_loss
        })

    scenario_df = pd.DataFrame(results)

    # =========================
    # SCENARIO OUTPUT
    # =========================
    st.markdown("---")
    st.subheader("📈 Scenario Impact Summary")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric(
            "Current Monthly Loss",
            f"₹{int(scenario_df.current_loss.sum()):,}"
        )
    with col2:
        st.metric(
            "Optimized Monthly Loss",
            f"₹{int(scenario_df.optimized_loss.sum()):,}"
        )
    with col3:
        st.metric(
            "Revenue Saved",
            f"₹{int(scenario_df.revenue_saved.sum()):,}"
        )

    fig, ax = plt.subplots()
    ax.barh(
        scenario_df["issue"],
        scenario_df["current_loss"],
        label="Current"
    )
    ax.barh(
        scenario_df["issue"],
        scenario_df["optimized_loss"],
        label="Optimized"
    )
    ax.legend()
    ax.set_title("Revenue Loss: Before vs After Budget Allocation")
    st.pyplot(fig)

    st.dataframe(
        scenario_df.sort_values(
            "revenue_saved", ascending=False
        )
    )

    # =========================
    # SKU MODEL
    # =========================
    reviews["issue_enc"] = le.fit_transform(reviews["issue"])
    X = reviews[["issue_enc"]]
    y = reviews["is_negative"]

    sku_model = LogisticRegression()
    sku_model.fit(X, y)

# =========================
# PRE-LAUNCH SKU SCORING
# =========================
if sku_file and sku_model is not None:

    st.markdown("---")
    st.subheader("🚀 Pre-Launch SKU Risk Prediction")

    skus = normalize(pd.read_csv(sku_file))
    top_issue = issue_baseline.sort_values(
        "negative_rate", ascending=False
    ).iloc[0]["issue"]

    enc = le.transform([top_issue])[0]

    output = []
    for _, sku in skus.iterrows():
        risk = sku_model.predict_proba([[enc]])[0][1]

        decision = (
            "🟢 Launch"
            if risk < 0.25 else
            "🟠 Improve Messaging"
            if risk < 0.35 else
            "🔴 Fix Product Before Launch"
        )

        output.append({
            "SKU": sku["sku_name"],
            "Category": sku["category"],
            "Price": sku["price"],
            "Return Risk": round(risk, 2),
            "Decision": decision
        })

    st.dataframe(pd.DataFrame(output))

# =========================
# AI COPILOT (CMO)
# =========================
st.markdown("---")
st.subheader("🤖 AI Copilot (CMO)")

q = st.text_input(
    "Ask: Where should I invest? Product or Marketing? What saves more money?"
)

if q:
    st.success(
        "CMO Insight: Product investments yield highest risk reduction, "
        "Marketing works best for expectation gaps, and Ops improves trust. "
        "Focus budgets where revenue saved per % is highest."
    )
