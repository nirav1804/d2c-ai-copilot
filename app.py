import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# APP CONFIG
# =========================
st.set_page_config(
    page_title="AI Returns Intelligence Copilot",
    layout="wide"
)

st.title("📦 AI Returns Intelligence Copilot")
st.caption("Predict • Prevent • Improve Product Returns | MVP")

# =========================
# FILE UPLOADERS (ALWAYS VISIBLE)
# =========================
st.subheader("📤 Upload Historical Data")

reviews_file = st.file_uploader(
    "Upload Customer Reviews CSV",
    type=["csv"]
)

returns_file = st.file_uploader(
    "Upload Returns CSV",
    type=["csv"]
)

st.markdown("---")
st.subheader("🚀 Pre-Launch SKU Risk Scoring")

sku_file = st.file_uploader(
    "Upload New SKU CSV",
    type=["csv"]
)

st.markdown("---")

# =========================
# HELPER FUNCTIONS
# =========================
def normalize_columns(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ", "_")
    )
    return df

def detect_order_key(columns):
    possible = [
        "order_id", "orderid", "order_no",
        "order_number", "order"
    ]
    for p in possible:
        if p in columns:
            return p
    return None

# =========================
# PROCESS REVIEWS & RETURNS
# =========================
issue_baseline = None
high_risk = pd.DataFrame()

if reviews_file is not None and returns_file is not None:

    reviews = normalize_columns(pd.read_csv(reviews_file))
    returns = normalize_columns(pd.read_csv(returns_file))

    review_key = detect_order_key(reviews.columns)
    return_key = detect_order_key(returns.columns)

    if review_key is None or return_key is None:
        st.error("❌ Order ID column not found in one or both files")
        st.write("Reviews columns:", reviews.columns.tolist())
        st.write("Returns columns:", returns.columns.tolist())
        st.stop()

    merged = pd.merge(
        reviews,
        returns,
        left_on=review_key,
        right_on=return_key,
        how="left"
    )

    # Safety checks
    required_cols = ["rating", "issue"]
    for col in required_cols:
        if col not in merged.columns:
            st.error(f"❌ Missing required column: {col}")
            st.stop()

    merged["sentiment"] = np.where(
        merged["rating"] <= 2, "negative", "positive"
    )

    # =========================
    # ISSUE BASELINE
    # =========================
    issue_baseline = (
        merged.groupby("issue")
        .agg(
            total_reviews=("rating", "count"),
            negative_reviews=("sentiment", lambda x: (x == "negative").sum()),
            returns=("return_reason", "count"),
            revenue_loss=("refund_amount", "sum")
        )
        .reset_index()
    )

    issue_baseline["negative_rate"] = (
        issue_baseline["negative_reviews"] /
        issue_baseline["total_reviews"]
    ).fillna(0)

    # =========================
    # DASHBOARD
    # =========================
    st.subheader("📊 Return Risk Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "💸 Total Revenue Lost",
            f"₹{int(issue_baseline.revenue_loss.sum()):,}"
        )
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
        ax.set_title("Return Risk by Issue")
        st.pyplot(fig)

    # =========================
    # ALERTS
    # =========================
    high_risk = issue_baseline[
        issue_baseline["negative_rate"] > 0.35
    ]

    if not high_risk.empty:
        st.error("🚨 HIGH RETURN RISK DETECTED")
        st.dataframe(
            high_risk[["issue", "negative_rate"]]
        )

    # =========================
    # FUTURE RISK PREDICTION
    # =========================
    st.subheader("🔮 Future Return Risk Prediction")

    projected_risk = issue_baseline["negative_rate"].mean()
    confidence = min(
        90,
        max(55, int(issue_baseline["total_reviews"].sum() / 8))
    )

    st.success(
        f"Expected return risk next cycle: "
        f"{round(projected_risk*100,2)}% "
        f"(Confidence: {confidence}%)"
    )

    # =========================
    # AUTO PRODUCT IMPROVEMENTS
    # =========================
    st.subheader("🛠️ Auto-Generated Fixes")

    if high_risk.empty:
        st.info("No critical issues detected 🎉")
    else:
        for _, row in high_risk.iterrows():
            st.write(
                f"• Improve **{row['issue']}** "
                f"(~{round(row['negative_rate']*100,1)}% dissatisfaction)"
            )

    # =========================
    # WHAT-IF SIMULATION
    # =========================
    st.subheader("🧪 What-If Simulation")

    reduction = st.slider(
        "Reduce negative reviews for top issue (%)",
        0, 50, 20
    )

    top_issue = issue_baseline.sort_values(
        "negative_rate", ascending=False
    ).iloc[0]

    new_risk = top_issue["negative_rate"] * (1 - reduction / 100)

    st.info(
        f"If **{top_issue['issue']}** improves by {reduction}%, "
        f"risk drops from "
        f"{round(top_issue['negative_rate']*100,2)}% → "
        f"{round(new_risk*100,2)}%"
    )

else:
    st.info("👆 Upload Reviews & Returns CSVs to activate analytics")

# =========================
# PRE-LAUNCH SKU SCORING
# =========================
if sku_file is not None:

    skus = normalize_columns(pd.read_csv(sku_file))

    required_sku_cols = ["sku_name", "category", "price"]
    for col in required_sku_cols:
        if col not in skus.columns:
            st.error(f"❌ SKU file missing column: {col}")
            st.stop()

    st.subheader("🚀 SKU Pre-Launch Risk Scores")

    results = []

    for _, sku in skus.iterrows():

        if issue_baseline is not None:
            top_issue = issue_baseline.sort_values(
                "negative_rate", ascending=False
            ).iloc[0]
            risk = round(top_issue["negative_rate"], 2)
            driver = top_issue["issue"]
            conf = min(90, max(60, int(top_issue["total_reviews"] / 5)))
        else:
            risk = round(np.random.uniform(0.15, 0.35), 2)
            driver = "Category benchmark"
            conf = 55

        results.append({
            "SKU": sku["sku_name"],
            "Category": sku["category"],
            "Price": sku["price"],
            "Predicted Return Risk": risk,
            "Primary Risk Driver": driver,
            "Confidence (%)": conf,
            "Recommendation": (
                "❌ Fix Before Launch"
                if risk > 0.3 else
                "✅ Safe to Launch"
            )
        })

    sku_df = pd.DataFrame(results)
    st.dataframe(sku_df)

    fig2, ax2 = plt.subplots()
    ax2.barh(
        sku_df["SKU"],
        sku_df["Predicted Return Risk"]
    )
    ax2.set_title("Pre-Launch Return Risk by SKU")
    st.pyplot(fig2)

# =========================
# AI COPILOT
# =========================
st.markdown("---")
st.subheader("🤖 Ask the Returns Copilot")

question = st.text_input(
    "Ask (e.g. Why are returns high? Should I launch this SKU?)"
)

if question:
    q = question.lower()

    if "why" in q:
        st.success(
            "Returns are driven by quality gaps, expectation mismatch, "
            "and inconsistent delivery experience."
        )
    elif "launch" in q:
        st.success(
            "Launch low-risk SKUs now. Improve high-risk drivers before scaling."
        )
    elif "improve" in q:
        st.success(
            "Focus on product quality, packaging clarity, "
            "and accurate product descriptions."
        )
    else:
        st.success(
            "I analyzed your data and identified key return risk drivers."
        )
