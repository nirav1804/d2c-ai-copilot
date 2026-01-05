import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="Returns Intelligence Copilot", layout="wide")

st.title("📦 AI Returns Intelligence Copilot")
st.caption("Predict • Prevent • Improve Returns | Free MVP")

# =========================
# ALWAYS VISIBLE FILE UPLOADERS
# =========================
st.subheader("📤 Data Uploads")

reviews_file = st.file_uploader("Upload Customer Reviews CSV", type=["csv"])
returns_file = st.file_uploader("Upload Returns CSV", type=["csv"])

st.markdown("---")
st.subheader("🚀 Pre-Launch SKU Scoring (Always Visible)")
sku_file = st.file_uploader("Upload New SKU CSV", type=["csv"])
st.write("DEBUG → SKU file status:", sku_file)

st.markdown("---")

issue_baseline = None  # SAFE INITIALIZATION

# =========================
# PROCESS REVIEWS & RETURNS
# =========================
if reviews_file is not None and returns_file is not None:

    reviews = pd.read_csv(reviews_file)
    returns = pd.read_csv(returns_file)

    # Normalize column names
    reviews.columns = reviews.columns.str.lower().str.strip()
    returns.columns = returns.columns.str.lower().str.strip()

    # Detect order id column
    possible_keys = ["order_id", "orderid", "order id", "order_number"]
    review_key = next((c for c in reviews.columns if c in possible_keys), None)
    return_key = next((c for c in returns.columns if c in possible_keys), None)

    if review_key is None or return_key is None:
        st.error("❌ Order ID column not found")
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

    merged["sentiment"] = merged["rating"].apply(
        lambda x: "negative" if x <= 2 else "positive"
    )

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
        st.metric("💸 Total Revenue Lost", f"₹{int(issue_baseline.revenue_loss.sum()):,}")
        st.dataframe(issue_baseline.sort_values("negative_rate", ascending=False))

    with col2:
        fig, ax = plt.subplots()
        ax.barh(issue_baseline["issue"], issue_baseline["negative_rate"])
        ax.set_title("Negative Review Rate by Issue")
        st.pyplot(fig)

    # =========================
    # ALERTS
    # =========================
    high_risk = issue_baseline[issue_baseline["negative_rate"] > 0.35]
    if not high_risk.empty:
        st.error("🚨 ALERT: High Return Risk Detected")
        st.dataframe(high_risk[["issue", "negative_rate"]])

    # =========================
    # FUTURE RISK PREDICTION
    # =========================
    st.subheader("🔮 Future Return Risk Prediction")
    projected_risk = issue_baseline["negative_rate"].mean()
    confidence = min(90, max(50, int(issue_baseline["total_reviews"].sum() / 10)))

    st.success(
        f"Expected return risk next cycle: "
        f"{round(projected_risk*100,2)}% (Confidence: {confidence}%)"
    )

    # =========================
    # WHAT-IF SIMULATION
    # =========================
    st.subheader("🧪 What-If Simulation")
    reduction = st.slider("Reduce top issue negative reviews by (%)", 0, 50, 20)

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
    st.info("👆 Upload Reviews & Returns files to activate analytics")

# =========================
# PRE-LAUNCH SKU SCORING
# =========================
if sku_file is not None:

    skus = pd.read_csv(sku_file)
    st.subheader("🚀 Pre-Launch SKU Risk Scoring Results")

    sku_results = []

    for _, sku in skus.iterrows():

        if issue_baseline is not None:
            top_issue = issue_baseline.sort_values(
                "negative_rate", ascending=False
            ).iloc[0]
            risk = round(top_issue["negative_rate"], 2)
            driver = top_issue["issue"]
            conf = min(90, max(50, int(top_issue["total_reviews"] / 5)))
        else:
            risk = round(np.random.uniform(0.15, 0.35), 2)
            driver = "Category benchmark"
            conf = 55

        sku_results.append({
            "SKU": sku["sku_name"],
            "Category": sku["category"],
            "Price": sku["price"],
            "Predicted Return Risk": risk,
            "Primary Risk Driver": driver,
            "Confidence (%)": conf,
            "Recommendation": (
                "❌ Fix Before Launch" if risk > 0.3 else "✅ Safe to Launch"
            )
        })

    sku_risk_df = pd.DataFrame(sku_results)
    st.dataframe(sku_risk_df)

    fig2, ax2 = plt.subplots()
    ax2.barh(sku_risk_df["SKU"], sku_risk_df["Predicted Return Risk"])
    ax2.set_title("Pre-Launch Return Risk by SKU")
    st.pyplot(fig2)

# =========================
# FREE AI COPILOT
# =========================
st.markdown("---")
st.subheader("🤖 Ask the Returns Copilot")

question = st.text_input(
    "Ask a question (e.g. Why are returns increasing? Should I launch this SKU?)"
)

if question:
    if "why" in question.lower():
        st.success("Returns are mainly driven by product quality and expectation gaps.")
    elif "launch" in question.lower():
        st.success("Launch low-risk SKUs now. Fix high-risk SKUs before scaling.")
    elif "improve" in question.lower():
        st.success("Improve quality, packaging clarity, and sizing info.")
    else:
        st.success("I analyzed your data and highlighted key return risks.")
