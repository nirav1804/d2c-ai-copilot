import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(page_title="Returns Intelligence Copilot", layout="wide")

st.title("📦 AI Returns Intelligence Copilot")
st.caption("Predict • Prevent • Improve Returns (MVP – Free Stack)")

# =========================
# FILE UPLOADS (FIX APPLIED)
# =========================
reviews_file = st.file_uploader("📤 Upload Customer Reviews CSV", type=["csv"])
returns_file = st.file_uploader("📤 Upload Returns CSV", type=["csv"])

st.markdown("---")
st.subheader("🚀 Pre-Launch SKU Scoring (Optional)")
sku_file = st.file_uploader("📤 Upload New SKUs CSV", type=["csv"])

# =========================
# DATA PROCESSING
# =========================
if reviews_file and returns_file:
    reviews = pd.read_csv(reviews_file)
    returns = pd.read_csv(returns_file)

    reviews["sentiment"] = reviews["rating"].apply(
        lambda x: "negative" if x <= 2 else "positive"
    )

    merged = pd.merge(reviews, returns, on="order_id", how="left")

    # Review → Return Theme Mapping
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
        issue_baseline["negative_reviews"] / issue_baseline["total_reviews"]
    ).fillna(0)

    # =========================
    # DASHBOARD
    # =========================
    st.subheader("📊 Return Risk Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Total Revenue Lost", f"₹{int(issue_baseline.revenue_loss.sum()):,}")
        st.dataframe(issue_baseline.sort_values("negative_rate", ascending=False))

    with col2:
        fig, ax = plt.subplots()
        ax.barh(issue_baseline["issue"], issue_baseline["negative_rate"])
        ax.set_title("Negative Review Rate by Issue")
        st.pyplot(fig)

    # =========================
    # ALERT SYSTEM
    # =========================
    high_risk = issue_baseline[issue_baseline["negative_rate"] > 0.35]

    if not high_risk.empty:
        st.error("🚨 ALERT: High Return Risk Detected!")
        st.write(high_risk[["issue", "negative_rate"]])

    # =========================
    # FUTURE RETURN RISK
    # =========================
    st.subheader("🔮 Future Return Risk Prediction")

    projected_risk = issue_baseline["negative_rate"].mean()
    confidence = min(90, int(issue_baseline["total_reviews"].sum() / 20))

    st.success(
        f"📈 Expected Return Risk Next Cycle: **{round(projected_risk*100,2)}%** "
        f"(Confidence: {confidence}%)"
    )

    # =========================
    # AUTO PRODUCT IMPROVEMENT TASKS
    # =========================
    st.subheader("🛠️ Auto-Generated Product Fixes")

    for _, row in high_risk.iterrows():
        st.write(
            f"• Improve **{row['issue']}** — contributes "
            f"{round(row['negative_rate']*100,1)}% dissatisfaction"
        )

    # =========================
    # WHAT-IF SIMULATION
    # =========================
    st.subheader("🧪 What-If Simulation")

    reduction = st.slider(
        "Reduce top issue negative reviews by (%)", 0, 50, 20
    )

    top_issue = issue_baseline.sort_values(
        "negative_rate", ascending=False
    ).iloc[0]

    new_risk = max(0, top_issue["negative_rate"] * (1 - reduction / 100))

    st.info(
        f"If **{top_issue['issue']}** improves by {reduction}%, "
        f"return risk drops from "
        f"{round(top_issue['negative_rate']*100,2)}% → "
        f"{round(new_risk*100,2)}%"
    )

    # =========================
    # PRE-LAUNCH SKU SCORING
    # =========================
    if sku_file is not None:
        skus = pd.read_csv(sku_file)

        st.subheader("🚀 Pre-Launch SKU Risk Scoring")

        sku_results = []

        for _, sku in skus.iterrows():
            top_issue = issue_baseline.sort_values(
                "negative_rate", ascending=False
            ).iloc[0]

            sku_results.append({
                "SKU": sku["sku_name"],
                "Category": sku["category"],
                "Price": sku["price"],
                "Predicted Return Risk": round(top_issue["negative_rate"], 2),
                "Primary Risk Driver": top_issue["issue"],
                "Confidence (%)": min(90, max(50, int(top_issue["total_reviews"] / 5))),
                "Recommendation": (
                    "❌ Fix Before Launch"
                    if top_issue["negative_rate"] > 0.3
                    else "✅ Safe to Launch"
                )
            })

        sku_risk_df = pd.DataFrame(sku_results)
        st.dataframe(sku_risk_df)

        fig2, ax2 = plt.subplots()
        ax2.barh(sku_risk_df["SKU"], sku_risk_df["Predicted Return Risk"])
        ax2.set_title("Pre-Launch Return Risk by SKU")
        st.pyplot(fig2)

    # =========================
    # AI COPILOT (FREE / RULE-BASED)
    # =========================
    st.markdown("---")
    st.subheader("🤖 Ask the Returns Copilot")

    user_q = st.text_input("Ask a question (e.g. 'Why are returns increasing?')")

    if user_q:
        response = ""

        if "why" in user_q.lower():
            response = (
                f"Returns are mainly driven by **{top_issue['issue']}**, "
                f"causing ~{round(top_issue['negative_rate']*100,1)}% dissatisfaction."
            )
        elif "launch" in user_q.lower():
            response = "Some SKUs need fixes before launch to reduce return risk."
        elif "improve" in user_q.lower():
            response = "Focus on product quality, packaging, and expectation setting."
        else:
            response = "I analyzed your data and identified return risk drivers."

        st.success(response)

else:
    st.info("👆 Upload Reviews & Returns CSV files to activate the Copilot")
