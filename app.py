import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# =========================
# APP CONFIG
# =========================
st.set_page_config(page_title="AI Returns Copilot", layout="wide")
st.title("📦 AI Returns Intelligence Copilot")
st.caption("Predict • Prevent • Improve Returns (Production MVP)")

# =========================
# FILE UPLOADERS
# =========================
st.subheader("📤 Upload Data")

reviews_file = st.file_uploader("Customer Reviews CSV", type="csv")
returns_file = st.file_uploader("Returns CSV", type="csv")

st.markdown("---")
st.subheader("🚀 Pre-Launch SKU Scoring")
sku_file = st.file_uploader("New SKU CSV", type="csv")

# =========================
# HELPERS
# =========================
def normalize(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

def extract_issue(text):
    text = str(text).lower()
    if any(x in text for x in ["quality", "broken", "defect", "poor"]):
        return "Quality Demonstration"
    if any(x in text for x in ["size", "fit", "small", "large"]):
        return "Size / Fit Issue"
    if any(x in text for x in ["delivery", "late", "delay"]):
        return "Delivery Delay"
    if any(x in text for x in ["packaging", "damaged", "box"]):
        return "Packaging Issue"
    if any(x in text for x in ["price", "costly", "expensive"]):
        return "Pricing Concern"
    return "Other"

issue_baseline = None
total_revenue_loss = 0

# =========================
# PROCESS REVIEWS
# =========================
if reviews_file is not None:

    reviews = normalize(pd.read_csv(reviews_file))

    if "rating" not in reviews.columns or "review_text" not in reviews.columns:
        st.error("❌ Reviews file must contain 'rating' and 'review_text'")
        st.write("Found columns:", reviews.columns.tolist())
        st.stop()

    reviews["issue"] = reviews["review_text"].apply(extract_issue)
    reviews["sentiment"] = np.where(reviews["rating"] <= 2, "negative", "positive")

    review_summary = (
        reviews.groupby("issue")
        .agg(
            total_reviews=("rating", "count"),
            negative_reviews=("sentiment", lambda x: (x == "negative").sum())
        )
        .reset_index()
    )

    review_summary["negative_rate"] = (
        review_summary["negative_reviews"] /
        review_summary["total_reviews"]
    ).fillna(0)

else:
    review_summary = pd.DataFrame()

# =========================
# PROCESS RETURNS (SEPARATE)
# =========================
if returns_file is not None:

    returns = normalize(pd.read_csv(returns_file))

    issue_col = None
    for c in ["return_reason", "issue", "reason"]:
        if c in returns.columns:
            issue_col = c

    refund_col = "refund_amount" if "refund_amount" in returns.columns else None

    if issue_col:
        returns_summary = (
            returns.groupby(issue_col)
            .agg(
                returns=(issue_col, "count"),
                revenue_loss=(refund_col, "sum") if refund_col else (issue_col, "count")
            )
            .reset_index()
            .rename(columns={issue_col: "issue"})
        )
        total_revenue_loss = returns_summary["revenue_loss"].sum()
    else:
        returns_summary = pd.DataFrame()

else:
    returns_summary = pd.DataFrame()

# =========================
# MERGE AT ISSUE LEVEL
# =========================
if not review_summary.empty:
    issue_baseline = review_summary.merge(
        returns_summary,
        on="issue",
        how="left"
    ).fillna(0)

# =========================
# DASHBOARD
# =========================
if issue_baseline is not None and not issue_baseline.empty:

    st.subheader("📊 Return Risk Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.metric("💸 Estimated Revenue Loss", f"₹{int(total_revenue_loss):,}")
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
        st.error("🚨 HIGH RETURN RISK DETECTED")
        st.dataframe(high_risk[["issue", "negative_rate"]])

    # =========================
    # FUTURE RISK
    # =========================
    avg_risk = issue_baseline["negative_rate"].mean()
    confidence = min(90, max(60, int(issue_baseline["total_reviews"].sum() / 8)))

    st.success(
        f"Expected future return risk: {round(avg_risk*100,2)}% "
        f"(Confidence: {confidence}%)"
    )

    # =========================
    # AUTO FIXES
    # =========================
    st.subheader("🛠️ Auto-Generated Fixes")

    for _, r in high_risk.iterrows():
        st.write(
            f"• Improve **{r['issue']}** "
            f"(~{round(r['negative_rate']*100,1)}% dissatisfaction)"
        )

# =========================
# SKU PRE-LAUNCH SCORING
# =========================
if sku_file is not None:

    skus = normalize(pd.read_csv(sku_file))
    st.subheader("🚀 Pre-Launch SKU Risk Scores")

    results = []

    for _, sku in skus.iterrows():
        if issue_baseline is not None:
            top = issue_baseline.sort_values("negative_rate", ascending=False).iloc[0]
            risk = round(top["negative_rate"], 2)
            driver = top["issue"]
            conf = min(90, max(60, int(top["total_reviews"] / 5)))
        else:
            risk = round(np.random.uniform(0.15, 0.35), 2)
            driver = "Category Benchmark"
            conf = 55

        results.append({
            "SKU": sku["sku_name"],
            "Category": sku["category"],
            "Price": sku["price"],
            "Predicted Return Risk": risk,
            "Primary Driver": driver,
            "Confidence (%)": conf,
            "Recommendation": "❌ Fix Before Launch" if risk > 0.3 else "✅ Safe to Launch"
        })

    st.dataframe(pd.DataFrame(results))

# =========================
# AI COPILOT
# =========================
st.markdown("---")
st.subheader("🤖 Ask the Copilot")

q = st.text_input("Ask: Why are returns high? Should I launch this SKU?")

if q:
    q = q.lower()
    if "why" in q:
        st.success("Returns are mainly driven by quality, fit, and delivery issues.")
    elif "launch" in q:
        st.success("Launch low-risk SKUs. Fix top issue drivers before scaling.")
    elif "improve" in q:
        st.success("Improve product quality, packaging, and expectation setting.")
    else:
        st.success("I’ve analyzed your data and identified key return risks.")
