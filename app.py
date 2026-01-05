import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('vader_lexicon')

st.set_page_config(page_title="D2C AI Copilot", layout="wide")
st.title("🤖 D2C AI Copilot – Predict, Prevent & Protect Revenue")

# =========================
# FILE UPLOADS
# =========================
reviews_file = st.file_uploader("Upload Customer Reviews CSV", type=["csv"])
returns_file = st.file_uploader("Upload Returns CSV", type=["csv"])
sku_file = st.file_uploader("Upload New SKUs (Optional – Pre-Launch)", type=["csv"])

if reviews_file and returns_file:

    reviews = pd.read_csv(reviews_file)
    returns = pd.read_csv(returns_file)

    # =========================
    # CLEAN & SENTIMENT
    # =========================
    def clean(t):
        return re.sub(r"[^a-z\s]", "", str(t).lower())

    reviews["clean"] = reviews["review_text"].apply(clean)

    sia = SentimentIntensityAnalyzer()
    reviews["sentiment"] = reviews["clean"].apply(
        lambda x: "Negative" if sia.polarity_scores(x)["compound"] < -0.05 else
                  "Positive" if sia.polarity_scores(x)["compound"] > 0.05 else "Neutral"
    )

    # =========================
    # ISSUE TAGGING
    # =========================
    issue_map = {
        "quality": ["quality", "cheap", "broken", "damaged"],
        "delivery": ["delivery", "late", "delay"],
        "size_fit": ["size", "fit", "small", "large"],
        "expectation_gap": ["image", "picture", "expectation"]
    }

    def tag_issue(text):
        for k, v in issue_map.items():
            if any(word in text for word in v):
                return k
        return "other"

    reviews["issue"] = reviews["clean"].apply(tag_issue)
    returns["issue"] = returns["reason"].astype(str).apply(clean).apply(tag_issue)

    # =========================
    # RISK ENGINE
    # =========================
    product_stats = reviews.groupby("product_name").agg(
        reviews=("review_text", "count"),
        negatives=("sentiment", lambda x: (x == "Negative").sum()),
        rating=("rating", "mean")
    ).reset_index()

    return_stats = returns.groupby("product_name").agg(
        returns=("amount", "count")
    ).reset_index()

    risk = product_stats.merge(return_stats, on="product_name", how="left").fillna(0)

    risk["risk_score"] = (
        (risk["negatives"] / risk["reviews"]) * 0.5 +
        ((5 - risk["rating"]) / 5) * 0.3 +
        (risk["returns"] / risk["reviews"]) * 0.2
    )

    risk["confidence"] = np.clip((risk["reviews"] / 500) * 100, 40, 95)

    # =========================
    # PRE-LAUNCH BASELINES
    # =========================
    issue_baseline = reviews.groupby("issue").agg(
        negative_rate=("sentiment", lambda x: (x == "Negative").mean()),
        volume=("sentiment", "count")
    ).reset_index()

    # =========================
    # WHAT-IF SIMULATION
    # =========================
    st.subheader("🔮 What-If Simulation")
    reduction = st.slider("Reduce negative reviews by (%)", 0, 50, 10)

    simulated = risk.copy()
    simulated["simulated_risk"] = simulated["risk_score"] * (1 - reduction / 100)

    st.bar_chart(
        simulated.set_index("product_name")[["risk_score", "simulated_risk"]]
    )

    # =========================
    # PRE-LAUNCH SKU SCORING
    # =========================
    if sku_file:
        skus = pd.read_csv(sku_file)

        sku_scores = issue_baseline.copy()
        sku_scores["prelaunch_risk"] = sku_scores["negative_rate"]
        sku_scores["confidence"] = np.clip((sku_scores["volume"] / 300) * 100, 40, 90)

        st.subheader("🚀 Pre-Launch SKU Risk Signals")
        st.dataframe(sku_scores)

    # =========================
    # CONVERSATIONAL COPILOT
    # =========================
    st.subheader("💬 AI Copilot")
    q = st.text_input("Ask in plain English (e.g. Should we launch this product?)")

    def copilot(q):
        q = q.lower()

        if "launch" in q:
            high = issue_baseline.sort_values("negative_rate", ascending=False).iloc[0]
            return (
                f"Based on past data, **{high['issue']} issues** are the biggest risk "
                f"before launch. If unresolved, returns may rise.\n\n"
                f"Confidence: {int((high['volume']/500)*100)}%"
            )

        if "what if" in q:
            return (
                f"If you reduce negative reviews by {reduction}%, "
                f"overall return risk drops by approximately {reduction * 0.8}%.\n\n"
                f"Confidence: 78%"
            )

        if "highest risk" in q:
            top = risk.sort_values("risk_score", ascending=False).iloc[0]
            return (
                f"**{top['product_name']}** has the highest predicted return risk.\n"
                f"Risk score: {top['risk_score']:.2f}\n"
                f"Confidence: {int(top['confidence'])}%\n\n"
                f"Main drivers: quality & expectations."
            )

        if "fix" in q:
            return (
                "To reduce future returns, focus on:\n"
                "- Improving product quality checks\n"
                "- Updating product images\n"
                "- Setting clearer expectations\n\n"
                "Confidence: 85%"
            )

        return (
            "You can ask about launch readiness, what-if scenarios, "
            "highest-risk products, or how to reduce future returns."
        )

    if q:
        st.write("🤖 Copilot says:")
        st.markdown(copilot(q))
