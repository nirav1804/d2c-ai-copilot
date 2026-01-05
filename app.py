import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

nltk.download('vader_lexicon')

st.set_page_config(page_title="D2C AI Copilot", layout="wide")
st.title("🤖 D2C AI Consumer Intelligence Copilot")

# =========================
# FILE UPLOAD
# =========================
reviews_file = st.file_uploader("Upload Customer Reviews CSV", type=["csv"])
returns_file = st.file_uploader("Upload Returns CSV", type=["csv"])

if reviews_file and returns_file:

    reviews_df = pd.read_csv(reviews_file)
    returns_df = pd.read_csv(returns_file)

    # =========================
    # VALIDATION
    # =========================
    if not {"review_text", "product_name", "rating"}.issubset(reviews_df.columns):
        st.error("Reviews CSV must contain: review_text, product_name, rating")
        st.stop()

    if not {"reason", "amount", "product_name"}.issubset(returns_df.columns):
        st.error("Returns CSV must contain: reason, amount, product_name")
        st.stop()

    # =========================
    # CLEAN TEXT
    # =========================
    def clean_text(text):
        text = str(text).lower()
        return re.sub(r"[^a-z\s]", "", text)

    reviews_df["clean_review"] = reviews_df["review_text"].apply(clean_text)

    # =========================
    # SENTIMENT
    # =========================
    sia = SentimentIntensityAnalyzer()

    def sentiment(text):
        score = sia.polarity_scores(text)["compound"]
        if score > 0.05: return "Positive"
        if score < -0.05: return "Negative"
        return "Neutral"

    reviews_df["sentiment"] = reviews_df["clean_review"].apply(sentiment)

    # =========================
    # THEME EXTRACTION
    # =========================
    vectorizer = TfidfVectorizer(stop_words="english", max_features=30)
    X = vectorizer.fit_transform(reviews_df["clean_review"])
    themes = vectorizer.get_feature_names_out()

    theme_df = pd.DataFrame({
        "Theme": themes,
        "Frequency": np.sum(X.toarray() > 0, axis=0)
    }).sort_values(by="Frequency", ascending=False)

    # =========================
    # CUSTOMER SEGMENTS
    # =========================
    kmeans = KMeans(n_clusters=4, random_state=42)
    reviews_df["segment"] = kmeans.fit_predict(X)

    segment_map = {
        0: "Price Sensitive",
        1: "Brand Loyalists",
        2: "Unhappy First-Time Buyers",
        3: "Quality Seekers"
    }

    reviews_df["segment_label"] = reviews_df["segment"].map(segment_map)

    # =========================
    # RETURNS & REVENUE LOSS
    # =========================
    total_loss = returns_df["amount"].sum()
    loss_by_reason = returns_df.groupby("reason")["amount"].sum().sort_values(ascending=False)
    loss_by_product = returns_df.groupby("product_name")["amount"].sum().sort_values(ascending=False)

    # =========================
    # ISSUE MAPPING
    # =========================
    issue_map = {
        "quality": ["quality", "cheap", "broken", "defective", "damaged"],
        "size_fit": ["size", "fit", "small", "large"],
        "delivery": ["delivery", "late", "delay", "shipping"],
        "wrong_item": ["wrong", "missing", "different"],
        "expectation_gap": ["image", "picture", "expectation"]
    }

    def tag_issue(text):
        for issue, kws in issue_map.items():
            if any(k in text for k in kws):
                return issue
        return "other"

    reviews_df["issue_tag"] = reviews_df["clean_review"].apply(tag_issue)
    returns_df["issue_tag"] = returns_df["reason"].apply(tag_issue)

    issue_mapping_df = pd.DataFrame({
        "Review Complaints": reviews_df["issue_tag"].value_counts(),
        "Return Reasons": returns_df["issue_tag"].value_counts()
    }).fillna(0).astype(int)

    # =========================
    # FUTURE RETURN RISK (POST-LAUNCH)
    # =========================
    product_reviews = reviews_df.groupby("product_name").agg(
        total_reviews=("review_text", "count"),
        negative_reviews=("sentiment", lambda x: (x == "Negative").sum()),
        avg_rating=("rating", "mean")
    ).reset_index()

    product_returns = returns_df.groupby("product_name").agg(
        total_returns=("amount", "count"),
        total_return_value=("amount", "sum")
    ).reset_index()

    risk_df = pd.merge(product_reviews, product_returns, on="product_name", how="left")
    risk_df.fillna(0, inplace=True)

    risk_df["negative_ratio"] = risk_df["negative_reviews"] / risk_df["total_reviews"]
    risk_df["return_risk_score"] = (
        risk_df["negative_ratio"] * 0.5 +
        ((5 - risk_df["avg_rating"]) / 5) * 0.3 +
        (risk_df["total_returns"] / risk_df["total_reviews"]) * 0.2
    )

    risk_df = risk_df.sort_values("return_risk_score", ascending=False)

    # =========================
    # PRE-LAUNCH RISK PREDICTION
    # =========================
    prelaunch_risk = reviews_df.groupby("issue_tag").agg(
        negative_rate=("sentiment", lambda x: (x == "Negative").mean())
    ).reset_index()

    prelaunch_risk["prelaunch_risk_score"] = prelaunch_risk["negative_rate"]

    # =========================
    # AUTO TASK GENERATOR
    # =========================
    task_map = {
        "quality": "Improve material quality & add QC checks",
        "delivery": "Optimize courier SLA & packaging",
        "size_fit": "Update size charts & product descriptions",
        "expectation_gap": "Update images, add real customer photos",
        "wrong_item": "Improve warehouse picking validation"
    }

    tasks = []
    for issue, row in prelaunch_risk.iterrows():
        if row["prelaunch_risk_score"] > 0.3:
            tasks.append(task_map.get(row["issue_tag"], "Investigate issue"))

    # =========================
    # ALERTS
    # =========================
    st.subheader("🚨 Risk Alerts")
    high_risk_products = risk_df[risk_df["return_risk_score"] > 0.35]

    if not high_risk_products.empty:
        st.error("High Return Risk Detected!")
        st.dataframe(high_risk_products[["product_name", "return_risk_score"]])
    else:
        st.success("No critical return risks detected")

    # =========================
    # DASHBOARD
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(reviews_df))
    col2.metric("Total Returns", len(returns_df))
    col3.metric("Revenue Loss (₹)", f"{total_loss:,}")

    st.subheader("📌 Top Review Themes")
    st.dataframe(theme_df.head(10))

    st.subheader("📌 Revenue Loss by Reason")
    st.bar_chart(loss_by_reason)

    st.subheader("⚠️ High Future Return Risk Products")
    st.dataframe(risk_df.head(10))

    st.subheader("🧪 Pre-Launch Return Risk Drivers")
    st.dataframe(prelaunch_risk)

    st.subheader("🛠️ Auto-Generated Product Improvement Tasks")
    st.write(tasks)

    st.subheader("🔗 Review → Return Root Cause Mapping")
    st.dataframe(issue_mapping_df)

    # =========================
    # AI COPILOT
    # =========================
    st.subheader("💬 AI Copilot")
    q = st.text_input("Ask about risk, revenue, tasks, or root causes")

    def copilot(q):
        q = q.lower()
        if "prelaunch" in q or "before launch" in q:
            return prelaunch_risk
        if "task" in q or "fix" in q:
            return tasks
        if "risk" in q:
            return risk_df.head(5)
        if "loss" in q:
            return f"Total revenue loss: ₹{total_loss:,}"
        if "root" in q:
            return issue_mapping_df
        return "Ask about pre-launch risk, tasks, revenue loss, or return risk."

    if q:
        st.write("🤖 Copilot says:")
        st.write(copilot(q))
