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
    # SENTIMENT VISUAL
    # =========================
    st.subheader("😊 Customer Sentiment Distribution")
    sentiment_counts = reviews_df["sentiment"].value_counts()
    st.pyplot(sentiment_counts.plot.pie(autopct="%1.1f%%", ylabel="").figure)

    # =========================
    # THEME EXTRACTION
    # =========================
    vectorizer = TfidfVectorizer(stop_words="english", max_features=20)
    X = vectorizer.fit_transform(reviews_df["clean_review"])
    themes = vectorizer.get_feature_names_out()

    theme_df = pd.DataFrame({
        "Theme": themes,
        "Frequency": np.sum(X.toarray() > 0, axis=0)
    }).sort_values(by="Frequency", ascending=False)

    st.subheader("🗣️ Top Review Themes")
    st.bar_chart(theme_df.set_index("Theme"))

    # =========================
    # SEGMENTS
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

    st.subheader("👥 Customer Segments")
    st.bar_chart(reviews_df["segment_label"].value_counts())

    # =========================
    # RETURNS & REVENUE LOSS
    # =========================
    total_loss = returns_df["amount"].sum()

    loss_by_reason = returns_df.groupby("reason")["amount"].sum().sort_values(ascending=False)
    loss_by_product = returns_df.groupby("product_name")["amount"].sum().sort_values(ascending=False)

    st.subheader("💸 Revenue Loss by Return Reason")
    st.bar_chart(loss_by_reason)

    st.subheader("📦 Revenue Loss by Product")
    st.bar_chart(loss_by_product.head(10))

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

    issue_map_df = pd.DataFrame({
        "Reviews": reviews_df["issue_tag"].value_counts(),
        "Returns": returns_df["issue_tag"].value_counts()
    }).fillna(0)

    st.subheader("🔗 Review vs Return Root Cause Heatmap")
    st.dataframe(issue_map_df.style.background_gradient(cmap="Reds"))

    # =========================
    # RETURN RISK SCORE
    # =========================
    product_reviews = reviews_df.groupby("product_name").agg(
        reviews=("review_text", "count"),
        negative=("sentiment", lambda x: (x == "Negative").sum()),
        rating=("rating", "mean")
    ).reset_index()

    product_returns = returns_df.groupby("product_name").agg(
        returns=("amount", "count")
    ).reset_index()

    risk_df = pd.merge(product_reviews, product_returns, on="product_name", how="left").fillna(0)

    risk_df["risk_score"] = (
        (risk_df["negative"] / risk_df["reviews"]) * 0.5 +
        ((5 - risk_df["rating"]) / 5) * 0.3 +
        (risk_df["returns"] / risk_df["reviews"]) * 0.2
    )

    risk_df = risk_df.sort_values("risk_score", ascending=False)

    st.subheader("⚠️ Product Return Risk Score")
    st.bar_chart(risk_df.set_index("product_name")["risk_score"].head(10))

    # =========================
    # PRE-LAUNCH RISK
    # =========================
    prelaunch_df = reviews_df.groupby("issue_tag").agg(
        negative_rate=("sentiment", lambda x: (x == "Negative").mean())
    ).reset_index()

    st.subheader("🧪 Pre-Launch Return Risk Drivers")
    st.bar_chart(prelaunch_df.set_index("issue_tag"))

    # =========================
    # AUTO TASKS
    # =========================
    task_map = {
        "quality": "Improve product material & QC",
        "delivery": "Fix logistics & packaging",
        "size_fit": "Improve size chart & description",
        "expectation_gap": "Update product images",
        "wrong_item": "Improve warehouse checks"
    }

    tasks = [
        task_map[row["issue_tag"]]
        for _, row in prelaunch_df.iterrows()
        if row["negative_rate"] > 0.3 and row["issue_tag"] in task_map
    ]

    st.subheader("🛠️ Auto-Generated Product Improvement Tasks")
    for t in set(tasks):
        st.warning(t)

    # =========================
    # ALERTS
    # =========================
    st.subheader("🚨 Risk Alerts")
    if risk_df["risk_score"].max() > 0.35:
        st.error("High return risk detected — action required!")
    else:
        st.success("Return risk within safe range")

    # =========================
    # AI COPILOT
    # =========================
    st.subheader("💬 AI Copilot")
    q = st.text_input("Ask about risk, loss, tasks, pre-launch readiness")

    if q:
        q = q.lower()
        if "prelaunch" in q:
            st.dataframe(prelaunch_df)
        elif "task" in q:
            st.write(tasks)
        elif "risk" in q:
            st.dataframe(risk_df.head(5))
        elif "loss" in q:
            st.metric("Revenue Loss", f"₹{total_loss:,}")
        else:
            st.info("Try asking about risk, loss, tasks, or pre-launch issues")
