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

st.title("🤖 D2C AI Consumer Insights Copilot")
st.caption("Upload customer reviews & returns to uncover revenue leakage")

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
    if "review_text" not in reviews_df.columns:
        st.error("Reviews file must contain 'review_text'")
        st.stop()

    if "amount" not in returns_df.columns or "reason" not in returns_df.columns:
        st.error("Returns file must contain 'amount' and 'reason'")
        st.stop()

    # =========================
    # CLEAN TEXT
    # =========================
    def clean_text(text):
        text = str(text).lower()
        text = re.sub(r"[^a-z\s]", "", text)
        return text

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
    # SEGMENTATION
    # =========================
    kmeans = KMeans(n_clusters=4, random_state=42)
    reviews_df["segment"] = kmeans.fit_predict(X)

    segment_map = {
        0: "Price Sensitive Buyers",
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
    # DASHBOARD
    # =========================
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Reviews", len(reviews_df))
    col2.metric("Total Returns", len(returns_df))
    col3.metric("Revenue Loss (₹)", f"{total_loss:,}")

    st.subheader("📌 Top Customer Issues")
    st.dataframe(theme_df.head(10))

    st.subheader("📌 Revenue Loss by Reason")
    st.bar_chart(loss_by_reason)

    st.subheader("📌 Revenue Loss by Product")
    st.bar_chart(loss_by_product)

    # =========================
    # AI COPILOT
    # =========================
    st.subheader("💬 AI Copilot")
    user_q = st.text_input("Ask a question about your business")

    def copilot_answer(q):
        q = q.lower()

        if "unhappy" in q or "negative" in q:
            return theme_df.head(5)

        if "fix" in q or "priority" in q:
            return f"Fix these first: {', '.join(theme_df.head(3)['Theme'])}"

        if "returns" in q and "why" in q:
            return loss_by_reason.head(5)

        if "revenue" in q or "loss" in q:
            return f"Total revenue loss due to returns: ₹{total_loss:,}"

        if "product" in q and "loss" in q:
            return loss_by_product.head(5)

        if "segment" in q:
            return reviews_df["segment_label"].value_counts()

        if "summary" in q:
            pos = (reviews_df["sentiment"] == "Positive").mean() * 100
            neg = (reviews_df["sentiment"] == "Negative").mean() * 100
            return (
                f"{pos:.1f}% positive sentiment\n"
                f"{neg:.1f}% negative sentiment\n"
                f"Revenue loss: ₹{total_loss:,}"
            )

        return "Try asking about unhappy customers, returns, revenue loss, segments, or summary."

    if user_q:
        st.write("🤖 Copilot says:")
        st.write(copilot_answer(user_q))
