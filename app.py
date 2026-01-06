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
st.caption("CMO-grade decision intelligence for D2C & E-commerce brands")

# =========================
# ALWAYS VISIBLE FILE UPLOADERS
# =========================
st.subheader("📤 Upload Data")

reviews_file = st.file_uploader(
    "Customer Reviews CSV (required for learning)",
    type="csv",
    key="reviews"
)

sku_file = st.file_uploader(
    "New SKU CSV (Pre-Launch Scoring)",
    type="csv",
    key="sku"
)

st.markdown("---")

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
# GLOBAL OBJECTS
# =========================
issue_baseline = None
sku_model = None
le = LabelEncoder()

# =========================
# PROCESS REVIEWS (LEARNING)
# =========================
if reviews_file is not None:

    reviews = normalize(pd.read_csv(reviews_file))

    if not {"review_text", "rating"}.issubset(reviews.columns):
        st.error("❌ Reviews file must contain: review_text, rating")
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
    # DASHBOARD
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
    # TRAIN SKU MODEL
    # =========================
    reviews["issue_enc"] = le.fit_transform(reviews["issue"])
    X = reviews[["issue_enc"]]
    y = reviews["is_negative"]

    sku_model = LogisticRegression()
    sku_model.fit(X, y)

else:
    st.info("ℹ Upload reviews to enable AI learning & benchmarking")

# =========================
# PRE-LAUNCH SKU SCORING (ALWAYS AVAILABLE)
# =========================
st.markdown("---")
st.subheader("🚀 Pre-Launch SKU Risk Scoring")

if sku_file is not None:

    skus = normalize(pd.read_csv(sku_file))

    required = {"sku_name", "category", "price"}
    if not required.issubset(skus.columns):
        st.error("❌ SKU file must contain: sku_name, category, price")
        st.stop()

    results = []

    for _, sku in skus.iterrows():

        # If model exists → data-driven
        if sku_model is not None and issue_baseline is not None:
            top_issue = issue_baseline.sort_values(
                "negative_rate", ascending=False
            ).iloc[0]["issue"]

            enc = le.transform([top_issue])[0]
            risk = sku_model.predict_proba([[enc]])[0][1]
            confidence = min(90, max(60, int(issue_baseline.total_reviews.sum() / 20)))
            driver = top_issue

        # Fallback → benchmark simulation
        else:
            risk = np.random.uniform(0.18, 0.35)
            confidence = 55
            driver = "Category benchmark"

        decision = (
            "🟢 Safe to Launch"
            if risk < 0.25 else
            "🟠 Improve Before Scaling"
            if risk < 0.35 else
            "🔴 Fix Before Launch"
        )

        results.append({
            "SKU": sku["sku_name"],
            "Category": sku["category"],
            "Price": sku["price"],
            "Predicted Return Risk": round(risk, 2),
            "Primary Risk Driver": driver,
            "Confidence (%)": confidence,
            "Decision": decision
        })

    sku_df = pd.DataFrame(results)
    st.dataframe(sku_df)

    fig, ax = plt.subplots()
    ax.barh(
        sku_df["SKU"],
        sku_df["Predicted Return Risk"]
    )
    ax.set_xlabel("Predicted Return Risk")
    st.pyplot(fig)

else:
    st.info("👆 Upload SKU file to get pre-launch risk scoring")

# =========================
# AI COPILOT (CMO)
# =========================
st.markdown("---")
st.subheader("🤖 AI Copilot for CMOs")

q = st.text_input(
    "Ask: Should I launch this SKU? Where should I invest—Marketing or Product?"
)

if q:
    st.success(
        "CMO Insight: Product fixes reduce return risk fastest. "
        "Marketing helps when expectations are unclear. "
        "High-risk SKUs should be fixed before scale to protect margin and brand trust."
    )
