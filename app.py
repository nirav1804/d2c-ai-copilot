import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import google.generativeai as genai
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

# =========================
# APP CONFIG & AI SETUP
# =========================
st.set_page_config(page_title="AI Returns Intelligence Copilot", layout="wide")

# Securely fetch API Key from Streamlit Secrets
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except:
    st.warning("⚠️ Gemini API Key not found in Secrets. AI features will be limited.")

# =========================
# HELPERS
# =========================
def normalize(df):
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")
    return df

@st.cache_data
def cluster_issues(df, text_col, n_clusters=5):
    tfidf = TfidfVectorizer(stop_words="english", max_features=800)
    X = tfidf.fit_transform(df[text_col].astype(str))
    model = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    df["cluster"] = model.fit_predict(X)
    
    terms = tfidf.get_feature_names_out()
    labels = {}
    for i in range(n_clusters):
        top_terms = model.cluster_centers_[i].argsort()[-3:]
        labels[i] = ", ".join([terms[t] for t in top_terms])
    df["issue"] = df["cluster"].map(labels)
    return df, labels

def get_ai_recommendations(df):
    model = genai.GenerativeModel('gemini-1.5-flash')
    prompt = f"""
    Act as a Retail Strategy Expert. Analyze this return data:
    {df.to_string()}
    
    Suggest a budget allocation (0-50%) for Marketing (m), Product (p), and Ops (o) 
    per issue to maximize revenue recovery. 
    Return ONLY a JSON object: {{"issue_name": {{"m": 10, "p": 30, "o": 10}}, ...}}
    """
    response = model.generate_content(prompt)
    clean_json = re.search(r'\{.*\}', response.text, re.DOTALL).group()
    return json.loads(clean_json)

# =========================
# MAIN APP
# =========================
st.title("📦 AI Returns Intelligence Copilot")
st.caption("CMO-grade intelligence: Returns, Risk & Revenue")

# Sidebar for Business Baseline
with st.sidebar:
    st.header("📈 Global Assumptions")
    aov = st.number_input("Average Order Value (₹)", 500, 10000, 1500)
    monthly_orders = st.number_input("Monthly Orders", 1000, 500000, 10000)

# File Uploaders
col_u1, col_u2 = st.columns(2)
with col_u1:
    reviews_file = st.file_uploader("Upload Reviews CSV", type=["csv"])
with col_u2:
    sku_file = st.file_uploader("Upload New SKU CSV", type=["csv"])

issue_baseline = None

if reviews_file:
    # Process Data
    df_reviews = normalize(pd.read_csv(reviews_file))
    if "review_text" in df_reviews.columns:
        df_reviews, cluster_labels = cluster_issues(df_reviews, "review_text")
        df_reviews["is_negative"] = (df_reviews["rating"] <= 2).astype(int)
        
        issue_baseline = df_reviews.groupby("issue").agg(
            total_reviews=("rating", "count"),
            negative_reviews=("is_negative", "sum")
        ).reset_index()
        issue_baseline["negative_rate"] = issue_baseline["negative_reviews"] / issue_baseline["total_reviews"]
        issue_baseline["monthly_loss"] = issue_baseline["negative_rate"] * monthly_orders * aov

        # Dashboard
        st.subheader("📊 AI-Detected Return Risk Drivers")
        fig_bars = px.bar(issue_baseline, x="negative_rate", y="issue", orientation='h', 
                          title="Negative Rate by Issue Cluster", color="negative_rate")
        st.plotly_chart(fig_bars, use_container_width=True)

        # Scenario Planner
        st.markdown("---")
        c1, c2 = st.columns([3,1])
        with c1: st.subheader("🧪 Strategic Scenario Planner")
        with c2:
            if st.button("🪄 AI Auto-Optimize"):
                recs = get_ai_recommendations(issue_baseline)
                for issue, vals in recs.items():
                    st.session_state[f"m_{issue}"] = vals['m']
                    st.session_state[f"p_{issue}"] = vals['p']
                    st.session_state[f"o_{issue}"] = vals['o']
                st.rerun()

        EFFECTIVENESS = {"marketing": 0.4, "product": 0.8, "ops": 0.6}
        scenario_results = []

        for _, row in issue_baseline.iterrows():
            with st.expander(f"Fix Strategy for: {row['issue']}"):
                col1, col2, col3 = st.columns(3)
                mkt = col1.slider("Marketing %", 0, 50, st.session_state.get(f"m_{row['issue']}", 0), key=f"m_s_{row['issue']}")
                prd = col2.slider("Product %", 0, 50, st.session_state.get(f"p_{row['issue']}", 0), key=f"p_s_{row['issue']}")
                ops = col3.slider("Ops %", 0, 50, st.session_state.get(f"o_{row['issue']}", 0), key=f"o_s_{row['issue']}")

                reduction = (mkt*EFFECTIVENESS["marketing"] + prd*EFFECTIVENESS["product"] + ops*EFFECTIVENESS["ops"])/100
                opt_rate = max(0, row["negative_rate"]*(1-reduction))
                scenario_results.append({
                    "issue": row["issue"],
                    "current_loss": row["monthly_loss"],
                    "optimized_loss": opt_rate * monthly_orders * aov
                })

        sdf = pd.DataFrame(scenario_results)
        st.metric("Total Revenue Recovery Potential", f"₹{int(sdf.current_loss.sum() - sdf.optimized_loss.sum()):,}")

        # AI Copilot Chat
        st.markdown("---")
        st.subheader("🤖 AI Copilot (CMO Advisor)")
        if prompt := st.chat_input("Ask: Which issue is most critical?"):
            st.chat_message("user").write(prompt)
            with st.chat_message("assistant"):
                context = issue_baseline.to_string()
                model = genai.GenerativeModel('gemini-1.5-flash')
                response = model.generate_content(f"Data:\n{context}\n\nQuestion: {prompt}")
                st.write(response.text)

# Pre-launch SKU
if sku_file and issue_baseline is not None:
    st.markdown("---")
    st.subheader("🚀 Pre-Launch Risk Scoring")
    skus = normalize(pd.read_csv(sku_file))
    # Simplified Logic for demonstration
    skus["Risk Score"] = np.random.uniform(0.1, 0.4, len(skus))
    st.dataframe(skus)
