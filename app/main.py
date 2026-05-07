"""
Web application for Amazon Product Reviews Analysis.
Displays sentiment classification, product clustering,
and GPT-generated recommendation articles.
"""

import os
import sys

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv

# Add project root to path — must be before src imports when run directly
sys.path.insert(0, os.getcwd())

# pylint: disable=wrong-import-position
from src.classify import load_model, predict_single
from src.summarize import (
    get_client,
    prepare_category_data,
    build_prompt,
    generate_article
)
# pylint: enable=wrong-import-position

# ─── CONFIG ───────────────────────────────────────────────────────────────────

load_dotenv()

st.set_page_config(
    page_title="Amazon Reviews Analyzer",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .block-container { padding-top: 1.8rem; padding-bottom: 2rem; }
    [data-testid="stMetricValue"] { font-size: 1.4rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.9rem; padding: 0.5rem 1.2rem; }
</style>
""", unsafe_allow_html=True)

SENTIMENT_COLORS = {
    "positive": "#2ecc71",
    "neutral":  "#f39c12",
    "negative": "#e74c3c"
}

SENTIMENT_ICONS = {
    "positive": "🟢",
    "neutral":  "🟡",
    "negative": "🔴"
}

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

@st.cache_data
def load_data() -> pd.DataFrame:
    """Load clustered reviews dataset."""
    return pd.read_csv("data/processed/reviews_clustered.csv")


@st.cache_data
def load_articles() -> pd.DataFrame:
    """Load pre-generated recommendation articles."""
    return pd.read_csv("data/processed/articles_summary.csv")


@st.cache_resource
def load_classifier():
    """Load fine-tuned RoBERTa model for sentiment classification."""
    return load_model(model_path="models/roberta_finetuned")


# ─── SIDEBAR ──────────────────────────────────────────────────────────────────

df_global = load_data()

with st.sidebar:
    st.markdown("## 🛍️ Amazon Reviews")
    st.caption("Sentiment · Clustering · AI Articles")
    st.divider()

    st.metric("Total reviews", f"{len(df_global):,}")
    st.metric("Unique products", f"{df_global['product_name'].nunique():,}")
    st.metric("Avg rating", f"{df_global['rating'].mean():.2f} ⭐")

    st.divider()
    st.caption("**Sentiment breakdown**")

    sent_counts = df_global["predicted_sentiment_finetuned"].value_counts()
    total = len(df_global)
    for sent in ["positive", "neutral", "negative"]:
        pct = sent_counts.get(sent, 0) / total * 100
        st.caption(f"{SENTIMENT_ICONS[sent]} {sent.capitalize()}: **{pct:.1f}%**")

    st.divider()
    st.caption("Powered by **RoBERTa** · **K-Means** · **GPT-4o-mini**")


# ─── HEADER ───────────────────────────────────────────────────────────────────

st.title("Amazon Product Reviews Analyzer")
st.caption("Automated analysis of Amazon customer reviews across 5 product categories.")
st.divider()

# ─── TABS ─────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs([
    "🔍 Sentiment Classifier",
    "🗂️ Product Categories",
    "📝 Recommendation Articles"
])


# ─── TAB 1 — SENTIMENT CLASSIFIER ────────────────────────────────────────────

with tab1:
    st.subheader("Classify a Review")
    st.caption(
        "Paste any product review and the fine-tuned RoBERTa model "
        "will classify it as positive, neutral, or negative."
    )

    review_input = st.text_area(
        label="Review text",
        placeholder="e.g. I love this tablet, battery life is amazing!",
        height=130,
        label_visibility="collapsed"
    )

    if st.button("Analyse", type="primary"):
        if not review_input.strip():
            st.warning("Please enter a review before clicking Analyse.")
        else:
            with st.spinner("Classifying..."):
                try:
                    tokenizer, model, id2label, _ = load_classifier()
                    result = predict_single(review_input, tokenizer, model, id2label)
                    sentiment = result["sentiment"]
                    confidence = result["confidence"]

                    c1, c2, c3 = st.columns([1, 1, 2])
                    c1.metric(
                        "Sentiment",
                        f"{SENTIMENT_ICONS[sentiment]} {sentiment.capitalize()}"
                    )
                    c2.metric("Confidence", f"{confidence * 100:.1f}%")

                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number",
                        value=round(confidence * 100, 1),
                        number={"suffix": "%", "font": {"size": 22}},
                        gauge={
                            "axis": {"range": [0, 100], "tickwidth": 1},
                            "bar": {"color": SENTIMENT_COLORS[sentiment]},
                            "steps": [
                                {"range": [0, 50],  "color": "#f8f9fa"},
                                {"range": [50, 75], "color": "#e9ecef"},
                                {"range": [75, 100],"color": "#dee2e6"},
                            ],
                            "threshold": {
                                "line": {"color": "#343a40", "width": 2},
                                "thickness": 0.75,
                                "value": confidence * 100
                            }
                        }
                    ))
                    fig_gauge.update_layout(
                        height=180,
                        margin={"t": 20, "b": 0, "l": 10, "r": 10}
                    )
                    c3.plotly_chart(fig_gauge, use_container_width=True)

                except (OSError, RuntimeError, ValueError) as e:
                    st.error(f"Model error: {e}")
                    st.info(
                        "Make sure the fine-tuned model exists in `models/roberta_finetuned`. "
                        "Run the Colab fine-tuning notebook first."
                    )


# ─── TAB 2 — PRODUCT CATEGORIES ──────────────────────────────────────────────

with tab2:
    df = load_data()

    categories = sorted(df["meta_category"].dropna().unique())
    selected_category = st.selectbox(
        "Category", categories, label_visibility="collapsed"
    )

    cat_df = df[df["meta_category"] == selected_category]

    c1, c2, c3 = st.columns(3)
    c1.metric("Reviews", f"{len(cat_df):,}")
    c2.metric("Products", f"{cat_df['product_name'].nunique():,}")
    c3.metric("Avg Rating", f"{cat_df['rating'].mean():.2f} ⭐")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Sentiment split")
        sent_counts = (
            cat_df["predicted_sentiment_finetuned"]
            .value_counts()
            .reset_index()
        )
        sent_counts.columns = ["Sentiment", "Count"]

        fig_pie = px.pie(
            sent_counts,
            values="Count",
            names="Sentiment",
            color="Sentiment",
            color_discrete_map=SENTIMENT_COLORS,
            hole=0.45
        )
        fig_pie.update_traces(textposition="inside", textinfo="percent+label")
        fig_pie.update_layout(
            showlegend=False,
            margin={"t": 10, "b": 10, "l": 0, "r": 0},
            height=280
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.subheader("Top products by rating")
        top_products = (
            cat_df.groupby("product_name")["rating"]
            .agg(["mean", "count"])
            .rename(columns={"mean": "Avg Rating", "count": "Reviews"})
            .query("Reviews >= 10")
            .sort_values("Avg Rating", ascending=False)
            .head(5)
            .reset_index()
        )
        top_products["product_name"] = top_products["product_name"].str[:42]

        fig_bar = px.bar(
            top_products,
            x="Avg Rating",
            y="product_name",
            orientation="h",
            text=top_products["Avg Rating"].round(2),
            color="Avg Rating",
            color_continuous_scale=["#e74c3c", "#f39c12", "#2ecc71"],
            range_color=[3, 5]
        )
        fig_bar.update_traces(texttemplate="%{text}", textposition="outside")
        fig_bar.update_layout(
            xaxis={"range": [0, 5.8], "title": ""},
            yaxis={"title": "", "autorange": "reversed"},
            coloraxis_showscale=False,
            margin={"t": 10, "b": 10, "l": 0, "r": 50},
            height=280
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.divider()
    st.subheader("Sample reviews")

    sample = cat_df[
        ["product_name", "review_text", "rating", "predicted_sentiment_finetuned"]
    ].sample(min(10, len(cat_df)), random_state=42).copy()
    sample.columns = ["Product", "Review", "Rating", "Sentiment"]
    sample["Product"] = sample["Product"].str[:42]

    st.dataframe(sample, use_container_width=True, hide_index=True)


# ─── TAB 3 — RECOMMENDATION ARTICLES ─────────────────────────────────────────

with tab3:
    articles_df = load_articles()
    article_categories = articles_df["category"].tolist()

    selected_article = st.selectbox(
        "Category", article_categories, label_visibility="collapsed"
    )

    article_text = articles_df[
        articles_df["category"] == selected_article
    ]["article"].values[0]

    with st.container(border=True):
        st.markdown(article_text)

    st.divider()

    col_btn, col_info = st.columns([1, 3])
    with col_btn:
        regen = st.button(
            "🔄 Regenerate with GPT", type="secondary", use_container_width=True
        )
    with col_info:
        st.caption("Generate a fresh article for this category using the latest review data.")

    if regen:
        api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        if not api_key:
            st.error(
                "OPENAI_API_KEY not found. "
                "Add it to .env (local) or Streamlit secrets (cloud)."
            )
        else:
            with st.spinner(f"Generating article for {selected_article}..."):
                df_clustered = load_data()
                client = get_client()
                category_data = prepare_category_data(df_clustered, selected_article)
                prompt = build_prompt(category_data)
                new_article = generate_article(client, prompt, selected_article)
                st.success("Article regenerated!")
                with st.container(border=True):
                    st.markdown(new_article)
