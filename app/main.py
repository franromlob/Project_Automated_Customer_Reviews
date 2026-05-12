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
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Add project root to path — must be before src imports when run directly
sys.path.insert(0, os.getcwd())

# pylint: disable=wrong-import-position
from src.classify import load_model, predict_single, predict_sentiment
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

def build_product_prompt(
    product_name: str,
    positive_reviews: list,
    negative_reviews: list,
    avg_rating,
    num_reviews: int
) -> str:
    """Build a GPT prompt for a single-product recommendation article."""
    pos_text = " | ".join(positive_reviews[:5]) if positive_reviews else "No positive reviews available."
    neg_text = " | ".join(negative_reviews[:5]) if negative_reviews else "No negative reviews available."
    rating_str = (
        f"Average rating: {avg_rating:.2f}/5 — "
        if avg_rating is not None and not pd.isna(avg_rating)
        else ""
    )
    return f"""
You are a product reviewer writing for a consumer advice website.
Based on the following customer reviews for "{product_name}", write a brief product article.

PRODUCT DATA:
- Product: {product_name}
- {rating_str}{num_reviews} customer reviews
- Positive reviews: {pos_text}
- Negative reviews: {neg_text}

ARTICLE REQUIREMENTS:
Write a structured article with exactly these 4 sections:

1. PRODUCT OVERVIEW
   What this product is and what customers mainly use it for (2 sentences).

2. WHAT CUSTOMERS LOVE
   The 3 most praised aspects from positive reviews.

3. MAIN COMPLAINTS
   The 2-3 most common issues from negative reviews.
   If there are no negative reviews, say so honestly.

4. VERDICT
   One short paragraph: who should buy it and who should look elsewhere.

Tone: friendly, honest, helpful. Length: 200-300 words.
"""


def extract_keywords(texts: list, n: int = 6) -> list:
    """Return the N most frequent meaningful words across a list of reviews."""
    import re
    from collections import Counter

    stopwords = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "is", "it", "this", "that", "was", "are", "be", "been",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "i", "my", "me", "we", "our", "you", "your",
        "he", "she", "they", "their", "its", "not", "no", "very", "so", "just",
        "can", "get", "got", "one", "like", "really", "also", "when", "what",
        "how", "all", "more", "some", "than", "then", "them", "from", "into",
        "after", "too", "about", "were", "which", "there", "even", "much",
        "only", "still", "many", "need", "first", "work", "works", "came",
        "come", "time", "give", "take", "know", "able", "want", "make",
        "amazon", "product", "item", "review", "star", "stars", "bought",
        "buy", "use", "used", "using", "great", "good", "love", "nice",
        "well", "price", "would", "could", "thing", "think", "made", "said",
    }
    words = []
    for text in texts:
        words.extend(re.findall(r"\b[a-z]{4,}\b", str(text).lower()))
    filtered = [w for w in words if w not in stopwords]
    return [word for word, _ in Counter(filtered).most_common(n)]


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

tab1, tab2, tab3, tab4 = st.tabs([
    "🔍 Sentiment Classifier",
    "🗂️ Product Categories",
    "📝 Recommendation Articles",
    "📤 Upload & Analyse"
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
    st.subheader("Explore reviews")

    # ── Product selector — all 23 products ───────────────────────────────────
    all_products = sorted(df["product_name"].unique().tolist())
    sel_product = st.selectbox(
        "Select product", all_products,
        label_visibility="collapsed", key="tab2_product"
    )
    prod_df = df[df["product_name"] == sel_product]

    # ── Product metrics + donut ───────────────────────────────────────────────
    donut_col, metrics_col = st.columns([2, 1])

    with donut_col:
        prod_sent = (
            prod_df["predicted_sentiment_finetuned"]
            .value_counts().reset_index()
        )
        prod_sent.columns = ["Sentiment", "Count"]
        fig_prod = px.pie(
            prod_sent, values="Count", names="Sentiment",
            color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
            hole=0.55
        )
        fig_prod.update_traces(textposition="inside", textinfo="percent+label")
        fig_prod.update_layout(
            showlegend=False,
            margin={"t": 10, "b": 10, "l": 0, "r": 0},
            height=240
        )
        st.plotly_chart(fig_prod, use_container_width=True)

    with metrics_col:
        st.metric("Reviews", f"{len(prod_df):,}")
        avg_r = prod_df["rating"].mean()
        st.metric("Avg Rating", f"{avg_r:.2f} ⭐")
        pos_pct = (
            prod_df["predicted_sentiment_finetuned"] == "positive"
        ).mean() * 100
        st.metric("Positive", f"{pos_pct:.1f}%")
        cat_label = prod_df["meta_category"].iloc[0]
        st.caption(f"Category: **{cat_label}**")

    # ── Hashtag keywords ──────────────────────────────────────────────────────
    if "tab2_hashtag" not in st.session_state:
        st.session_state["tab2_hashtag"] = None

    keywords = extract_keywords(prod_df["review_text"].tolist(), n=6)

    if keywords:
        st.caption("**Top keywords — click any to filter reviews:**")
        kw_cols = st.columns(len(keywords) + 1)
        for i, kw in enumerate(keywords):
            is_active = st.session_state["tab2_hashtag"] == kw
            label = f"#{kw} ✓" if is_active else f"#{kw}"
            if kw_cols[i].button(label, key=f"kw_{kw}", use_container_width=True):
                st.session_state["tab2_hashtag"] = (
                    None if is_active else kw
                )

        if st.session_state["tab2_hashtag"]:
            if kw_cols[-1].button(
                "✕ clear", key="kw_clear", use_container_width=True
            ):
                st.session_state["tab2_hashtag"] = None

    # ── Custom keyword search ─────────────────────────────────────────────────
    custom_col, btn_col = st.columns([4, 1])
    custom_input = custom_col.text_input(
        "Or type your own keyword",
        placeholder="e.g. charger, screen, noise…",
        label_visibility="collapsed",
        key="tab2_custom_input"
    )
    if btn_col.button("Search", key="kw_custom_btn", use_container_width=True):
        typed = custom_input.strip().lower()
        if typed:
            st.session_state["tab2_hashtag"] = typed

    # ── Filtered reviews table ────────────────────────────────────────────────
    active_kw = st.session_state["tab2_hashtag"]

    if active_kw:
        review_df = prod_df[
            prod_df["review_text"].str.contains(
                active_kw, case=False, na=False
            )
        ]
        st.caption(f"{len(review_df)} review(s) mentioning **#{active_kw}**")
    else:
        review_df = prod_df.sample(min(15, len(prod_df)), random_state=42)
        st.caption(f"{len(review_df)} sample reviews")

    disp = review_df[
        ["review_text", "rating", "predicted_sentiment_finetuned"]
    ].copy()
    disp.columns = ["Review", "Rating", "Sentiment"]
    st.dataframe(disp, use_container_width=True, hide_index=True)


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
        try:
            api_key = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))
        except Exception:
            api_key = os.getenv("OPENAI_API_KEY")
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


# ─── TAB 4 — UPLOAD & ANALYSE ────────────────────────────────────────────────

with tab4:
    st.subheader("Analyse Your Own Product Reviews")
    st.caption(
        "Upload a CSV with your product reviews. "
        "The app will classify sentiment, group products into categories, "
        "and generate AI-powered recommendation articles."
    )

    uploaded_file = st.file_uploader(
        "Drop your CSV here", type=["csv"], label_visibility="collapsed"
    )

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file)

        if len(df_raw) > 2000:
            st.warning(
                f"Large file ({len(df_raw):,} rows). "
                "Only the first 2,000 reviews will be processed."
            )
            df_raw = df_raw.head(2000)

        with st.expander(
            f"Preview — {len(df_raw):,} rows × {df_raw.shape[1]} columns"
        ):
            st.dataframe(df_raw.head(5), use_container_width=True)

        st.divider()
        st.subheader("Map your columns")
        st.caption(
            "Tell the app which columns contain the product name, "
            "review text, and optionally the star rating."
        )

        all_cols = df_raw.columns.tolist()

        def _guess(cols: list, keywords: list) -> int:
            for kw in keywords:
                for i, col in enumerate(cols):
                    if kw.lower() in col.lower():
                        return i
            return 0

        c1, c2, c3 = st.columns(3)
        col_product = c1.selectbox(
            "Product name column", all_cols,
            index=_guess(all_cols, ["product", "name", "title", "item"]),
            key="up_product"
        )
        col_review = c2.selectbox(
            "Review text column", all_cols,
            index=_guess(all_cols, ["review", "text", "comment", "feedback", "body"]),
            key="up_review"
        )
        rating_opts = ["— none —"] + all_cols
        col_rating = c3.selectbox(
            "Star rating column (optional)", rating_opts,
            index=_guess(rating_opts, ["rating", "stars", "score"]),
            key="up_rating"
        )

        st.divider()

        _api_key = os.getenv("OPENAI_API_KEY")
        try:
            _api_key = st.secrets.get("OPENAI_API_KEY", _api_key)
        except Exception:
            pass
        has_api = bool(_api_key)

        if not has_api:
            st.info(
                "💡 No OpenAI API key found — classification and clustering will run, "
                "but article generation will be skipped. "
                "Add `OPENAI_API_KEY` to your `.env` to enable articles."
            )

        if st.button("🚀 Run Analysis", type="primary"):

            # ── STEP 1: Preprocess ────────────────────────────────────────
            with st.spinner("Step 1 / 3 — Preprocessing…"):
                df_proc = pd.DataFrame()
                df_proc["product_name"] = (
                    df_raw[col_product].astype(str).str.strip()
                )
                df_proc["review_text"] = (
                    df_raw[col_review].astype(str).str.lower().str.strip()
                )
                if col_rating != "— none —":
                    df_proc["rating"] = pd.to_numeric(
                        df_raw[col_rating], errors="coerce"
                    )
                else:
                    df_proc["rating"] = float("nan")

                df_proc = df_proc.dropna(subset=["review_text"])
                df_proc = df_proc[df_proc["review_text"].str.len() > 3]
                df_proc = df_proc.drop_duplicates(
                    subset=["product_name", "review_text"]
                ).reset_index(drop=True)

            # ── STEP 2: Classify ──────────────────────────────────────────
            with st.spinner(
                f"Step 2 / 3 — Classifying {len(df_proc):,} reviews with RoBERTa…"
            ):
                tokenizer_up, model_up, id2label_up, _ = load_classifier()
                preds = predict_sentiment(
                    df_proc["review_text"].tolist(),
                    tokenizer_up, model_up, id2label_up
                )
                df_proc["predicted_sentiment_finetuned"] = preds

            # ── STEP 3: Cluster ───────────────────────────────────────────
            with st.spinner("Step 3 / 3 — Clustering products…"):
                n_products = df_proc["product_name"].nunique()

                product_agg = (
                    df_proc.groupby("product_name")
                    .agg(
                        all_reviews=("review_text", " ".join),
                        num_reviews=("review_text", "count"),
                        avg_rating=("rating", "mean"),
                        pct_positive=(
                            "predicted_sentiment_finetuned",
                            lambda x: (x == "positive").mean()
                        )
                    )
                    .reset_index()
                )
                product_agg["combined_text"] = (
                    product_agg["product_name"] + " " +
                    product_agg["product_name"] + " " +
                    product_agg["all_reviews"]
                )

                if n_products >= 3:
                    tfidf_up = TfidfVectorizer(
                        max_features=500, stop_words="english",
                        ngram_range=(1, 2), min_df=1
                    )
                    mat_up = tfidf_up.fit_transform(
                        product_agg["combined_text"]
                    )
                    n_comp = min(n_products - 1, 10)
                    coords_up = PCA(
                        n_components=n_comp, random_state=42
                    ).fit_transform(mat_up.toarray())

                    best_k, best_score = 2, -1.0
                    for k_try in range(2, min(n_products // 2, 6) + 1):
                        lbs_try = KMeans(
                            n_clusters=k_try, random_state=42, n_init=10
                        ).fit_predict(coords_up)
                        if len(set(lbs_try)) > 1:
                            sc = silhouette_score(coords_up, lbs_try)
                            if sc > best_score:
                                best_score, best_k = sc, k_try

                    final_labels = KMeans(
                        n_clusters=best_k, random_state=42, n_init=10
                    ).fit_predict(coords_up)
                    product_agg["meta_category"] = [
                        f"Group {l + 1}" for l in final_labels
                    ]
                else:
                    best_k = 1
                    product_agg["meta_category"] = "Group 1"

                cat_map = product_agg.set_index(
                    "product_name"
                )["meta_category"].to_dict()
                df_proc["meta_category"] = df_proc["product_name"].map(cat_map)

            # ── ARTICLE GENERATION ────────────────────────────────────────
            arts_product: dict = {}
            arts_category: dict = {}

            if has_api:
                client_up = get_client()
                products_for_articles = product_agg[
                    product_agg["num_reviews"] >= 5
                ]["product_name"].tolist()[:10]

                with st.spinner(
                    f"Generating {len(products_for_articles)} product articles…"
                ):
                    for pname in products_for_articles:
                        p_df = df_proc[df_proc["product_name"] == pname]
                        pos_rev = (
                            p_df[
                                p_df["predicted_sentiment_finetuned"] == "positive"
                            ]["review_text"].head(5).tolist()
                        )
                        neg_rev = (
                            p_df[
                                p_df["predicted_sentiment_finetuned"] == "negative"
                            ]["review_text"].head(5).tolist()
                        )
                        p_row = product_agg[
                            product_agg["product_name"] == pname
                        ].iloc[0]
                        prompt_p = build_product_prompt(
                            pname, pos_rev, neg_rev,
                            p_row["avg_rating"], int(p_row["num_reviews"])
                        )
                        arts_product[pname] = generate_article(
                            client_up, prompt_p, pname
                        )

                categories_up = df_proc["meta_category"].dropna().unique().tolist()
                with st.spinner(
                    f"Generating {len(categories_up)} category articles…"
                ):
                    for cat in categories_up:
                        cat_data = prepare_category_data(df_proc, cat)
                        if cat_data["num_products"] == 0:
                            continue
                        cat_prompt = build_prompt(cat_data)
                        arts_category[cat] = generate_article(
                            client_up, cat_prompt, cat
                        )

            # ── PERSIST IN SESSION STATE ──────────────────────────────────
            st.session_state["up_df"] = df_proc
            st.session_state["up_products"] = product_agg
            st.session_state["up_arts_product"] = arts_product
            st.session_state["up_arts_category"] = arts_category
            st.session_state["up_has_rating"] = col_rating != "— none —"
            st.session_state["up_k"] = best_k

            st.success(
                f"✅ Analysis complete — {len(df_proc):,} reviews · "
                f"{n_products} products · {best_k} group(s) found"
            )

    # ── RESULTS ───────────────────────────────────────────────────────────────
    if "up_df" in st.session_state:
        df_res     = st.session_state["up_df"]
        prod_res   = st.session_state["up_products"]
        arts_prod  = st.session_state["up_arts_product"]
        arts_cat   = st.session_state["up_arts_category"]
        has_rating = st.session_state["up_has_rating"]

        st.divider()

        mc1, mc2, mc3, mc4 = st.columns(4)
        mc1.metric("Total Reviews", f"{len(df_res):,}")
        mc2.metric("Products", f"{df_res['product_name'].nunique():,}")
        mc3.metric("Groups Found", f"{df_res['meta_category'].nunique():,}")
        pos_pct = (
            df_res["predicted_sentiment_finetuned"] == "positive"
        ).mean() * 100
        mc4.metric("Positive Sentiment", f"{pos_pct:.1f}%")

        st.divider()

        r1, r2, r3 = st.tabs([
            "📊 Overview", "📄 Per Product", "🗂️ Per Category"
        ])

        # ── Overview ──────────────────────────────────────────────────────
        with r1:
            rl, rr = st.columns(2)

            with rl:
                st.subheader("Sentiment distribution")
                sent_up = (
                    df_res["predicted_sentiment_finetuned"]
                    .value_counts().reset_index()
                )
                sent_up.columns = ["Sentiment", "Count"]
                fig_up = px.pie(
                    sent_up, values="Count", names="Sentiment",
                    color="Sentiment", color_discrete_map=SENTIMENT_COLORS,
                    hole=0.45
                )
                fig_up.update_traces(
                    textposition="inside", textinfo="percent+label"
                )
                fig_up.update_layout(
                    showlegend=False,
                    margin={"t": 10, "b": 10, "l": 0, "r": 0},
                    height=280
                )
                st.plotly_chart(fig_up, use_container_width=True)

            with rr:
                st.subheader("Groups summary")
                grp = (
                    df_res.groupby("meta_category")
                    .agg(
                        Products=("product_name", "nunique"),
                        Reviews=("review_text", "count")
                    )
                    .reset_index()
                    .rename(columns={"meta_category": "Group"})
                )
                if has_rating:
                    grp_rating = (
                        df_res.groupby("meta_category")["rating"]
                        .mean().round(2).reset_index()
                        .rename(columns={
                            "meta_category": "Group", "rating": "Avg Rating"
                        })
                    )
                    grp = grp.merge(grp_rating, on="Group")
                st.dataframe(grp, use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("All products")
            tbl_cols = ["product_name", "num_reviews", "pct_positive", "meta_category"]
            tbl_rename = {
                "product_name": "Product",
                "num_reviews": "Reviews",
                "pct_positive": "% Positive",
                "meta_category": "Group"
            }
            if has_rating:
                tbl_cols.insert(2, "avg_rating")
                tbl_rename["avg_rating"] = "Avg Rating"

            prod_tbl = prod_res[tbl_cols].rename(columns=tbl_rename).copy()
            prod_tbl["% Positive"] = (
                prod_tbl["% Positive"] * 100
            ).round(1).astype(str) + "%"
            if "Avg Rating" in prod_tbl.columns:
                prod_tbl["Avg Rating"] = prod_tbl["Avg Rating"].round(2)
            st.dataframe(prod_tbl, use_container_width=True, hide_index=True)

        # ── Per Product ───────────────────────────────────────────────────
        with r2:
            if not arts_prod:
                st.info(
                    "Add an OpenAI API key and re-run the analysis "
                    "to generate product articles."
                )
            else:
                sel_prod = st.selectbox(
                    "Product", list(arts_prod.keys()),
                    label_visibility="collapsed", key="sel_prod"
                )
                p_row = prod_res[
                    prod_res["product_name"] == sel_prod
                ].iloc[0]

                pc1, pc2, pc3 = st.columns(3)
                pc1.metric("Reviews", int(p_row["num_reviews"]))
                pc2.metric(
                    "Avg Rating",
                    f"{p_row['avg_rating']:.2f} ⭐"
                    if has_rating and not pd.isna(p_row["avg_rating"])
                    else "N/A"
                )
                pc3.metric("% Positive", f"{p_row['pct_positive']*100:.1f}%")

                with st.container(border=True):
                    st.markdown(arts_prod[sel_prod])

        # ── Per Category ──────────────────────────────────────────────────
        with r3:
            if not arts_cat:
                st.info(
                    "Add an OpenAI API key and re-run the analysis "
                    "to generate category articles."
                )
            else:
                sel_cat = st.selectbox(
                    "Group", list(arts_cat.keys()),
                    label_visibility="collapsed", key="sel_cat"
                )
                cat_rows = df_res[df_res["meta_category"] == sel_cat]

                cc1, cc2, cc3 = st.columns(3)
                cc1.metric("Reviews", f"{len(cat_rows):,}")
                cc2.metric("Products", f"{cat_rows['product_name'].nunique():,}")
                cc3.metric(
                    "Avg Rating",
                    f"{cat_rows['rating'].mean():.2f} ⭐"
                    if has_rating and cat_rows["rating"].notna().any()
                    else "N/A"
                )

                with st.container(border=True):
                    st.markdown(arts_cat[sel_cat])
