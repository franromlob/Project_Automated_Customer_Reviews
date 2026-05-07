"""
Summarization utilities for Amazon product reviews.
Generates recommendation articles using the OpenAI GPT API.
"""

import os
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

MODEL_NAME = "gpt-4o-mini"
MAX_TOKENS = 1000
TEMPERATURE = 0.7
MIN_REVIEWS = 10

CATEGORIES = [
    "Fire Tablets",
    "Alexa Devices",
    "E-Readers & Accessories",
    "E-Readers Premium"
]


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def get_client() -> OpenAI:
    """Initialize and return OpenAI client.

    Returns:
        OpenAI client instance.
    """
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not found. Set it in the .env file.")
    print("✅ OpenAI client initialized")
    return OpenAI(api_key=api_key)


def prepare_category_data(df: pd.DataFrame, category: str) -> dict:
    """Prepare review data for a given meta-category.

    Args:
        df: Full clustered reviews DataFrame.
        category: Meta-category name.

    Returns:
        Dict with category stats, top products and sample reviews.
    """
    cat_df = df[df["meta_category"] == category].copy()

    # Product-level stats
    product_stats = (
        cat_df.groupby("product_name")
        .agg(
            num_reviews=("review_text", "count"),
            avg_rating=("rating", "mean"),
            pct_positive=("predicted_sentiment_finetuned",
                          lambda x: (x == "positive").mean())
        )
        .query(f"num_reviews >= {MIN_REVIEWS}")
        .sort_values("avg_rating", ascending=False)
        .reset_index()
    )

    # Sample reviews per product
    product_reviews = {}
    for product in product_stats["product_name"].tolist():
        prod_df = cat_df[cat_df["product_name"] == product]

        positive = (
            prod_df[prod_df["predicted_sentiment_finetuned"] == "positive"]
            ["review_text"].head(5).tolist()
        )
        negative = (
            prod_df[prod_df["predicted_sentiment_finetuned"] == "negative"]
            ["review_text"].head(5).tolist()
        )

        product_reviews[product] = {
            "positive":    positive,
            "negative":    negative,
            "avg_rating":  round(
                product_stats[product_stats["product_name"] == product]
                ["avg_rating"].values[0], 2
            ),
            "num_reviews": int(
                product_stats[product_stats["product_name"] == product]
                ["num_reviews"].values[0]
            )
        }

    return {
        "category":        category,
        "total_reviews":   len(cat_df),
        "num_products":    len(product_stats),
        "product_stats":   product_stats,
        "product_reviews": product_reviews
    }


def build_prompt(category_data: dict) -> str:
    """Build a structured prompt for GPT article generation.

    Args:
        category_data: Dict with category stats and product reviews.

    Returns:
        Formatted prompt string.
    """
    category = category_data["category"]
    product_reviews = category_data["product_reviews"]

    products_text = ""
    for i, (product, reviews) in enumerate(product_reviews.items(), 1):
        positive_samples = " | ".join(reviews["positive"][:3])
        negative_samples = " | ".join(reviews["negative"][:3])

        products_text += f"""
Product {i}: {product}
- Average rating: {reviews["avg_rating"]}/5 ({reviews["num_reviews"]} reviews)
- Positive reviews: {positive_samples if positive_samples else "None"}
- Negative reviews: {negative_samples if negative_samples else "None"}
"""

    prompt = f"""
You are a tech product reviewer writing for a consumer advice website.
Based on the following Amazon customer reviews for the category "{category}",
write a recommendation article in English.

PRODUCT DATA:
{products_text}

ARTICLE REQUIREMENTS:
Write a structured article with exactly these sections:

1. CATEGORY OVERVIEW
   Brief introduction to the {category} category (2-3 sentences).

2. TOP 3 PRODUCTS
   For each of the top 3 products by rating:
   - Product name
   - Why it stands out
   - Key differences from the others
   - Best suited for (type of user)

3. TOP COMPLAINTS
   For each top 3 product, list the 2-3 most common complaints
   from negative reviews.

4. PRODUCT TO AVOID
   Name the worst product in this category and explain specifically
   why customers were disappointed, based on the negative reviews.

5. FINAL RECOMMENDATION
   One paragraph summarizing who should buy what in this category.

Keep the tone friendly, informative and helpful.
Use the actual review content to support your points.
Article length: 400-600 words.
"""
    return prompt


def generate_article(client: OpenAI, prompt: str, category: str) -> str:
    """Generate a recommendation article using GPT.

    Args:
        client: OpenAI client instance.
        prompt: Structured prompt for the article.
        category: Category name for logging.

    Returns:
        Generated article as string.
    """
    print(f"Generating article for: {category}...")

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an expert product reviewer who writes "
                    "clear, helpful, and honest consumer advice articles."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        max_tokens=MAX_TOKENS,
        temperature=TEMPERATURE
    )

    article = response.choices[0].message.content
    print(f"✅ Article generated ({len(article)} characters)")
    return article


def save_articles(articles: dict, output_dir: str) -> None:
    """Save generated articles as text files and CSV.

    Args:
        articles: Dict mapping category name to article text.
        output_dir: Directory path for output files.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save individual text files
    for category, article in articles.items():
        filename = category.lower().replace(" ", "_").replace("&", "and")
        filepath = f"{output_dir}/{filename}.txt"
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(f"# {category}\n\n")
            f.write(article)
        print(f"✅ Saved: {filepath}")

    # Save combined CSV
    articles_df = pd.DataFrame([
        {"category": cat, "article": art}
        for cat, art in articles.items()
    ])
    csv_path = f"{output_dir}/../articles_summary.csv"
    articles_df.to_csv(csv_path, index=False)
    print(f"✅ CSV saved: {csv_path}")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main():
    """Run the full summarization pipeline."""
    # Load data
    df = pd.read_csv("data/processed/reviews_clustered.csv")
    print(f"✅ Data loaded: {df.shape}")

    # Initialize client
    client = get_client()

    # Generate articles
    articles = {}
    for category in CATEGORIES:
        category_data = prepare_category_data(df, category)
        prompt = build_prompt(category_data)
        articles[category] = generate_article(client, prompt, category)

    # Save results
    save_articles(articles, "data/processed/articles")
    print("\n✅ Summarization pipeline complete")


if __name__ == "__main__":
    main()
