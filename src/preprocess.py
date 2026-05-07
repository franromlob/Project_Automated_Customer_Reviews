"""
Module for loading and preprocessing Amazon product reviews data.
"""

import pandas as pd


# ─── CONSTANTS ───────────────────────────────────────────────────────────────

# Path to the raw CSV file (Input)
RAW_DATA_PATH = "data/raw/amazon_reviews.csv"

# Path where the cleaned data will be saved (Output)
PROCESSED_DATA_PATH = "data/processed/reviews_clean.csv"

# Columns we actually need for our project (the rest are noise)
USEFUL_COLUMNS = [
    "name",
    "brand",
    "categories",
    "primaryCategories",
    "reviews.rating",
    "reviews.text",
    "reviews.title",
    "reviews.doRecommend",
    "reviews.numHelpful",
    "reviews.username",
]


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

# Opens the CSV file and loads it into memory
def load_data(filepath: str) -> pd.DataFrame:
    """Load raw CSV data into a DataFrame."""
    df = pd.read_csv(filepath, low_memory=False)
    print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
    return df

# Remove the columns we don't use (URLs, IDs, etc.)
def select_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """Keep only the columns relevant to our project."""
    df = df[columns].copy()
    print(f"✅ Columns selected: {df.shape[1]} kept")
    return df

# It uses cleaner names without periods (reviews.rating → rating)
def rename_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename columns to simpler snake_case names."""
    df = df.rename(columns={
        "name":                 "product_name",
        "brand":                "brand",
        "categories":           "categories",
        "primaryCategories":    "primary_category",
        "reviews.rating":       "rating",
        "reviews.text":         "review_text",
        "reviews.title":        "review_title",
        "reviews.doRecommend":  "recommended",
        "reviews.numHelpful":   "helpful_votes",
        "reviews.username":     "username",
    })
    print("✅ Columns renamed")
    return df

# Remove rows with no review text or no punctuation
def drop_nulls(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows where the review text or rating is missing."""
    before = len(df)
    df = df.dropna(subset=["review_text", "rating"])
    after = len(df)
    print(f"✅ Nulls dropped: {before - after} rows removed")
    return df

# Remove duplicate reviews
def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicate reviews based on text and product name."""
    before = len(df)
    df = df.drop_duplicates(subset=["review_text", "product_name"])
    after = len(df)
    print(f"✅ Duplicates dropped: {before - after} rows removed")
    return df

# It puts the text in lowercase and removes spaces.
def clean_text(df: pd.DataFrame) -> pd.DataFrame:
    """Basic text cleaning: strip whitespace and lowercase review text."""
    df["review_text"] = (
        df["review_text"]
        .astype(str)       # make sure every value is a string
        .str.strip()       # remove leading/trailing spaces
        .str.lower()       # convert to lowercase
    )
    print("✅ Text cleaned")
    return df

# Convert ⭐ numbers into words: 1-2=negative, 3=neutral, 4-5=positive
def map_sentiment(rating: float) -> str:
    """Map a numeric star rating to a sentiment label.

    1-2 stars  → negative
    3 stars    → neutral
    4-5 stars  → positive
    """
    if rating <= 2:
        return "negative"
    if rating == 3:
        return "neutral"
    return "positive"

# Apply the above function to the entire dataset
def add_sentiment_column(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sentiment mapping to the rating column."""
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["sentiment"] = df["rating"].apply(map_sentiment)
    print("✅ Sentiment column added")
    print(df["sentiment"].value_counts())
    return df

# Save the clean result in data/processed/
def save_data(df: pd.DataFrame, filepath: str) -> None:
    """Save the cleaned DataFrame to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"✅ Clean data saved to: {filepath}")



def clean_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Keep only the first category from the messy categories columns."""
    df["categories"] = (
        df["categories"]
        .astype(str)
        .apply(lambda x: x.split(",")[0].strip())
    )
    # Also clean primary_category (some rows have multiple values)
    df["primary_category"] = (
        df["primary_category"]
        .astype(str)
        .apply(lambda x: x.split(",")[0].strip())
    )
    print("✅ Categories cleaned")
    return df


# ─── MAIN ─────────────────────────────────────────────────────────────────────

# Execute everything in order, from top to bottom
def main():
    """Run the full preprocessing pipeline."""
    df = load_data(RAW_DATA_PATH)
    df = select_columns(df, USEFUL_COLUMNS)
    df = rename_columns(df)
    df = drop_nulls(df)
    df = drop_duplicates(df)
    df = clean_text(df)
    df = clean_categories(df)
    df = add_sentiment_column(df)
    save_data(df, PROCESSED_DATA_PATH)
    print(f"\n🎉 Preprocessing complete! Final shape: {df.shape}")


if __name__ == "__main__":
    main()
