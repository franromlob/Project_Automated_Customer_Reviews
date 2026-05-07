"""
Inference script: nlptown/bert-base-multilingual-uncased-sentiment
Predicts 1-5 star ratings on Amazon reviews, then maps to pos/neu/neg.

Mapping:
  1-2 stars -> negative
  3 stars   -> neutral
  4-5 stars -> positive
"""

import pandas as pd
import torch
from transformers import pipeline
from pathlib import Path

# Resolve paths relative to this file so the script works from any directory
DATA_DIR = Path(__file__).parent.parent / "data" / "processed"
INPUT_CSV = DATA_DIR / "reviews_clean.csv"
OUTPUT_CSV = DATA_DIR / "reviews_with_predictions_nlptown.csv"

MODEL_ID = "nlptown/bert-base-multilingual-uncased-sentiment"
BATCH_SIZE = 32   # reduce if running out of RAM on CPU
MAX_LENGTH = 512  # BERT hard limit; reviews exceeding this are truncated


def stars_to_sentiment(label: str) -> str:
    """Map nlptown star label to a three-class sentiment string.

    Args:
        label: Raw model output label, e.g. "1 star" or "4 stars".

    Returns:
        One of "positive", "neutral", or "negative".
    """
    star_map = {
        "1 star": "negative",
        "2 stars": "negative",
        "3 stars": "neutral",   # ambiguous reviews land here
        "4 stars": "positive",
        "5 stars": "positive",
    }
    return star_map[label]


def main() -> None:
    """Run full inference pipeline and save predictions to CSV."""
    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} reviews")

    # Use GPU index 0 when available; -1 tells pipeline to run on CPU
    device = 0 if torch.cuda.is_available() else -1
    device_name = "GPU" if device == 0 else "CPU"
    print(f"Running on: {device_name}")

    classifier = pipeline(
        "text-classification",
        model=MODEL_ID,
        device=device,
        truncation=True,
        max_length=MAX_LENGTH,
    )

    # Replace NaN with empty string to avoid pipeline errors on missing reviews
    texts = df["review_text"].fillna("").tolist()

    print(f"Running inference on {len(texts)} reviews (batch_size={BATCH_SIZE})...")
    results = classifier(texts, batch_size=BATCH_SIZE)

    # Store raw star label, confidence score, and mapped 3-class sentiment
    df["predicted_stars_nlptown"] = [r["label"] for r in results]
    df["predicted_score_nlptown"] = [round(r["score"], 4) for r in results]
    df["predicted_sentiment_nlptown"] = df["predicted_stars_nlptown"].apply(stars_to_sentiment)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved predictions to {OUTPUT_CSV}")
    print("\nPrediction distribution:")
    print(df["predicted_sentiment_nlptown"].value_counts())
    print("\nStar distribution:")
    print(df["predicted_stars_nlptown"].value_counts().sort_index())


if __name__ == "__main__":
    main()
