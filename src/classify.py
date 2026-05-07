"""
Classification utilities for Amazon product reviews sentiment analysis.
Uses a fine-tuned RoBERTa model trained on Twitter data.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# ─── CONSTANTS ────────────────────────────────────────────────────────────────

# Hugging Face model identifier (pretrained)
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Maximum number of tokens per review (our reviews are short, 128 is plenty)
MAX_LENGTH = 128


# ─── FUNCTIONS ────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return GPU if available, otherwise CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_model(model_path: str = None):
    """Load tokenizer, model and the label mapping.

    If model_path is provided and exists, the fine‑tuned model is loaded.
    Otherwise the original pretrained model is used.

    Returns:
        tokenizer  : HuggingFace tokenizer
        model      : HuggingFace model (in eval mode, on the correct device)
        id2label   : dict mapping numeric label → human‑readable sentiment
        label2id   : reverse of id2label
    """
    # Decide which model to load
    if model_path and os.path.isdir(model_path):
        source = model_path
    else:
        source = MODEL_NAME
        if model_path:
            print(f"⚠️  Model path '{model_path}' not found, using pretrained model.")

    # Load the tokenizer and the model
    tokenizer = AutoTokenizer.from_pretrained(source)
    model = AutoModelForSequenceClassification.from_pretrained(source)

    # Retrieve the native label mapping (automatically adjusts to the model)
    id2label = model.config.id2label             # e.g. {0:'negative',1:'neutral',2:'positive'}
    label2id = {v: k for k, v in id2label.items()}  # inverse dictionary

    # Move the model to GPU or CPU and set it to evaluation mode
    device = get_device()
    model = model.to(device)
    model.eval()

    print(f"✅ Model loaded from: {source}")
    print(f"   Device: {device}")
    print(f"   Labels : {id2label}")
    return tokenizer, model, id2label, label2id


def predict_sentiment(
    texts: list,
    tokenizer,
    model,
    id2label: dict,
    batch_size: int = 32
) -> list:
    """Predict sentiment for a list of review texts.

    Args:
        texts     : list of review strings
        tokenizer : HuggingFace tokenizer
        model     : HuggingFace model
        id2label  : mapping from int → label string
        batch_size: number of reviews processed together

    Returns:
        List of sentiment labels (positive/neutral/negative).
    """
    device = get_device()
    predictions = []

    # Process reviews in batches to avoid memory issues
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]

        # Convert text to tensors, truncating and padding to MAX_LENGTH
        encoding = tokenizer(
            batch,
            truncation=True,
            max_length=MAX_LENGTH,
            padding="max_length",
            return_tensors="pt"
        ).to(device)

        # Disable gradient calculations (faster inference, less memory)
        with torch.no_grad():
            outputs = model(**encoding)
            # Get the index of the highest scoring class for each review
            preds = torch.argmax(outputs.logits, dim=1)

        # Convert numeric predictions to human-readable labels
        batch_predictions = [id2label[p.item()] for p in preds]
        predictions.extend(batch_predictions)

    return predictions


def predict_single(text: str, tokenizer, model, id2label: dict) -> dict:
    """Predict sentiment for a single review text and return confidence.

    Args:
        text      : a single review string
        tokenizer : HuggingFace tokenizer
        model     : HuggingFace model
        id2label  : mapping from int → label string

    Returns:
        A dictionary with 'sentiment' and 'confidence' (float 0-1).
    """
    device = get_device()

    # Tokenize the single text (unsqueeze to add batch dimension)
    encoding = tokenizer(
        text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        outputs = model(**encoding)
        # Convert logits to probabilities with softmax
        probs = torch.softmax(outputs.logits, dim=1)
        # Index of the most likely class
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {
        "sentiment": id2label[pred],
        "confidence": round(confidence, 4)
    }


# ─── MAIN (test) ─────────────────────────────────────────────────────────────

def main():
    """Quick sanity check with a few example reviews."""

    test_reviews = [
        "I absolutely love this product, best purchase ever!",
        "It is okay, nothing special but does the job.",
        "Terrible quality, broke after one week. Waste of money."
    ]

    # Try to load the fine‑tuned model; if it doesn't exist, fall back to pretrained.
    fine_tuned_path = "./roberta_finetuned"
    tokenizer, model, id2label, _ = load_model(fine_tuned_path)

    print("\nBatch prediction:")
    print("-" * 50)
    results = predict_sentiment(test_reviews, tokenizer, model, id2label)
    for text, label in zip(test_reviews, results):
        print(f"  {text[:60]}")
        print(f"  → {label}")
        print("-" * 50)

    print("\nSingle prediction with confidence:")
    result = predict_single(test_reviews[0], tokenizer, model, id2label)
    print(f"  {result}")


if __name__ == "__main__":
    main()
