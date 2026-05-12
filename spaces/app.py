"""Gradio Space: Amazon Review Sentiment Classifier.

Loads the fine-tuned RoBERTa model from HuggingFace Hub and exposes it as an
interactive demo plus a REST API endpoint.

Environment variable required in Space settings:
    MODEL_ID  — e.g. "your-username/roberta-amazon-sentiment"
"""

import os

import gradio as gr
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# ── Model configuration ────────────────────────────────────────────────────────

MODEL_ID = os.environ.get("MODEL_ID", "FranMRL/roberta-amazon-sentiment")
MAX_LENGTH = 128

# Load once at startup (HuggingFace Spaces caches this between requests)
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_ID)
model.eval()

id2label: dict = model.config.id2label  # {0: "negative", 1: "neutral", 2: "positive"}


# ── Inference ─────────────────────────────────────────────────────────────────

def predict(review_text: str) -> dict:
    """Predict sentiment for a single review text.

    Args:
        review_text: Raw Amazon product review string.

    Returns:
        dict with keys 'sentiment' (str) and 'confidence' (float 0-1).
    """
    inputs = tokenizer(
        review_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
        return_tensors="pt",
    )
    with torch.no_grad():
        outputs = model(**inputs)
        probs = torch.softmax(outputs.logits, dim=1)
        pred = torch.argmax(probs, dim=1).item()
        confidence = probs[0][pred].item()

    return {"sentiment": id2label[pred], "confidence": round(confidence, 4)}


# ── Gradio Interface ───────────────────────────────────────────────────────────

demo = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(
        label="Review text",
        placeholder="Enter an Amazon product review...",
        lines=3,
    ),
    outputs=gr.JSON(label="Prediction"),
    title="🛍️ Amazon Review Sentiment Classifier",
    description=(
        "Fine-tuned RoBERTa model trained on Amazon product reviews. "
        "Returns sentiment (positive / neutral / negative) and a confidence score.\n\n"
        "**API usage:** POST to `/run/predict` with `{\"data\": [\"your review text\"]}`"
    ),
    examples=[
        ["I absolutely love this product, best purchase ever!"],
        ["It is okay, nothing special but does the job."],
        ["Terrible quality, broke after one week. Waste of money."],
        ["Fast shipping and exactly as described. Happy with the purchase."],
        ["Doesn't match the product description at all. Very disappointed."],
    ],
    allow_flagging="never",
    theme=gr.themes.Soft(),
)

if __name__ == "__main__":
    demo.launch()
