"""Remote inference client for the HuggingFace Inference API.

Drop-in replacement for classify.predict_sentiment() that calls the
cloud-hosted model instead of loading it locally.

Required .env variables:
    HF_TOKEN    — HuggingFace API token (read from huggingface.co/settings/tokens)
    HF_MODEL_ID — Model repo ID, e.g. "franromlob/roberta-amazon-sentiment"

Usage:
    from src.inference_client import predict_sentiment_remote

    labels = predict_sentiment_remote(["Great product!", "Terrible quality."])
    # → ["positive", "negative"]
"""

import os
import time

import httpx
from dotenv import load_dotenv

load_dotenv()

_HF_TOKEN: str = os.getenv("HF_TOKEN", "")
_MODEL_ID: str = os.getenv("HF_MODEL_ID", "")
_API_URL: str = f"https://api-inference.huggingface.co/models/{_MODEL_ID}"
_HEADERS: dict = {"Authorization": f"Bearer {_HF_TOKEN}"}

# HuggingFace free tier models go to sleep after inactivity.
# On first call they need ~20s to warm up — we retry automatically.
_MAX_RETRIES: int = 3
_RETRY_WAIT: int = 20  # seconds


def _normalize_label(raw_label: str) -> str:
    """Map HuggingFace label string to project labels (positive/neutral/negative)."""
    label = raw_label.lower()
    if "pos" in label:
        return "positive"
    if "neg" in label:
        return "negative"
    return "neutral"


def predict_sentiment_remote(
    texts: list[str],
    timeout: int = 60,
) -> list[str]:
    """Predict sentiment for a list of reviews via HuggingFace Inference API.

    Handles model warm-up retries automatically (free tier models sleep after
    inactivity and need ~20 seconds to reload on first request).

    Args:
        texts  : List of review strings.
        timeout: Per-request HTTP timeout in seconds.

    Returns:
        List of sentiment labels matching input order (positive/neutral/negative).

    Raises:
        EnvironmentError : If HF_TOKEN or HF_MODEL_ID are not set in .env.
        httpx.HTTPError  : If the API returns an unrecoverable error.
    """
    if not _HF_TOKEN or not _MODEL_ID:
        raise EnvironmentError(
            "HF_TOKEN and HF_MODEL_ID must be set in your .env file.\n"
            "Get your token at: https://huggingface.co/settings/tokens"
        )

    for attempt in range(1, _MAX_RETRIES + 1):
        response = httpx.post(
            _API_URL,
            headers=_HEADERS,
            json={"inputs": texts},
            timeout=timeout,
        )

        # 503 = model is loading (cold start) — wait and retry
        if response.status_code == 503 and attempt < _MAX_RETRIES:
            print(f"  ⏳ Model is loading on HuggingFace... retrying in {_RETRY_WAIT}s "
                  f"(attempt {attempt}/{_MAX_RETRIES})")
            time.sleep(_RETRY_WAIT)
            continue

        response.raise_for_status()
        break

    # Response format for batch inputs:
    # [[{"label": "positive", "score": 0.98}, ...], ...]  ← one list per input text
    raw: list = response.json()

    predictions: list[str] = []
    for item in raw:
        # Each item is a list of {label, score} dicts — pick the highest score
        best = max(item, key=lambda x: x["score"])
        predictions.append(_normalize_label(best["label"]))

    return predictions
