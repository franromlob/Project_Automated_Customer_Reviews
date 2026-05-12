"""Deploy fine-tuned RoBERTa model and Gradio Space to HuggingFace Hub.

Steps performed:
    1. Creates a model repository on HuggingFace Hub.
    2. Uploads all files from models/roberta_finetuned/ (~480 MB).
    3. Creates a Gradio Space on HuggingFace.
    4. Uploads spaces/app.py and spaces/requirements.txt to the Space.
    5. Sets MODEL_ID as a Space secret so app.py knows which model to load.

Usage:
    python scripts/deploy_to_hub.py --username YOUR_HF_USERNAME

Requirements:
    pip install huggingface_hub>=0.20.0
    huggingface-cli login   (run once to authenticate)
"""

import argparse
from pathlib import Path

from huggingface_hub import HfApi


# ── Configuration ──────────────────────────────────────────────────────────────

MODEL_REPO_SUFFIX = "roberta-amazon-sentiment"
SPACE_REPO_SUFFIX = "amazon-review-sentiment"
MODEL_LOCAL_PATH = Path("models/roberta_finetuned")
SPACE_APP_PATH = Path("spaces/app.py")
SPACE_REQ_PATH = Path("spaces/requirements.txt")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _validate_paths() -> None:
    for path in (MODEL_LOCAL_PATH, SPACE_APP_PATH, SPACE_REQ_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required path not found: {path}")


# ── Main deployment ───────────────────────────────────────────────────────────

def deploy(username: str) -> None:
    """Run the full deployment pipeline.

    Args:
        username: Your HuggingFace username (e.g. "franromlob").
    """
    _validate_paths()
    api = HfApi()

    model_repo_id = f"{username}/{MODEL_REPO_SUFFIX}"
    space_repo_id = f"{username}/{SPACE_REPO_SUFFIX}"

    # ── Step 1: Upload fine-tuned model ───────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 1 — Uploading fine-tuned model to HuggingFace Hub")
    print(f"{'='*60}")
    print(f"  Target repo : {model_repo_id}")
    print(f"  Local path  : {MODEL_LOCAL_PATH} (~480 MB, may take a few minutes)")

    api.create_repo(repo_id=model_repo_id, repo_type="model", exist_ok=True)
    api.upload_folder(
        folder_path=str(MODEL_LOCAL_PATH),
        repo_id=model_repo_id,
        repo_type="model",
    )
    print(f"\n  ✅ Model live at: https://huggingface.co/{model_repo_id}")

    # ── Step 2: Create Gradio Space ───────────────────────────────────────────
    print(f"\n{'='*60}")
    print("STEP 2 — Creating Gradio Space")
    print(f"{'='*60}")
    print(f"  Target Space: {space_repo_id}")

    api.create_repo(
        repo_id=space_repo_id,
        repo_type="space",
        space_sdk="gradio",
        exist_ok=True,
    )

    # ── Step 3: Upload Space files ────────────────────────────────────────────
    print("\n  Uploading app.py and requirements.txt...")
    api.upload_file(
        path_or_fileobj=str(SPACE_APP_PATH),
        path_in_repo="app.py",
        repo_id=space_repo_id,
        repo_type="space",
    )
    api.upload_file(
        path_or_fileobj=str(SPACE_REQ_PATH),
        path_in_repo="requirements.txt",
        repo_id=space_repo_id,
        repo_type="space",
    )

    # ── Step 4: Set MODEL_ID as Space secret ──────────────────────────────────
    print(f"\n  Setting MODEL_ID secret → {model_repo_id}")
    api.add_space_secret(
        repo_id=space_repo_id,
        key="MODEL_ID",
        value=model_repo_id,
    )

    print(f"\n  ✅ Space live at: https://huggingface.co/spaces/{space_repo_id}")
    print("\n" + "="*60)
    print("🎉 Deployment complete!")
    print(f"\n  Model repo : https://huggingface.co/{model_repo_id}")
    print(f"  Gradio demo: https://huggingface.co/spaces/{space_repo_id}")
    print(f"  API endpoint (REST):")
    print(f"    POST https://{username}-{SPACE_REPO_SUFFIX}.hf.space/run/predict")
    print(f"    Body: {{\"data\": [\"your review text\"]}}")
    print("\n  ⏳ The Space may take 2-3 minutes to build on first launch.")
    print("="*60)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deploy RoBERTa model and Gradio Space to HuggingFace Hub."
    )
    parser.add_argument(
        "--username",
        required=True,
        help="Your HuggingFace username (e.g. franromlob)",
    )
    args = parser.parse_args()
    deploy(args.username)
