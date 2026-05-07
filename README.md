# Amazon Product Reviews Analyzer

Automated NLP pipeline that classifies sentiment, clusters products into meta-categories, and generates AI-powered recommendation articles from Amazon customer reviews — served via an interactive Streamlit web app.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Dataset](#dataset)
5. [Setup](#setup)
6. [Pipeline](#pipeline)
7. [Web App](#web-app)
8. [Results](#results)
9. [Tech Stack](#tech-stack)
10. [Code Quality](#code-quality)
11. [Notes & Limitations](#notes--limitations)

---

## Project Overview

This project builds an end-to-end NLP pipeline on Amazon product reviews (Consumer Electronics). It addresses three business questions:

| Question | Approach | Model |
|---|---|---|
| Is this review positive, neutral, or negative? | Fine-tuned transformer | RoBERTa |
| What product category does this belong to? | Unsupervised clustering | TF-IDF + K-Means |
| What should I buy in this category? | Generative AI article | GPT-4o-mini |

---

## Architecture

```
Raw CSV
   │
   ▼
src/preprocess.py       → reviews_clean.csv
   │
   ▼
Colab GPU notebook      → reviews_with_predictions_finetuned.csv
(RoBERTa fine-tuned)
   │
   ▼
src/cluster.py          → reviews_clustered.csv
(TF-IDF + K-Means)
   │
   ▼
src/summarize.py        → articles_summary.csv
(GPT-4o-mini)
   │
   ▼
app/main.py             → Streamlit web app
```

---

## Project Structure

```
amazon-reviews-project/
│
├── app/
│   └── main.py                              # Streamlit web application
│
├── data/
│   ├── raw/                                 # Original dataset (not tracked by Git)
│   │   └── amazon_reviews.csv
│   └── processed/                           # Pipeline outputs
│       ├── reviews_clean.csv
│       ├── reviews_with_predictions.csv
│       ├── reviews_with_predictions_finetuned.csv
│       ├── reviews_clustered.csv
│       ├── products_clustered.csv
│       ├── articles_summary.csv
│       ├── articles/                        # Per-category .txt articles
│       └── *.png                            # EDA and evaluation plots
│
├── models/                                  # Saved model weights (not tracked by Git)
│   └── roberta_finetuned/
│
├── notebooks/
│   ├── 01_eda.ipynb                         # Exploratory Data Analysis
│   ├── 02_classification.ipynb              # Classification — results & evaluation
│   ├── 03_clustering.ipynb                  # Product clustering
│   ├── 04_summarization.ipynb               # Article generation
│   └── colab/
│       ├── 02_classification_GPU.ipynb      # RoBERTa zero-shot inference (Colab T4)
│       └── 02_classification_finetuned_GPU.ipynb  # RoBERTa fine-tuning (Colab T4)
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py                        # Data loading and cleaning pipeline
│   ├── classify.py                          # Inference utilities (batch + single)
│   ├── cluster.py                           # TF-IDF + K-Means clustering pipeline
│   └── summarize.py                         # GPT article generation pipeline
│
├── .streamlit/
│   └── config.toml
├── .pylintrc
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Dataset

**Source:** Amazon Product Reviews — Consumer Electronics

| Attribute | Value |
|---|---|
| Raw reviews | 5,000 |
| Clean reviews | 4,805 |
| Unique products | 23 |
| Unique brands | 1 (Amazon) |
| Review length (avg) | 31 words |
| Review length (p95) | 80 words |

**Sentiment distribution (star-rating mapping):**

| Label | Rating | Count | % |
|---|---|---|---|
| Positive | 4–5 ⭐ | 4,506 | 93.8% |
| Neutral | 3 ⭐ | 191 | 4.0% |
| Negative | 1–2 ⭐ | 108 | 2.2% |

> **Class imbalance:** The dataset is heavily skewed toward positive reviews. This was addressed using class weights during fine-tuning.

---

## Setup

### Prerequisites

- Python 3.13
- A virtual environment (recommended)
- OpenAI API key (for article generation only)
- Google Colab access with T4 GPU (for classification fine-tuning only)

### Installation

```bash
# Clone the repository
git clone <repo-url>
cd amazon-reviews-project

# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate       # Mac / Linux
# .venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

### Environment variables

Create a `.env` file in the project root:

```
OPENAI_API_KEY=sk-...
```

---

## Pipeline

### Step 1 — Preprocessing

Cleans the raw dataset and saves `data/processed/reviews_clean.csv`.

```bash
python src/preprocess.py
```

What it does:
- Selects and renames relevant columns
- Drops rows with missing review text or rating
- Removes duplicate reviews
- Lowercases and strips whitespace from review text
- Cleans messy category columns
- Maps star ratings → sentiment labels

---

### Step 2 — Sentiment Classification (GPU)

Run in **Google Colab** (T4 GPU required):

1. Open `notebooks/colab/02_classification_GPU.ipynb` for zero-shot inference, or
   `notebooks/colab/02_classification_finetuned_GPU.ipynb` for fine-tuning.
2. Set runtime: `Runtime → Change runtime type → T4 GPU`
3. Upload `data/processed/reviews_clean.csv` when prompted
4. Run all cells
5. Download the output and save it to `data/processed/reviews_with_predictions_finetuned.csv`

Then open `notebooks/02_classification.ipynb` locally to review evaluation results.

**Model:** `cardiffnlp/twitter-roberta-base-sentiment-latest`

| Model | Accuracy | F1 Neutral | F1 Negative | F1 Weighted |
|---|---|---|---|---|
| RoBERTa pretrained | 88.82% | 0.16 | 0.45 | 0.91 |
| **RoBERTa fine-tuned** | **87.83%** | **0.36** | **0.73** | **0.91** |

Fine-tuning with class weights improved neutral F1 from 0.16 → 0.36 and negative F1 from 0.45 → 0.73, at the cost of a marginal drop in overall accuracy.

---

### Step 3 — Product Clustering

Groups the 23 products into 5 meta-categories using TF-IDF features and K-Means.

```bash
python src/cluster.py
```

Outputs `data/processed/reviews_clustered.csv` and `data/processed/products_clustered.csv`.

See `notebooks/03_clustering.ipynb` for elbow curve, silhouette analysis, and PCA visualization.

---

### Step 4 — Article Generation

Generates one recommendation article per meta-category using GPT-4o-mini.

```bash
python src/summarize.py
```

Requires `OPENAI_API_KEY` set in `.env`. Outputs `data/processed/articles_summary.csv` and individual `.txt` files under `data/processed/articles/`.

See `notebooks/04_summarization.ipynb` for prompt design and output examples.

---

## Web App

Interactive Streamlit app with three sections:

| Tab | Description |
|---|---|
| Sentiment Classifier | Classify any review text in real time using the fine-tuned RoBERTa model |
| Product Categories | Explore sentiment distribution and top products per meta-category |
| Recommendation Articles | Read or regenerate AI-written articles per category |

### Run locally

```bash
streamlit run app/main.py
```

App opens at **http://localhost:8501**

> The classifier tab requires the fine-tuned model at `models/roberta_finetuned/`.
> The articles tab requires `data/processed/articles_summary.csv` (pre-generated or run Step 4).

### Deploy publicly — Streamlit Community Cloud

1. Push the repository to GitHub (ensure `data/processed/` is committed)
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repo
3. Set **Main file path** to `app/main.py`
4. Set **Requirements file** to `requirements_cloud.txt`
5. Add your secret under **Settings → Secrets**:

```toml
OPENAI_API_KEY = "sk-..."
```

> **Note:** The fine-tuned model (`models/roberta_finetuned/`) is excluded from Git.
> On first load the classifier falls back to the pretrained RoBERTa from Hugging Face automatically.
> Streamlit Cloud free tier provides 1 GB RAM — sufficient for the pretrained model on CPU.

---

## Results

### Classification

| Model | Accuracy | F1 (weighted) |
|---|---|---|
| RoBERTa pretrained | 88.82% | 0.91 |
| RoBERTa fine-tuned | **91.96%** | **0.93** |

### Clustering

**K-Means (K=5) — TF-IDF 500 features + PCA 10 components**

| Meta-Category | Products | Reviews | Avg Rating |
|---|---|---|---|
| Fire Tablets | 12 | 2,818 | 4.56 ⭐ |
| Alexa Devices | 3 | 1,566 | 4.64 ⭐ |
| E-Readers & Accessories | 5 | 361 | 4.66 ⭐ |
| E-Readers Premium | 2 | 56 | 4.72 ⭐ |
| Streaming Devices | 1 | 4 | 5.00 ⭐ |

**Overall Silhouette Score: 0.40**

### Key EDA Findings

- Average review: **31 words** (median: 21) — short enough for 128-token truncation
- Most reviewed product: **All-New Fire HD 8 Tablet (Magenta)**
- Most consistent product: **Amazon Echo Plus Silver** (avg: 4.75 ⭐, lowest std)
- Most divisive product: **Fire HD 8 32GB Blue** (std: 0.92)

---

## Tech Stack

| Component | Tool |
|---|---|
| Language | Python 3.13 |
| Data manipulation | Pandas, NumPy |
| Visualization | Plotly, Matplotlib, Seaborn |
| ML utilities | Scikit-learn |
| NLP / Classification | HuggingFace Transformers (RoBERTa) |
| Clustering | TF-IDF + K-Means (Scikit-learn) |
| Dimensionality reduction | PCA (Scikit-learn) |
| Generative AI | OpenAI GPT-4o-mini |
| Web app | Streamlit |
| GPU inference | Google Colab (T4) |
| Linting | Pylint (10.00/10) |

---

## Code Quality

Linting is applied to all Python source files:

```bash
pylint app/main.py src/classify.py src/cluster.py src/preprocess.py src/summarize.py
```

Current score: **10.00/10**

> Notebooks are excluded from linting. Code quality in notebooks is maintained through inline markdown documentation.

---

## Notes & Limitations

- `data/raw/` and `models/` are excluded from Git (large files — add manually after cloning)
- GPU inference via Google Colab is required for the classification pipeline; local CPU inference is slow but possible via the fallback in `src/classify.py`
- Class imbalance (94% positive) is the main challenge — not the model architecture
- GPT-4o-mini article generation is billed per API call; pre-generated articles are included in the repo to avoid unnecessary costs
- Streamlit 1.57+ required for `st.container(border=True)` support
