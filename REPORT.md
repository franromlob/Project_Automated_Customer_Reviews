# Technical Report — Automated Customer Reviews Analysis

**Project:** Amazon Product Reviews — Business Case  
**Programme:** AI Engineer Bootcamp · IronHack  
**Author:** Francisco M. Romero Lobato
**Date:** May 2026

---

> **How to use this document**  
> Each `##` section maps to one PowerPoint slide.  
> Bullet points → slide body. Tables → slide visuals. _Italic text_ → speaker notes (do not put on slide).

---

## SLIDE 1 — Title

**Automated Customer Review Analysis**  
_An end-to-end NLP pipeline for Amazon product reviews_

- Sentiment Classification · Product Clustering · AI Summarization
- AI Engineer Bootcamp · IronHack · May 2026
- Francisco

---

## SLIDE 2 — The Business Problem

**Manual review analysis doesn't scale**

- Thousands of product reviews arrive daily — impossible to read manually
- Marketing and product teams need fast, consistent, actionable insights
- Three questions every team asks:
  - _What do customers feel about our products?_
  - _Which products belong together?_
  - _What should we tell buyers?_

**This project automates all three — with zero manual labelling**

_Speaker note: Frame this as a real business pain, not a technical exercise. Every e-commerce team faces this._

---

## SLIDE 3 — Solution Architecture

**Three-task NLP pipeline, one web interface**

| Task                         | Technique               | Output                                         |
| ---------------------------- | ----------------------- | ---------------------------------------------- |
| 1 — Sentiment Classification | Transformer fine-tuning | positive / neutral / negative label per review |
| 2 — Product Clustering       | TF-IDF + K-Means        | 5 product meta-categories                      |
| 3 — Review Summarization     | GPT-4o-mini prompting   | Structured recommendation article per category |

**Stack:** Python · HuggingFace Transformers · scikit-learn · OpenAI API · Streamlit

_Speaker note: Show the flow left-to-right: raw reviews → classification → clustering → summarization → web app._

---

## SLIDE 4 — Dataset & Preprocessing

**Amazon Consumer Electronics Reviews**

| Attribute              | Value     |
| ---------------------- | --------- |
| Raw reviews            | 5,000     |
| After cleaning         | **4,805** |
| Unique products        | 23        |
| Brand                  | Amazon    |
| Median review length   | 21 words  |
| 95th percentile length | 80 words  |

**Preprocessing steps:**

1. Drop nulls in `review_text` / `rating` → removed 195 rows
2. Deduplicate on `(review_text, product_name)`
3. Lowercase + strip whitespace
4. Map star ratings → sentiment: 1–2★ = negative · 3★ = neutral · 4–5★ = positive

_Speaker note: The 21-word median is key — it explains why short-text models (Twitter-trained) are a reasonable baseline._

---

## SLIDE 5 — Class Imbalance: The Core Challenge

**The dataset is heavily skewed toward positive reviews**

| Sentiment | Count | Share     |
| --------- | ----- | --------- |
| Positive  | 4,506 | **93.8%** |
| Neutral   | 191   | 4.0%      |
| Negative  | 108   | 2.2%      |

**Why this matters:**

- A model that always predicts "positive" scores 93.8% accuracy — but is useless
- Minority classes (neutral, negative) are the most business-critical predictions
- Solution: **class-weighted loss** during fine-tuning

_Speaker note: This slide sets up why accuracy alone is a misleading metric and why we focus on F1 per class._

---

## SLIDE 6 — Task 1: Model Selection Journey

**Four models evaluated — iterative improvement**

| Model                                  | Domain                      | Output         | Approach                    |
| -------------------------------------- | --------------------------- | -------------- | --------------------------- |
| `distilbert-sst-2`                     | General                     | 2 classes      | Baseline — no neutral class |
| `twitter-roberta`                      | Twitter / social media      | 3 classes      | Zero-shot inference         |
| `twitter-roberta` fine-tuned           | Twitter + our data          | 3 classes      | Class-weighted fine-tuning  |
| `nlptown/bert-multilingual`            | **Amazon · Yelp · Booking** | 5★ → 3 classes | Zero-shot inference         |
| `nlptown/bert-multilingual` fine-tuned | **Amazon + our data**       | 3 classes      | Class-weighted fine-tuning  |

**Key design decision:** Replace nlptown's 5-class head with a 3-class head — keep the domain-aligned encoder, discard the star-rating output layer.

_Speaker note: The model selection narrative is the most interesting part technically. Emphasise domain alignment as the hypothesis._

---

## SLIDE 7 — Task 1: Classification Results

**4-model comparison on 4,805 reviews**

| Model              | Accuracy   | F1 neutral | F1 negative | F1 weighted | F1 macro |
| ------------------ | ---------- | ---------- | ----------- | ----------- | -------- |
| RoBERTa pretrained | 88.82%     | 0.16       | 0.45        | 0.91        | 0.52     |
| RoBERTa fine-tuned | 87.83%     | 0.36       | **0.73**    | 0.91        | **0.68** |
| nlptown pretrained | **91.53%** | 0.35       | 0.58        | **0.93**    | 0.63     |
| nlptown fine-tuned | 90.97%     | **0.38**   | 0.58        | 0.92        | 0.64     |

**Winner per use case:**

- Best overall accuracy → **nlptown pretrained** (91.53%, no training needed, runs on CPU)
- Best negative detection → **RoBERTa fine-tuned** (F1-neg = 0.73)
- Best balanced performance → **RoBERTa fine-tuned** (F1 macro = 0.68)

_Speaker note: Point out that nlptown pretrained beats RoBERTa by 2.7pp accuracy with zero fine-tuning. That's the power of domain alignment. But for detecting complaints, RoBERTa fine-tuned still wins — more valuable in production._

---

## SLIDE 8 — Task 1: Key Insight — Why the Trade-off Exists

**Domain alignment vs. class-weighted learning**

```
nlptown pretrained:   Accuracy 91.5% │ F1-neg 0.58 │ F1-macro 0.63
RoBERTa fine-tuned:   Accuracy 87.8% │ F1-neg 0.73 │ F1-macro 0.68
```

**Why nlptown fine-tuning didn't improve negative F1:**

- Negative recall was already 0.88 — the model detects most negatives
- Bottleneck is **precision** (0.42): model over-predicts "negative" → more false alarms
- Class weights push recall up, but also inflate false positives → net F1 unchanged
- Root cause: only **108 negative samples** — not enough to learn precise boundaries

**Selected model for production:** RoBERTa fine-tuned  
→ Missing a complaint costs more than a false alarm in a support pipeline

_Speaker note: This is the most nuanced technical insight of the project. Precision/recall trade-off under extreme imbalance. Great interview talking point._

---

## SLIDE 9 — Task 2: Product Clustering

**Unsupervised grouping of 23 products into 5 meta-categories**

**Pipeline:**

1. Concatenate reviews per product (TF-IDF, 500 features, bigrams)
2. PCA → 10 components (89.15% variance explained)
3. K-Means, K=2–8 evaluated via elbow + silhouette → **K=5 selected** (score = 0.40)

**Results:**

| Meta-Category           | Products | Reviews | Avg Rating |
| ----------------------- | -------- | ------- | ---------- |
| Fire Tablets            | 12       | 2,818   | 4.56 ⭐    |
| Alexa Devices           | 3        | 1,566   | 4.64 ⭐    |
| E-Readers & Accessories | 5        | 361     | 4.66 ⭐    |
| E-Readers Premium       | 2        | 56      | 4.72 ⭐    |
| Streaming Devices       | 1        | 4       | 5.00 ⭐    |

_Speaker note: No labels were given — the algorithm discovered Amazon's actual product lines. Silhouette 0.40 is moderate but expected with 23 products and overlapping vocabulary._

---

## SLIDE 10 — Task 3: Review Summarization

**GPT-4o-mini generates structured recommendation articles per category**

**Why GPT-4o-mini over BART / T5:**

- No training data required
- Follows structured instructions reliably
- Cost-effective at this scale (~$0.01 per article)

**Prompt structure (5 mandatory sections):**

1. Category overview
2. Top 3 products — differentiators + target user
3. Top complaints per product
4. Product to avoid + reasons
5. Final buying recommendation

**Output:** 4 articles (~500 words each) · Temperature 0.7 · Max 1,000 tokens

_Speaker note: Streaming Devices excluded — only 1 product with 4 reviews, too little signal for a useful article._

---

## SLIDE 11 — Web Application

**Single interface for non-technical users**

| Tab                         | What it does                                       |
| --------------------------- | -------------------------------------------------- |
| **Sentiment Classifier**    | Paste any review → get label + confidence gauge    |
| **Product Categories**      | Explore clusters → metrics, charts, sample reviews |
| **Recommendation Articles** | Read or regenerate AI articles per category        |

**Tech stack:** Streamlit · Plotly · HuggingFace Transformers · OpenAI API

**Run locally:**

```bash
streamlit run app/main.py
# → http://localhost:8501
```

_Speaker note: Demo the app live if possible. The gauge chart for confidence is visually impactful._

---

## SLIDE 12 — Results Summary

**All three tasks delivered**

### Classification

| Metric            | RoBERTa fine-tuned (selected) | nlptown pretrained (runner-up) |
| ----------------- | ----------------------------- | ------------------------------ |
| Accuracy          | 87.83%                        | **91.53%**                     |
| F1 negative       | **0.73**                      | 0.58                           |
| F1 weighted       | 0.91                          | **0.93**                       |
| Training required | Yes (GPU)                     | No (CPU)                       |

### Clustering

- 5 semantically coherent clusters · Silhouette = 0.40
- Matches Amazon's actual product lines with zero supervision

### Summarization

- 4 structured articles generated · All 5 sections present
- Factually grounded in real review content

**Code quality: Pylint 10.00 / 10**

---

## SLIDE 13 — Conclusions & Next Steps

### What we proved

1. **Domain alignment > model size** — nlptown (product-review-trained) beats Twitter-trained RoBERTa on accuracy with zero fine-tuning
2. **Class weighting is essential** for imbalanced NLP — without it, minority classes are invisible
3. **Prompt engineering replaces fine-tuning** for structured generation at small scale
4. **K-Means + TF-IDF is sufficient** for domain-specific clustering when interpretability matters

### Limitations

| Area           | Issue                                | Fix                                                   |
| -------------- | ------------------------------------ | ----------------------------------------------------- |
| Data           | 93.8% positive — extreme imbalance   | Collect balanced dataset or apply SMOTE               |
| Negative class | Only 108 samples → precision ceiling | More negative reviews would break the 0.58 F1 barrier |
| Clustering     | 23 products only                     | Valid at scale with larger catalogues                 |
| Summarization  | No objective metric (ROUGE)          | Collect reference summaries for evaluation            |

### Next steps

- Deploy on Hugging Face Spaces (GPU inference)
- Extend to multi-brand, multi-retailer review sources
- Replace star-rating sentiment labels with human annotation for cleaner ground truth

_Speaker note: The "domain alignment" conclusion is the one to emphasise for interviews. It's a transferable lesson: always check where your pretrained model was trained, not just how big it is._

---

## Appendix A — Classification: Full Metrics

### RoBERTa pretrained (zero-shot)

| Class            | Precision | Recall   | F1       | Support   |
| ---------------- | --------- | -------- | -------- | --------- |
| Positive         | 0.98      | 0.92     | 0.95     | 4,506     |
| Neutral          | 0.13      | 0.20     | 0.16     | 191       |
| Negative         | 0.31      | 0.79     | 0.45     | 108       |
| **Weighted avg** | **0.93**  | **0.89** | **0.91** | **4,805** |

**Accuracy: 88.82%**

### RoBERTa fine-tuned (class weights)

| Class            | Precision | Recall   | F1       | Support   |
| ---------------- | --------- | -------- | -------- | --------- |
| Positive         | 0.99      | 0.88     | 0.94     | 4,506     |
| Neutral          | 0.23      | 0.85     | 0.36     | 191       |
| Negative         | 0.80      | 0.68     | 0.73     | 108       |
| **Weighted avg** | **0.96**  | **0.88** | **0.91** | **4,805** |

**Accuracy: 87.83%**

### nlptown pretrained (zero-shot, star → 3-class mapping)

| Class            | Precision | Recall   | F1       | Support   |
| ---------------- | --------- | -------- | -------- | --------- |
| Positive         | 0.98      | 0.94     | 0.96     | 4,506     |
| Neutral          | 0.28      | 0.45     | 0.35     | 191       |
| Negative         | 0.44      | 0.88     | 0.58     | 108       |
| **Weighted avg** | **0.94**  | **0.92** | **0.93** | **4,805** |

**Accuracy: 91.53%**

### nlptown fine-tuned (class weights, 3-class head)

| Class            | Precision | Recall   | F1       | Support   |
| ---------------- | --------- | -------- | -------- | --------- |
| Positive         | 0.99      | 0.92     | 0.96     | 4,506     |
| Neutral          | 0.29      | 0.55     | 0.38     | 191       |
| Negative         | 0.42      | 0.92     | 0.58     | 108       |
| **Weighted avg** | **0.95**  | **0.91** | **0.92** | **4,805** |

**Accuracy: 90.97%**

---

## Appendix B — Clustering Hyperparameters

| Parameter                      | Value                |
| ------------------------------ | -------------------- |
| TF-IDF max features            | 500                  |
| N-gram range                   | (1, 2)               |
| PCA components (clustering)    | 10 — 89.15% variance |
| PCA components (visualisation) | 2 — 44.55% variance  |
| K-Means K                      | 5                    |
| K-Means n_init                 | 10                   |
| Random state                   | 42                   |

---

## Appendix C — Output Files

| File                                                            | Description                       |
| --------------------------------------------------------------- | --------------------------------- |
| `data/processed/reviews_with_predictions.csv`                   | RoBERTa pretrained predictions    |
| `data/processed/reviews_with_predictions_finetuned.csv`         | RoBERTa fine-tuned predictions    |
| `data/processed/reviews_with_predictions_nlptown.csv`           | nlptown pretrained predictions    |
| `data/processed/reviews_with_predictions_nlptown_finetuned.csv` | nlptown fine-tuned predictions    |
| `data/processed/model_comparison_4models.png`                   | 4-model F1 bar chart              |
| `data/processed/products_clustered.csv`                         | Products with cluster assignments |
| `data/processed/articles_summary.csv`                           | Generated recommendation articles |
