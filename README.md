# NLP Insights from Unstructured Data: Amazon Fine Food Reviews

An end-to-end NLP pipeline that extracts actionable insights from ~30,000 unstructured
customer reviews — combining sentiment analysis, transformer benchmarking, and
unsupervised topic modeling to identify what drives customer dissatisfaction.

**[Live Dashboard](#)** 

---

## Overview

Customer reviews are unstructured text with no built-in categories — this project
answers two practical questions from that raw text:
1. **What sentiment approach actually works well**, and why?
2. **What specific themes drive negative reviews**, beyond just "the food is bad"?

Rather than picking one model and reporting a single accuracy number, this project
benchmarks three sentiment approaches head-to-head on the same data, and uses the
mismatches between them to explain *why* one architecture beats another.

## Dataset

[Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews)
(Kaggle) — ~500K reviews, 1999–2012. A 30,000-review sample was used for modeling
speed; the pipeline is designed to scale to the full dataset given more compute/time.

## Pipeline

```
Raw reviews (Text, Score)
        │
        ▼
Text cleaning (lowercase, strip HTML/punctuation, tokenize)
        │
        ├──► Sentiment Analysis (3 approaches benchmarked, see below)
        │
        └──► Topic Modeling (BERTopic: embed → UMAP → HDBSCAN → c-TF-IDF)
                    │
                    ▼
        Merge sentiment + topics → identify which themes skew negative
                    │
                    ▼
        Interactive Streamlit dashboard
```

## Key Findings

### 1. Model architecture matters as much as model quality
Three sentiment approaches were benchmarked on identical held-out reviews:

| Model | Accuracy | Macro F1 | Notes |
|---|---|---|---|
| TF-IDF + Logistic Regression | 0.71 | 0.55 | Struggles with contrastive reviews ("delicious *but* too expensive") |
| Binary Transformer + confidence threshold | 0.72 | 0.49 | Fixed negative recall (0.66→0.91) but broke neutral (recall 0.07) |
| **Native 3-class Transformer** | **0.81** | **0.62** | Best on every metric — trained on 3 classes natively, no threshold hack |

**Takeaway:** naively bolting a "neutral" threshold onto a binary sentiment model
performed *worse* on macro F1 than the original TF-IDF baseline, despite using a far
more sophisticated model. A model trained natively on 3 classes was necessary to
actually fix the problem — proving that architecture fit matters more than raw model
sophistication.

### 2. Meat/jerky products and order fulfillment drive disproportionate complaints
BERTopic discovered 26 topics from 8,000 reviews (unsupervised, no manual labeling).
Merging topics with sentiment showed:

| Topic | Top words | % Negative | vs. 14.4% baseline |
|---|---|---|---|
| Jerky/meat | jerky, beef, meat | **24.0%** | +9.6 pts |
| Product/flavor | product, flavor, tastes | 22.2% | +7.8 pts |
| Order/fulfillment | product, price, order, amazon, service | 19.9% | +5.5 pts |

Meat/jerky products had a negative sentiment rate nearly double the baseline. A
separate topic dominated by "order, amazon, service, price" suggests a meaningful
share of complaints are about **fulfillment and pricing, not food quality** —
an actionable distinction a naive "negative reviews = bad product" read would miss.

### 3. Sentiment trends are only reliable from ~2006 onward
Early years (2000–2004) had very low review volume (single digits to low hundreds),
causing wild swings in sentiment percentage that aren't statistically meaningful.
From 2006–2012, once volume scaled into the thousands per year, negative sentiment
rose steadily from ~8% to ~16% — a real trend worth flagging, but only once volume
is large enough to trust.

## Tech Stack

- **Preprocessing:** pandas, re, nltk
- **Classical ML baseline:** scikit-learn (TF-IDF, Logistic Regression)
- **Transformers:** Hugging Face `transformers` (`distilbert-base-uncased-finetuned-sst-2-english`,
  `cardiffnlp/twitter-roberta-base-sentiment`)
- **Topic modeling:** BERTopic (sentence-transformers + UMAP + HDBSCAN + c-TF-IDF)
- **Keyword extraction:** KeyBERT
- **Dashboard:** Streamlit, Plotly, WordCloud

## Project Structure

```
├── tfidf.ipynb             # Preprocessing + TF-IDF baseline sentiment model
├── bi_transformer.ipynb           # Binary transformer sentiment + threshold hack
├── native3.ipynb        # Native 3-class transformer sentiment
├── topic.ipynb       # BERTopic + KeyBERT + topic-sentiment merge
├── dashboard.py             # Streamlit dashboard
└── README.md
```

## Running Locally

```bash
pip install pandas scikit-learn transformers torch bertopic keybert \
            sentence-transformers streamlit plotly wordcloud

# Download Reviews.csv from Kaggle and place it in this folder, then:
tfidf.ipynb
bi_transformer.ipynb
native3.ipynb
topic.ipynb
streamlit run dashboard.py
```

## Limitations & Honest Caveats

- Modeled on a 30K (sentiment) / 8K (topics) sample rather than the full 500K
  reviews, for iteration speed — the pipeline generalizes but hasn't been validated
  at full scale.
- Neutral sentiment remained the hardest class across all three approaches
  (best recall: 0.34), consistent with it being an inherently ambiguous, subjective
  category even for human raters.
- A small number of non-food items (e.g., a coconut-oil/shampoo topic) appear in the
  "Fine Food Reviews" dataset — a data quality quirk noted rather than hidden.
- Sentiment-over-time trends before 2005 are not statistically meaningful due to low
  review volume in those years.

## Future Work

- Fine-tune a transformer directly on this domain's labeled data (Score → sentiment)
  rather than relying on off-the-shelf pretrained models
- Run topic modeling on the full 500K reviews with GPU acceleration
- Aspect-based sentiment (e.g., sentiment specifically about "shipping" vs "taste"
  within the same review, not just whole-review sentiment)
