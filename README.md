# VOIS — Customer Feedback Sentiment Analysis & Topic Modeling

**Repository**: VOIS_Sentiment_Analysis

**Short description**
A Colab-based end-to-end pipeline for cleaning customer feedback, running multi-method sentiment analysis (VADER, TextBlob, Transformers/BERT), extracting topics (LDA, BERTopic, KMeans), producing visualizations (wordclouds, bar/pie charts), and building a small Streamlit dashboard to explore results.

---

## Contents

* `notebooks/` — Jupyter/Colab notebooks and the main analysis workflow (the file you shared).
* `data/` — raw and processed data (e.g., `customer_feedback_cleaned.xlsx`, `customer_feedback_with_sentiment.xlsx`).
* `app.py` — Streamlit dashboard to explore sentiment/topic outputs.
* `requirements.txt` — pinned Python packages for reproducibility.
* `README.md` — this file.

---

## Features

* Multiple sentiment methods: **VADER**, **TextBlob**, and **transformers** (BERT/DistilBERT) for robust classification.
* Topic modeling with **LDA**, **BERTopic**, and **KMeans** (TF-IDF + clustering).
* Aggregation and segmentation: sentiment by `Contract`, `InternetService`, `Region`, `PhoneService`, etc.
* Visualizations: sentiment distribution plots, stacked bar charts, word clouds, and a Streamlit dashboard for interactive exploration.
* Optional: train a simple RandomForest to predict sentiment from customer metadata/features.

---

## Quickstart (Colab)

1. Open the main Colab notebook (or upload `VOIS_Sentiment_analysis_feedback_addition.ipynb`) to Google Colab.
2. Upload your cleaned feedback file using the Colab `files.upload()` widget or mount Google Drive.

   * Expected filename used in the notebook: `customer_feedback_cleaned.xlsx` (adjust code if different).
3. Run cells in order. Key steps included are:

   * Install packages: `vaderSentiment`, `textblob`, `transformers`, `bertopic`, `nltk`, `wordcloud`, `streamlit`, `pyngrok`.
   * VADER/TextBlob sentiment labeling.
   * (Optional) `transformers` pipeline for BERT-based labels — note this can be slow if you run over many rows.
   * Topic modeling with LDA / BERTopic / KMeans.
   * Save outputs: `customer_feedback_with_sentiment.xlsx`, `customer_feedback_with_topics.xlsx`.

---

## Streamlit Dashboard

* The repo contains `app.py` — a simple Streamlit app for uploading the result CSV and exploring sentiment/topic summaries.
* To run locally:

```bash
# create a venv (recommended)
python -m venv venv
source venv/bin/activate  # or `venv\Scripts\activate` on Windows
pip install -r requirements.txt
streamlit run app.py
```

* If you plan to expose the app publicly using `ngrok` or `pyngrok`, **never commit your authtoken** in the repo. Instead export it as an env var or add to a local-only config file.

---

## Example code snippets

**Apply VADER sentiment:**

```py
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()

def get_vader_sentiment(text):
    score = analyzer.polarity_scores(str(text))["compound"]
    if score >= 0.05:
        return "Positive"
    elif score <= -0.05:
        return "Negative"
    else:
        return "Neutral"

df["sentiment_vader"] = df["CleanedFeedback"].apply(get_vader_sentiment)
```

**Save outputs:**

```py
df.to_excel("customer_feedback_with_sentiment.xlsx", index=False)
```

---

## Requirements (suggested)

A minimal `requirements.txt` for reproducibility (adjust versions as needed):

```
pandas
numpy
scikit-learn
nltk
vaderSentiment
textblob
transformers
bertopic
wordcloud
matplotlib
seaborn
streamlit
pyngrok
plotly
openpyxl
```

> Tip: Install heavy packages (transformers, bertopic) only when needed. On Colab you can install at runtime per notebook cell.

---

## Notes & Best Practices

* **Privacy & tokens**: Remove any API tokens (ngrok authtoken) or private keys before committing. The notebook examples show placeholder tokens; do not push real tokens to Git.
* **Transformer models**: The `transformers` pipeline downloads model weights (can be large). For quick experimentation, process a sample of rows or use smaller models like `distilbert-base-uncased-finetuned-sst-2-english`.
* **BERTopic**: Requires additional dependencies (UMAP/HDBSCAN) — if you face install issues, consult BERTopic docs and install `bertopic[visualization]` or the specific extras.
* **Reproducibility**: Save intermediate CSV/XLSX files (`customer_feedback_with_sentiment.xlsx`) so you can iterate on visualization or downstream steps without repeating slow NLP steps.

---

## Example workflow / checklist

1. Prepare and clean feedback → `customer_feedback_cleaned.xlsx`.
2. Run VADER/TextBlob/BERT sentiment labeling → add `sentiment_label` column.
3. Run topic modeling (LDA or BERTopic) → add topic columns.
4. Produce visualizations and summary tables.
5. Save outputs and use `app.py` Streamlit for exploration.

---

## Troubleshooting

* If `nltk` VADER lexicon missing: run `nltk.download('vader_lexicon')`.
* If BERTopic installation fails, ensure you have `pip` updated and install `bertopic` with extras or install `umap-learn` and `hdbscan` first.
* If `transformers` pipeline is slow/out of memory, try processing fewer rows or use a smaller model.

---

## Contributing

Contributions welcome: file issues for bugs or feature requests, or open PRs to add improved visualizations, more robust preprocessing, or better model selection.

---

## 📊 Final Dashboard (Power BI + Streamlit)

### Dashboard Preview

Below is a preview snippet of the Power BI dashboard used in this project:

![Dashboard Screenshot](dashboard_snippet.png)



This project includes a complete **interactive analytics dashboard** built using:

### **1. Power BI Dashboard (`Final dashboard.pbix`)**

The Power BI dashboard provides executive-level insights including:

* ⭐ **Overall Sentiment Scorecards** (Positive / Neutral / Negative)
* 🌍 **Sentiment by Region & State**
* 📡 **Service-wise Sentiment** (Internet, Phone, Streaming)
* 📉 **Churn-Risk Indicators & Negative Feedback Drivers**
* 📊 **Customer Segmentation Filters** (Contract, Tenure, Payment Method)
* 🔍 **Drill-through pages** for detailed customer feedback entries

> The `.pbix` file is included in the repository for further customization.

### **2. Streamlit Dashboard (`app.py`)**

A lightweight web dashboard for interactive feedback exploration.
Features include:

* CSV Upload & Preview
* Sentiment distribution (bar & pie charts)
* **Top Pain Points** extracted using CountVectorizer
* **Positive theme extraction**
* Word Cloud visualization
* Region & Service-based filtering
* Sample feedback viewer (first 10 rows)



