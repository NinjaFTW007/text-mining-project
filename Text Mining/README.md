# 🧠 Multi-Domain Text Mining System

> Intelligent NLP Platform using Sentiment Analysis, Clustering, and Spam Detection

---

## 📋 Project Overview

This project implements a **complete text mining system** across three real-world domains using Python, Streamlit, and scikit-learn. Each module is independently developed, uses its own dataset and model, and is integrated into a single professional Streamlit application with Light/Dark theme support.

---

## 👥 Team Members

| Module | Domain | Developer |
|--------|--------|-----------|
| 🎬 Sentiment Analysis | Entertainment | **Niranjan** |
| 🛍️ Customer Review Clustering | E-Commerce | **Vibhu** |
| 📩 Spam Detection | Communication | **Gowtham** |

---

## 🗂️ Project Structure

```
Text Mining/
│
├── app.py                    ← Main Streamlit application
│
├── modules/
│   ├── __init__.py
│   ├── sentiment.py          ← Module 1: Sentiment Analysis (Niranjan)
│   ├── clustering.py         ← Module 2: Customer Review Clustering (Vibhu)
│   └── spam.py               ← Module 3: Spam Detection (Gowtham)
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py      ← Shared NLP preprocessing utilities
│
├── data/
│   ├── movie_reviews.csv     ← Sentiment dataset (review, sentiment)
│   ├── customer_reviews.csv  ← Clustering dataset (review, category)
│   └── spam.csv              ← Spam dataset (label, message)
│
├── .streamlit/
│   └── config.toml           ← Streamlit theme configuration
│
├── requirements.txt
├── README.md
└── SETUP.md
```

---

## 🚀 Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the application
streamlit run app.py
```

App opens at **http://localhost:8501**

---

## 📦 Module Details

### 🎬 Module 1 — Sentiment Analysis (Niranjan)

- **Dataset:** `data/movie_reviews.csv` (columns: `review`, `sentiment`)
- **Preprocessing:** lowercase → remove URLs/digits/punctuation → remove stopwords
- **Vectorisation:** TF-IDF (unigrams + bigrams, max 5 000 features)
- **Models:** Logistic Regression, Multinomial Naive Bayes
- **Features:**
  - Live prediction with confidence gauge
  - Token highlighting (green=positive, red=negative)
  - Side-by-side model comparison bar chart
  - Confusion matrix heatmap
  - Prediction history log (last 10)

---

### 🛍️ Module 2 — Customer Review Clustering (Vibhu)

- **Dataset:** `data/customer_reviews.csv` (columns: `review`, `category`)
- **Preprocessing:** lowercase → remove punctuation → remove stopwords
- **Vectorisation:** TF-IDF (3 000 features) → TruncatedSVD (50 components) → L2 Normalizer
- **Model:** K-Means (K=2 to K=8)
- **Features:**
  - Elbow method graph (inertia vs K)
  - Silhouette score chart
  - Interactive K slider
  - Cluster details with top keywords
  - Word cloud per cluster
  - t-SNE 2-D visualisation

---

### 📩 Module 3 — Spam Detection (Gowtham)

- **Dataset:** `data/spam.csv` (columns: `label`, `message`)
- **Preprocessing:** lowercase → remove URLs/digits/punctuation → remove stopwords
- **Vectorisation:** TF-IDF (unigrams + bigrams, max 4 000 features)
- **Models:** Multinomial Naive Bayes, Logistic Regression
- **Features:**
  - Live spam/ham classification with confidence gauge
  - Full metrics (accuracy, precision, recall, F1)
  - Confusion matrix heatmap
  - Model comparison chart
  - Batch CSV upload with downloadable results
  - Prediction history log (last 10)

---

## 🎨 UI Features

| Feature | Detail |
|---------|--------|
| Theme Toggle | Dark/Light mode switch in sidebar |
| Dark Mode | Deep navy (#0F172A) background, purple accent |
| Light Mode | Clean white (#F8FAFC) background, purple accent |
| Typography | Google Fonts — Inter |
| Charts | Plotly interactive charts |
| Layout | st.sidebar, st.columns, st.container, st.expander |

---

## 🔧 Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Web UI framework |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models and evaluation |
| `nltk` | NLP - stopwords |
| `plotly` | Interactive charts |
| `matplotlib` + `seaborn` | Confusion matrix plots |
| `wordcloud` | Cluster word clouds |

---

## 📊 Evaluation Metrics Used

- **Accuracy** — Overall correct predictions
- **F1 Score (Macro)** — Harmonic mean of precision / recall
- **Precision** — Positive predictive value
- **Recall** — Sensitivity / true positive rate
- **Confusion Matrix** — TP / TN / FP / FN breakdown
- **Silhouette Score** — Clustering cohesion and separation
- **Inertia** — Within-cluster sum of squared distances

---

## 📚 References

1. Bird, S., Loper, E. & Klein, E. (2009). *Natural Language Processing with Python*. O'Reilly.
2. Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
3. Maas, A. et al. (2011). Learning Word Vectors for Sentiment Analysis. *ACL 2011*.
4. Almeida, T. & Hidalgo, J. (2011). SMS Spam Collection. UCI Machine Learning Repository.
