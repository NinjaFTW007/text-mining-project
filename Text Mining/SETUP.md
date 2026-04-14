# 🧠 Setup Guide — Multi-Domain Text Mining System

### 5 Simple Steps to Run the Project

---

## Prerequisites

- **Python 3.9 or higher** (Python 3.10+ recommended)
- **pip** (bundled with Python)
- Internet connection for first-run NLTK downloads

---

## Step 1 — Ensure Project Files Are in Place

Your project folder should look like this:

```
Text Mining/
├── app.py
├── requirements.txt
├── README.md
├── SETUP.md
│
├── modules/
│   ├── __init__.py
│   ├── sentiment.py
│   ├── clustering.py
│   └── spam.py
│
├── utils/
│   ├── __init__.py
│   └── preprocessing.py
│
├── data/
│   ├── movie_reviews.csv
│   ├── customer_reviews.csv
│   └── spam.csv
│
└── .streamlit/
    └── config.toml
```

---

## Step 2 — (Optional) Add Real Datasets

To use real-world datasets instead of the built-in synthetic data,
place these files in the `data/` folder:

| File | Required Columns | Source |
|------|-----------------|--------|
| `movie_reviews.csv` | `review`, `sentiment` (positive/negative) | Kaggle — IMDB Dataset of 50K Movie Reviews |
| `customer_reviews.csv` | `review`, `category` | Any e-commerce review dataset |
| `spam.csv` | `label` (spam/ham), `message` | [UCI SMS Spam Collection](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection) |

> ✅ **If these files are absent**, the app automatically uses built-in
> synthetic data and will still run completely without any errors.

---

## Step 3 — Create & Activate a Virtual Environment

### Windows (PowerShell — recommended)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Windows (Command Prompt)
```cmd
python -m venv .venv
.venv\Scripts\activate.bat
```

### macOS / Linux
```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## Step 4 — Install Dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **First run** also auto-downloads NLTK corpora (`stopwords`, `punkt`,
> `wordnet`) — a one-time internet connection is required.

---

## Step 5 — Run the Application

```bash
cd "g:\Text Mining"        # Windows
# or
cd "/path/to/Text Mining"  # macOS/Linux

streamlit run app.py
```

The app will open automatically at **http://localhost:8501** 🚀

---

## 🧭 Using the Application

### Navigation
- Use the **sidebar menu** to switch between modules
- Toggle **Dark/Light mode** with the theme switch at the top of the sidebar

### Module 1 — Sentiment Analysis (Niranjan)
1. Navigate to **🎬 Sentiment Analysis**
2. Type or paste a movie review
3. Select a model (Logistic Regression or Naive Bayes)
4. Click **🚀 Predict** to see the sentiment and confidence
5. Scroll down to view model comparison charts and confusion matrix

### Module 2 — Review Clustering (Vibhu)
1. Navigate to **🛍️ Review Clustering**
2. Study the Elbow and Silhouette charts to pick the best K
3. Drag the **K slider** to set number of clusters
4. Expand each cluster card to view keywords, sample reviews, and word clouds
5. Scroll down for the t-SNE 2-D scatter visualisation

### Module 3 — Spam Detection (Gowtham)
1. Navigate to **📩 Spam Detection**
2. Type an SMS message and click **🚀 Classify**
3. View the classification and confidence gauge
4. Upload a CSV file (with a `message` column) for batch prediction
5. Download results as CSV

---

## 🛠️ Troubleshooting

| Issue | Solution |
|-------|---------|
| `ModuleNotFoundError` | Re-run `pip install -r requirements.txt` |
| `NLTK LookupError` | Run `python -c "import nltk; nltk.download('all')"` |
| Port 8501 already in use | Run `streamlit run app.py --server.port 8502` |
| `wordcloud` install fails on Windows | Run `pip install --pre wordcloud` |
| t-SNE is slow | Normal for large datasets — reduce data size if needed |
| Execution policy error (PowerShell) | Run `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` |

---

## 📦 Dependencies Summary

```
streamlit>=1.32.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
nltk>=3.8.1
plotly>=5.18.0
matplotlib>=3.7.0
seaborn>=0.12.0
wordcloud>=1.9.0
```

---

## 👥 Team

| Module | Developer | Contribution |
|--------|-----------|-------------|
| M1 — Sentiment Analysis | **Niranjan** | Data loading, TF-IDF, LR + NB, UI, token highlighting |
| M2 — Customer Review Clustering | **Vibhu** | Data loading, TF-IDF+SVD, K-Means, word clouds, t-SNE |
| M3 — Spam Detection | **Gowtham** | Data loading, TF-IDF, NB + LR, batch upload, UI |
| Integration & Dashboard | **All Three** | app.py, theme system, home page, preprocessing |
