"""
sentiment.py
─────────────────────────────────────────────────────────────────────────────
Module 1 – Sentiment Analysis (Entertainment Domain)
Developer : Niranjan

Description
───────────
Loads movie review data, trains two classifiers (Logistic Regression and
Naive Bayes) using TF-IDF bigrams, and exposes a Streamlit UI for:
  • Live review prediction with confidence gauge
  • Side-by-side model comparison (accuracy, F1, precision, recall)
  • Confusion matrix heatmap
  • Prediction history log
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import datetime
import random

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score,
    recall_score, confusion_matrix,
)

# ── Local Utilities ───────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from utils.preprocessing import clean_text, word_count, truncate
from utils.db import log_prediction, log_feedback
from utils.scraper import fetch_hackernews_titles
from deep_translator import GoogleTranslator
import numpy as np

# ── Module Constants ──────────────────────────────────────────────────────────
_DATA_PATH   = os.path.join(_ROOT, "data", "movie_reviews.csv")
_MAX_FEAT    = 5_000
_NGRAM       = (1, 2)
_TEST_SIZE   = 0.20
_RAND        = 42
_HISTORY_MAX = 5


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data():
    """
    Attempt to load data/movie_reviews.csv.
    Expected columns: review (str), sentiment ('positive'|'negative')
    Falls back to a built-in synthetic corpus if file is absent.
    """
    try:
        df = pd.read_csv(_DATA_PATH)
        df = df[["review", "sentiment"]].dropna()
        df["sentiment"] = df["sentiment"].str.strip().str.lower()
        df = df[df["sentiment"].isin(["positive", "negative"])]
        if len(df) < 20:
            raise ValueError("Dataset too small — using synthetic data.")
        df["label"] = (df["sentiment"] == "positive").astype(int)
        return df["review"].tolist(), df["label"].tolist(), f"movie_reviews.csv ({len(df)} rows)"
    except Exception:
        return _synthetic_fallback()


def _synthetic_fallback():
    """Generate 200 synthetic reviews when the CSV is unavailable."""
    random.seed(_RAND)
    pos = [
        "absolutely brilliant masterpiece stunning visuals outstanding performance",
        "loved every moment perfect storytelling moving emotional journey",
        "incredible performances breathtaking cinematography amazing film",
        "hilarious entertaining thoroughly enjoyable great comedy brilliant",
        "superb direction talented cast compelling narrative deeply engaging",
        "beautiful touching heartfelt story wonderful acting superb screenplay",
        "excellent script clever dialogue engaging plot fantastic acting",
        "fantastic movie unforgettable experience highly recommend masterpiece",
        "thrilling suspenseful gripping exceptional direction cinematography",
        "wonderful uplifting family friendly brilliant feel-good film",
    ]
    neg = [
        "terrible boring waste time awful acting dreadful screenplay",
        "horrible plot makes no sense confusing storyline bad acting awful",
        "completely unwatchable slow paced dull disappointing utter failure",
        "worst movie ever seen poor script bad direction terrible acting",
        "painfully bad acting embarrassing weak storyline rubbish mediocre",
        "dreadful screenplay flat characters pointless narrative waste money",
        "boring tedious completely forgettable mediocre disappointing",
        "atrocious film no redeeming qualities whatsoever terrible experience",
        "poor direction confusing plot terrible acting deeply disappointing",
        "cheap production wooden acting terrible script complete waste time",
    ]
    reviews, labels = [], []
    for _ in range(100):
        reviews.append(random.choice(pos) + " " + random.choice(pos[:5]))
        labels.append(1)
    for _ in range(100):
        reviews.append(random.choice(neg) + " " + random.choice(neg[:5]))
        labels.append(0)
    pairs = list(zip(reviews, labels))
    random.shuffle(pairs)
    rv, lb = zip(*pairs)
    return list(rv), list(lb), "Synthetic Movie Reviews (200 samples)"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _train():
    """
    Fit TF-IDF + LR and TF-IDF + NB pipelines on training split.
    Returns a bundle dict with everything needed for inference and display.
    """
    reviews, labels, source = _load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        reviews, labels,
        test_size=_TEST_SIZE, random_state=_RAND, stratify=labels,
    )

    # Preprocess text
    Xt_c = [clean_text(r) for r in X_train]
    Xe_c = [clean_text(r) for r in X_test]

    # TF-IDF vectorisation (shared vectoriser)
    vec = TfidfVectorizer(
        ngram_range=_NGRAM,
        max_features=_MAX_FEAT,
        sublinear_tf=True,
    )
    Xtr = vec.fit_transform(Xt_c)
    Xte = vec.transform(Xe_c)
    vocab = vec.get_feature_names_out()

    # Logistic Regression
    lr = LogisticRegression(max_iter=1_000, random_state=_RAND, C=1.0)
    lr.fit(Xtr, y_train)
    lr_pred = lr.predict(Xte)

    # Multinomial Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(Xtr, y_train)
    nb_pred = nb.predict(Xte)

    def _metrics(pred):
        return {
            "accuracy" : round(accuracy_score(y_test, pred), 4),
            "f1"       : round(f1_score(y_test, pred, average="macro"), 4),
            "precision": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "recall"   : round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "cm"       : confusion_matrix(y_test, pred),
        }

    # Top positive / negative feature words (for explainability)
    coef     = lr.coef_[0]
    pos_words = set(vocab[i] for i in np.argsort(coef)[-30:][::-1])
    neg_words = set(vocab[i] for i in np.argsort(coef)[:30])

    return {
        "vec"       : vec,
        "lr"        : lr,
        "nb"        : nb,
        "vocab"     : vocab,
        "coef"      : coef,
        "lr_metrics": _metrics(lr_pred),
        "nb_metrics": _metrics(nb_pred),
        "pos_words" : pos_words,
        "neg_words" : neg_words,
        "source"    : source,
        "n_total"   : len(reviews),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _predict(text: str, model_name: str, bundle: dict):
    """Return (label_str, confidence_float, pred_int)."""
    import re
    # ── Text Normalization (Fix 1) ──
    norm_text = text.lower()
    norm_text = re.sub(r'(.)\1{2,}', r'\1\1', norm_text) # e.g. goooood -> good
    norm_text = re.sub(r'\bgbu\b', 'good', norm_text)
    norm_text = re.sub(r'\bmacha\b', '', norm_text)
    
    cleaned = clean_text(norm_text)
    model   = bundle["lr"] if model_name == "Logistic Regression" else bundle["nb"]
    x       = bundle["vec"].transform([cleaned])
    pred    = int(model.predict(x)[0])
    prob    = model.predict_proba(x)[0]
    conf    = float(prob[pred])
    
    if conf < 0.60:
        label = "Uncertain 🤔"
        pred = -1 # Special case
    else:
        label = "Positive 😊" if pred == 1 else "Negative 😞"
        
    return label, conf, pred


def _token_highlight(text: str, pos_words: set, neg_words: set) -> str:
    """Wrap tokens in coloured HTML spans for explainability."""
    import string as _str
    parts = []
    for tok in text.split():
        clean = tok.lower().strip(_str.punctuation)
        if clean in pos_words:
            parts.append(
                f'<span style="background:#15803d;color:#d1fae5;'
                f'padding:2px 6px;border-radius:5px;margin:2px">{tok}</span>'
            )
        elif clean in neg_words:
            parts.append(
                f'<span style="background:#991b1b;color:#fee2e2;'
                f'padding:2px 6px;border-radius:5px;margin:2px">{tok}</span>'
            )
        else:
            parts.append(tok)
    return " ".join(parts)


    return " ".join(parts)


def _explain_prediction(text: str, model_name: str, bundle: dict, colors: dict) -> go.Figure:
    """Generate LIME-style explainability chart using coefficient weights."""
    vec = bundle["vec"].transform([text])
    feature_names = bundle["vec"].get_feature_names_out()
    
    non_zero_idx = vec.nonzero()[1]
    words, impacts = [], []
    
    model = bundle["lr"] if "Logistic" in model_name else bundle["nb"]
    
    for idx in non_zero_idx:
        word = feature_names[idx]
        if "Logistic" in model_name:
            weight = model.coef_[0][idx]
        else:
            weight = model.feature_log_prob_[1][idx] - model.feature_log_prob_[0][idx]
        
        words.append(word)
        impacts.append(weight * vec[0, idx])
        
    df = pd.DataFrame({"Word": words, "Impact": impacts})
    if df.empty:
        return go.Figure()
        
    df["AbsImpact"] = df["Impact"].abs()
    df = df.sort_values("AbsImpact", ascending=False).head(6)
    df = df.sort_values("Impact")
    
    fig = go.Figure(go.Bar(
        x=df["Impact"], y=df["Word"], orientation='h',
        marker_color=['#ef4444' if val < 0 else '#10b981' for val in df["Impact"]]
    ))
    fig.update_layout(
        title={"text": "🧠 Explainable AI: Top Impacting Words", "font": {"color": colors["text"], "size": 14}},
        height=220, margin=dict(l=0, r=20, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=colors["border"]),
        yaxis=dict(showgrid=False, tickfont=dict(color=colors["text"]))
    )
    return fig

# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI  — MODULE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_sentiment_module(colors: dict) -> None:
    """
    Render the full Sentiment Analysis module UI.

    Parameters
    ----------
    colors : dict
        Theme color dict from app.py (keys: bg, card_bg, text, subtext,
        accent, border).
    """
    PURPLE = "#7C3AED"
    GREEN  = "#10B981"
    RED    = "#EF4444"

    # ── Header Banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background-color: {colors["card_bg"]};
         border-top: 4px solid #8B5CF6;
         border-left: 1px solid {colors["border"]}; border-right: 1px solid {colors["border"]}; border-bottom: 1px solid {colors["border"]};
         padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.03);'>
      <div style='color: {colors["text"]}; font-size: 1.8rem; font-weight: 800; margin: 0; display:flex; align-items:center;'>
         <span style='margin-right:12px; font-size: 2.2rem;'>🚀</span> Intelligent Sentiment Analysis System
      </div>
      <div style='color: {colors["subtext"]}; margin-top: 8px; font-size: 1.05rem;'>
        Domain: <span style='font-weight:600; color:{colors["text"]};'>Entertainment</span> &nbsp;|&nbsp; Developer: <span style='font-weight:600; color:{colors["text"]};'>Niranjan</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load / Train ───────────────────────────────────────────────────────
    with st.spinner("Loading data and training models…"):
        bundle = _train()

    st.success(
        f"✅ Models ready — trained on **{bundle['source']}** "
        f"({bundle['n_total']} reviews)."
    )

    # ── Sidebar info ───────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("#### ℹ️ Module 1 — Sentiment")
        with st.expander("Details", expanded=False):
            st.write(f"**Dataset:** {bundle['source']}")
            st.write(f"**Samples:** {bundle['n_total']}")
            st.write("**Vectorizer:** TF-IDF (unigrams + bigrams)")
            st.write("**Models:** Logistic Regression, Naive Bayes")

    # Session state for history
    if "m1_history" not in st.session_state:
        st.session_state["m1_history"] = []

    if "m1_text_input" not in st.session_state:
        st.session_state.m1_text_input = ""

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1 — PREDICT
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 🔍 Predict Sentiment")
    
    # ── Input Section ──
    st.markdown("#### 1. Input Section")
    
    def set_pos():
        st.session_state.m1_text_input = "This movie was absolutely amazing! The acting was brilliant and the soundtrack was stunning. I loved every second of it."
        
    def set_neg():
        st.session_state.m1_text_input = "Terrible experience. The plot was completely incoherent, the dialogue felt incredibly forced, and it was a complete waste of time."

    btn_c1, btn_c2, btn_gap = st.columns([1, 1, 3])
    with btn_c1:
        st.button("Positive Example", key="m1_ex_pos", on_click=set_pos)
    with btn_c2:
        st.button("Negative Example", key="m1_ex_neg", on_click=set_neg)
        
    review_text = st.text_area(
        "Enter a text to analyze:",
        height=130,
        key="m1_text_input",
        placeholder="e.g. The film was absolutely stunning with incredible performances…",
    )
    
    enable_multi = st.checkbox("🌍 Enable Multilingual Support (Analyze text in any language)")
    if enable_multi:
        st.info("Input will be translated to English before analysis.")

    # ── Model Selection ──
    st.markdown("#### 2. Model Selection")
    col_mod, col_btn = st.columns([2, 1])
    with col_mod:
        model_choice = st.selectbox(
            "Choose ML Engine:", ["Logistic Regression", "Naive Bayes"], key="m1_model"
        )
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        predict_btn = st.button("🚀 Analyze Text", key="m1_predict", use_container_width=True)

    do_predict = False
    do_translate = False
    final_text = ""
    final_model = ""

    if predict_btn:
        if not review_text.strip() or word_count(review_text) < 3:
            st.warning("⚠️ Please enter at least 3 words to get a prediction.")
        else:
            final_text = review_text
            final_model = model_choice
            do_translate = enable_multi
            do_predict = True

    if do_predict:
            import time
            start_time = time.time()
            if do_translate:
                with st.spinner("Translating text..."):
                    translator = GoogleTranslator(source='auto', target='en')
                    final_text = translator.translate(final_text)
                    st.info(f"**Translated Context:** {final_text}")
                    
            with st.spinner("Analysing sentiment…"):
                label, conf, pred_int = _predict(final_text, final_model, bundle)
                
            end_time = time.time()
            proc_time = end_time - start_time
                
            # ── Result Section ──
            st.markdown("#### 3. Result")
            
            if conf < 0.60:
                st.warning("⚠️ Low confidence prediction. Result may not be accurate.")
                st.warning("### 🤔 Uncertain Prediction")
                st.info("This text expresses a 🤔 Uncertain sentiment.")
            elif pred_int == 1:
                st.success("😊 Positive Sentiment")
                st.info("This text expresses a Positive sentiment.")
            else:
                st.error("😞 Negative Sentiment")
                st.info("This text expresses a Negative sentiment.")
            
            # Metric and Progress Bar
            r_col1, r_col2, r_col3 = st.columns(3)
            r_col1.metric("Predicted Sentiment", label)
            r_col2.metric("Confidence Score", f"{conf*100:.1f}%")
            r_col3.metric("Processing Time", f"{proc_time:.2f} s")
            
            st.write("**Confidence Line:**")
            st.progress(float(conf))
            
            st.markdown("---")
            
            # ── Explanation Section ──
            st.markdown("#### 4. 🧠 Why this prediction?")
            st.write("Top contributing words pushing the model towards this sentiment:")
            xai_fig = _explain_prediction(final_text, final_model, bundle, colors)
            st.plotly_chart(xai_fig, use_container_width=True)

            # Human in the loop feedback
            with st.container():
                st.markdown(f"<div style='background:{colors['input_bg']}; border:1px solid {colors['border']}; padding:10px; border-radius:10px; text-align:center;'>", unsafe_allow_html=True)
                st.markdown("**Was this prediction correct? (Continuous Learning Loop)**")
                fc1, fc2, fc3 = st.columns([1,1,1])
                # Save to history & SQLite DB
                row_id = log_prediction("Sentiment Analysis", truncate(final_text, 150), final_model, label, conf)
                
                with fc2:
                    st.button("👍 Correct", key=f"fb_yes_{row_id}", on_click=lambda id=row_id: log_feedback(id, True))
                    st.button("👎 Incorrect", key=f"fb_no_{row_id}", on_click=lambda id=row_id: log_feedback(id, False))
                st.markdown("</div>", unsafe_allow_html=True)
            
            st.session_state["m1_history"].insert(0, {
                "Time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                "Review"    : truncate(final_text, 55),
                "Model"     : final_model,
                "Prediction": label,
                "Confidence": f"{conf*100:.1f}%",
            })
            st.session_state["m1_history"] = st.session_state["m1_history"][:_HISTORY_MAX]

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2 — MODEL COMPARISON
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Comparison — LR vs Naive Bayes")

    lr_m = bundle["lr_metrics"]
    nb_m = bundle["nb_metrics"]

    # metrics cards
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("LR Accuracy",  f"{lr_m['accuracy']*100:.1f}%")
    mc2.metric("LR F1",        f"{lr_m['f1']*100:.1f}%")
    mc3.metric("NB Accuracy",  f"{nb_m['accuracy']*100:.1f}%")
    mc4.metric("NB F1",        f"{nb_m['f1']*100:.1f}%")

    col_bar, col_cm = st.columns(2)

    # Grouped bar chart
    with col_bar:
        metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
        lr_vals = [lr_m["accuracy"], lr_m["f1"], lr_m["precision"], lr_m["recall"]]
        nb_vals = [nb_m["accuracy"], nb_m["f1"], nb_m["precision"], nb_m["recall"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Logistic Regression", x=metric_names, y=lr_vals,
            marker_color=PURPLE, text=[f"{v*100:.1f}%" for v in lr_vals],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name="Naive Bayes", x=metric_names, y=nb_vals,
            marker_color=GREEN, text=[f"{v*100:.1f}%" for v in nb_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            barmode="group",
            title="LR vs NB — Performance Metrics",
            yaxis=dict(range=[0, 1.15], gridcolor=colors["border"]),
            xaxis=dict(gridcolor=colors["border"]),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]},
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion matrix
    with col_cm:
        cm_sel = st.selectbox(
            "Confusion Matrix for:", ["Logistic Regression", "Naive Bayes"],
            key="m1_cm_sel",
        )
        cm_data = lr_m["cm"] if cm_sel == "Logistic Regression" else nb_m["cm"]
        is_dark = colors["bg"] < "#888"   # rough dark-mode check
        bg_col  = "#1E293B" if is_dark else "#F8FAFC"
        tk_col  = colors["text"]

        fig_cm, ax = plt.subplots(figsize=(4, 3.5))
        fig_cm.patch.set_facecolor(bg_col)
        ax.set_facecolor(bg_col)
        sns.heatmap(
            cm_data, annot=True, fmt="d", cmap="Purples",
            xticklabels=["Negative", "Positive"],
            yticklabels=["Negative", "Positive"],
            ax=ax, annot_kws={"size": 14},
        )
        ax.tick_params(colors=tk_col)
        ax.set_xlabel("Predicted", color=tk_col)
        ax.set_ylabel("Actual", color=tk_col)
        ax.set_title("Confusion Matrix", color=tk_col, fontsize=13)
        plt.tight_layout()
        st.pyplot(fig_cm)
        plt.close(fig_cm)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3 — PREDICTION HISTORY
    # ─────────────────────────────────────────────────────────────────────
    if st.session_state["m1_history"]:
        st.markdown("---")
        st.markdown("### 🕑 Prediction History (last 5)")
        st.dataframe(
            pd.DataFrame(st.session_state["m1_history"]),
            use_container_width=True,
        )


