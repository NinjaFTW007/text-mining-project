"""
spam.py
─────────────────────────────────────────────────────────────────────────────
Module 3 – Spam Detection (Communication Domain)
Developer : Gowtham

Description
───────────
Loads SMS data, trains Naive Bayes and Logistic Regression classifiers using
TF-IDF, and provides a Streamlit UI for:
  • Single message classification with confidence
  • Full metrics (accuracy, precision, recall, F1)
  • Confusion matrix heatmap
  • Model comparison bar chart
  • Batch CSV upload & prediction
  • Prediction history
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import datetime
import io
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
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
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
import numpy as np
import pandas as pd

# ── Module Constants ──────────────────────────────────────────────────────────
_DATA_PATH   = os.path.join(_ROOT, "data", "spam.csv")
_MAX_FEAT    = 4_000
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
    Attempt to load data/spam.csv.
    Expected columns: label ('spam'|'ham'), message (str)
    Falls back to synthetic dataset if file is unavailable.
    """
    try:
        # Support common SMS spam format variations
        df = pd.read_csv(_DATA_PATH)
        # Normalise column names
        df.columns = [c.strip().lower() for c in df.columns]
        if "v1" in df.columns and "v2" in df.columns:
            df = df.rename(columns={"v1": "label", "v2": "message"})
        df = df[["label", "message"]].dropna()
        df["label"] = df["label"].str.strip().str.lower()
        df = df[df["label"].isin(["spam", "ham"])]
        if len(df) < 20:
            raise ValueError("Dataset too small.")
        df["target"] = (df["label"] == "spam").astype(int)
        src = f"spam.csv ({len(df)} messages)"
        return df["message"].tolist(), df["target"].tolist(), src
    except Exception:
        return _synthetic_fallback()


def _synthetic_fallback():
    """Generate 200 synthetic SMS messages (100 ham, 100 spam)."""
    random.seed(_RAND)

    ham = [
        "Hey, are you coming to the meeting tomorrow afternoon?",
        "Can you call me back when you get a chance please?",
        "Dinner was great last night! Let's go back soon.",
        "I'll pick you up at seven. Does that still work?",
        "Just checking in. How are you feeling today?",
        "The meeting has moved to Thursday at two pm.",
        "Happy birthday! Hope you have a wonderful day.",
        "Running ten minutes late. Please go ahead without me.",
        "Did you watch the game last night? Amazing finish!",
        "The report is due Friday. Let's meet to finalise.",
        "Thanks for helping me move. Really appreciate it.",
        "I'll be in your area. Want to grab coffee?",
        "Package delivered this morning. All fine.",
        "Can you send me the homework? I missed the email.",
        "Looking forward to the trip. Have you packed yet?",
    ]
    spam_msgs = [
        "WINNER! You have been selected to receive a £900 prize reward!",
        "Congratulations! You won a FREE holiday for two. Reply WIN now.",
        "FREE entry in our weekly competition to win cash! Text ENTER.",
        "URGENT: Your account is compromised. Verify immediately!",
        "You are pre-approved for a loan of up to £5000. Apply now!",
        "Get rich working from home! Earn £500 per day. Call us!",
        "Your mobile number has won a £1000 gift card. Call to claim.",
        "FINAL NOTICE: You owe £800 in taxes. Call immediately!",
        "Lose 20 kg in 30 days with our miracle pill. Order now!",
        "Act fast! Get iPhone 15 for FREE while stocks last!",
        "You have been selected for a £500 survey reward. Click here!",
        "Make money online with no experience. Earn £200 per hour!",
        "Your parcel could not be delivered. Confirm address now!",
        "DATING: Hot singles in your area want to meet tonight!",
        "CONGRATS! You are today's lucky winner. Claim now!",
    ]

    messages, labels = [], []
    for _ in range(100):
        messages.append(random.choice(ham) + " " + random.choice(ham[:7]))
        labels.append(0)
    for _ in range(100):
        messages.append(random.choice(spam_msgs) + " " + random.choice(spam_msgs[:7]))
        labels.append(1)

    pairs = list(zip(messages, labels))
    random.shuffle(pairs)
    msgs, lbs = zip(*pairs)
    return list(msgs), list(lbs), "Synthetic SMS Dataset (200 messages)"


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _train():
    """Fit TF-IDF + NB and TF-IDF + LR on the spam dataset."""
    messages, targets, source = _load_data()

    X_train, X_test, y_train, y_test = train_test_split(
        messages, targets,
        test_size=_TEST_SIZE, random_state=_RAND, stratify=targets,
    )

    Xt_c = [clean_text(m) for m in X_train]
    Xe_c = [clean_text(m) for m in X_test]

    # Shared TF-IDF vectoriser
    vec = TfidfVectorizer(
        ngram_range=_NGRAM,
        max_features=_MAX_FEAT,
        sublinear_tf=True,
    )
    Xtr = vec.fit_transform(Xt_c)
    Xte = vec.transform(Xe_c)

    # Naive Bayes
    nb = MultinomialNB(alpha=0.1)
    nb.fit(Xtr, y_train)
    nb_pred = nb.predict(Xte)

    # Logistic Regression
    lr = LogisticRegression(max_iter=1_000, random_state=_RAND)
    lr.fit(Xtr, y_train)
    lr_pred = lr.predict(Xte)

    def _metrics(pred):
        return {
            "accuracy" : round(accuracy_score(y_test, pred), 4),
            "f1"       : round(f1_score(y_test, pred, average="macro"), 4),
            "precision": round(precision_score(y_test, pred, average="macro", zero_division=0), 4),
            "recall"   : round(recall_score(y_test, pred, average="macro", zero_division=0), 4),
            "cm"       : confusion_matrix(y_test, pred),
        }

    return {
        "vec"       : vec,
        "nb"        : nb,
        "lr"        : lr,
        "nb_metrics": _metrics(nb_pred),
        "lr_metrics": _metrics(lr_pred),
        "source"    : source,
        "n_total"   : len(messages),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _predict(text: str, model_name: str, bundle: dict):
    """Return (label_str, confidence_float, is_spam_int)."""
    cleaned = clean_text(text)
    model   = bundle["nb"] if model_name == "Naive Bayes" else bundle["lr"]
    x       = bundle["vec"].transform([cleaned])
    pred    = int(model.predict(x)[0])
    prob    = model.predict_proba(x)[0]
    conf    = float(prob[pred])
    label   = "🚨 SPAM" if pred == 1 else "✅ HAM (Not Spam)"
    return label, conf, pred


def _batch_predict(texts: list[str], model_name: str, bundle: dict) -> list[dict]:
    """Predict labels for a list of messages."""
    results = []
    for msg in texts:
        label, conf, pred = _predict(msg, model_name, bundle)
        results.append({
            "Message"   : truncate(msg, 80),
            "Prediction": label,
            "Confidence": f"{conf*100:.1f}%",
        })
    return results


def _explain_prediction(text: str, model_name: str, bundle: dict, colors: dict) -> go.Figure:
    """Generate LIME-style explainability chart using coefficient weights."""
    vec = bundle["vec"].transform([text])
    feature_names = bundle["vec"].get_feature_names_out()
    
    non_zero_idx = vec.nonzero()[1]
    words, impacts = [], []
    
    model = bundle["nb"] if "Naive" in model_name else bundle["lr"]
    
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
    df = df.sort_values("AbsImpact", ascending=False).head(5)
    df = df.sort_values("Impact")
    
    fig = go.Figure(go.Bar(
        x=df["Impact"], y=df["Word"], orientation='h',
        marker_color=['#10b981' if val < 0 else '#ef4444' for val in df["Impact"]] # spam is red
    ))
    fig.update_layout(
        title={"text": "🧠 Explainable AI: Triggers", "font": {"color": colors["text"], "size": 14}},
        height=220, margin=dict(l=0, r=20, t=30, b=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, zeroline=True, zerolinecolor=colors["border"]),
        yaxis=dict(showgrid=False, tickfont=dict(color=colors["text"]))
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI  — MODULE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_spam_module(colors: dict) -> None:
    """Render the full Spam Detection module UI."""
    AMBER  = "#F59E0B"
    RED    = "#EF4444"
    GREEN  = "#10B981"
    PURPLE = "#7C3AED"

    # ── Header Banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background-color: {colors["card_bg"]};
         border-top: 4px solid #F59E0B;
         border-left: 1px solid {colors["border"]}; border-right: 1px solid {colors["border"]}; border-bottom: 1px solid {colors["border"]};
         padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.03);'>
      <div style='color: {colors["text"]}; font-size: 1.8rem; font-weight: 800; margin: 0; display:flex; align-items:center;'>
         <span style='margin-right:12px; font-size: 2.2rem;'>🚀</span> Intelligent Spam Detection System
      </div>
      <div style='color: {colors["subtext"]}; margin-top: 8px; font-size: 1.05rem;'>
        Domain: <span style='font-weight:600; color:{colors["text"]};'>Communication</span> &nbsp;|&nbsp; Developer: <span style='font-weight:600; color:{colors["text"]};'>Gowtham</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Load / Train ───────────────────────────────────────────────────────
    with st.spinner("Loading data and training classifiers…"):
        bundle = _train()

    st.success(
        f"✅ Classifiers ready — trained on **{bundle['source']}** "
        f"({bundle['n_total']} messages)."
    )

    # Sidebar info
    with st.sidebar:
        st.markdown("#### ℹ️ Module 3 — Spam Detection")
        with st.expander("Details", expanded=False):
            st.write(f"**Dataset:** {bundle['source']}")
            st.write(f"**Samples:** {bundle['n_total']}")
            st.write("**Vectorizer:** TF-IDF (unigrams + bigrams)")
            st.write("**Models:** Naive Bayes, Logistic Regression")

    # Session state
    if "m3_history" not in st.session_state:
        st.session_state["m3_history"] = []

    if "m3_text_input" not in st.session_state:
        st.session_state.m3_text_input = ""

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1 — CLASSIFY A MESSAGE
    # ─────────────────────────────────────────────────────────────────────
    st.subheader("1. Input Section")
    
    with st.container():
        def set_spam():
            st.session_state.m3_text_input = "Win a FREE iPhone now!!!"
            
        def set_ham():
            st.session_state.m3_text_input = "Hey, are you coming to class?"
            
        msg_text = st.text_area(
            "Enter an SMS message:",
            height=120,
            key="m3_text_input",
            placeholder="e.g. Congratulations! You have won a FREE prize! Call now to claim.",
        )
        
        btn_c1, btn_c2, btn_gap = st.columns([2, 2, 4])
        with btn_c1:
            st.button("⚠️ Try Spam Example", key="m3_ex_spam", use_container_width=True, on_click=set_spam)
        with btn_c2:
            st.button("✅ Try Normal Message", key="m3_ex_ham", use_container_width=True, on_click=set_ham)
                
    st.divider()
    st.subheader("2. Model Selection")
    
    with st.container():
        col_mod, col_btn = st.columns([2, 1])
        with col_mod:
            model_choice = st.selectbox(
                "Choose Model",
                ["Naive Bayes", "Logistic Regression"],
                key="m3_model",
            )
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            classify_btn = st.button("🚀 Classify", key="m3_classify", use_container_width=True)

    if classify_btn:
        if not msg_text.strip() or word_count(msg_text) < 2:
            st.warning("⚠️ Please enter at least 2 words.")
        else:
            with st.spinner("Classifying…"):
                label, conf, is_spam = _predict(msg_text, model_choice, bundle)

            st.divider()
            st.subheader("3. Result Section")
            
            with st.container():
                if conf < 0.60:
                    st.warning("⚠️ Low confidence prediction. Result may not be accurate.", icon="⚠️")
                    
                if is_spam == 1:
                    st.error("🚨 SPAM MESSAGE")
                    clf_status = "SPAM"
                    risk = "🔴 High Risk"
                else:
                    st.success("✅ SAFE MESSAGE")
                    clf_status = "SAFE"
                    risk = "🟢 Low Risk"
                    
                st.info(f"This message is classified as **{clf_status}** with {conf*100:.1f}% confidence.")
                
                m_col1, m_col2, m_col3 = st.columns(3)
                m_col1.metric("Predicted Label", clf_status)
                m_col2.metric("Confidence Score", f"{conf*100:.1f}%")
                m_col3.metric("Risk Indicator", risk)
                
                # Probability Breakdown
                st.markdown("**Probability Breakdown:**")
                p_c1, p_c2 = st.columns(2)
                p_c1.write(f"🛑 Spam: {(conf*100 if is_spam else (1-conf)*100):.1f}%")
                p_c2.write(f"✅ Ham: {((1-conf)*100 if is_spam else conf*100):.1f}%")
                
                # Highlight Important Words
                highlighted_text = msg_text
                import re
                for word in ["free", "win", "offer", "prize", "urgent"]:
                    highlighted_text = re.sub(f"(?i)({word})", r"<span style='background-color:#F59E0B; padding:2px; border-radius:4px; font-weight:bold;'>\1</span>", highlighted_text)
                
                st.markdown(f"<div style='margin:10px 0; padding:10px; border-left:3px solid #EF4444;'>{highlighted_text}</div>", unsafe_allow_html=True)
                
                st.write("**Confidence Line:**")
                st.progress(float(conf))
                
                st.markdown("---")
                
                # Explanation
                st.markdown("#### 🧠 Why this message was classified?")
                st.write("Top contributing token triggers:")
                xai_fig = _explain_prediction(msg_text, model_choice, bundle, colors)
                st.plotly_chart(xai_fig, use_container_width=True)

                # Human in the loop feedback
                st.markdown(f"<div style='background:{colors['input_bg']}; border:1px solid {colors['border']}; padding:10px; border-radius:10px; text-align:center;'>", unsafe_allow_html=True)
                st.markdown("**Was this classification correct? (Continuous Learning Loop)**")
                fc1, fc2, fc3 = st.columns([1,1,1])
                row_id = log_prediction("Spam Detection", truncate(msg_text, 150), model_choice, label, conf)
                
                with fc2:
                    st.button("👍 Correct", key=f"spam_yes_{row_id}", on_click=lambda id=row_id: log_feedback(id, True))
                    st.button("👎 Incorrect", key=f"spam_no_{row_id}", on_click=lambda id=row_id: log_feedback(id, False))
                st.markdown("</div>", unsafe_allow_html=True)

            # Save to history & DB
            st.session_state["m3_history"].insert(0, {
                "Time"      : datetime.datetime.now().strftime("%H:%M:%S"),
                "Message"   : truncate(msg_text, 55),
                "Model"     : model_choice,
                "Label"     : label,
                "Confidence": f"{conf*100:.1f}%",
            })
            st.session_state["m3_history"] = st.session_state["m3_history"][:_HISTORY_MAX]

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2 — MODEL COMPARISON
    # ─────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📊 Model Comparison — NB vs Logistic Regression")

    nb_m = bundle["nb_metrics"]
    lr_m = bundle["lr_metrics"]

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("NB Accuracy",  f"{nb_m['accuracy']*100:.1f}%")
    mc2.metric("NB Precision", f"{nb_m['precision']*100:.1f}%")
    mc3.metric("LR Accuracy",  f"{lr_m['accuracy']*100:.1f}%")
    mc4.metric("LR Precision", f"{lr_m['precision']*100:.1f}%")

    col_bar, col_cm = st.columns(2)

    # Grouped bar chart
    with col_bar:
        metric_names = ["Accuracy", "F1 Score", "Precision", "Recall"]
        nb_vals = [nb_m["accuracy"], nb_m["f1"], nb_m["precision"], nb_m["recall"]]
        lr_vals = [lr_m["accuracy"], lr_m["f1"], lr_m["precision"], lr_m["recall"]]

        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Naive Bayes", x=metric_names, y=nb_vals,
            marker_color=AMBER, text=[f"{v*100:.1f}%" for v in nb_vals],
            textposition="outside",
        ))
        fig_bar.add_trace(go.Bar(
            name="Logistic Regression", x=metric_names, y=lr_vals,
            marker_color=PURPLE, text=[f"{v*100:.1f}%" for v in lr_vals],
            textposition="outside",
        ))
        fig_bar.update_layout(
            barmode="group",
            title="NB vs LR — Performance Metrics",
            yaxis=dict(range=[0, 1.2], gridcolor=colors["border"]),
            xaxis=dict(gridcolor=colors["border"]),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]},
            legend=dict(bgcolor="rgba(0,0,0,0)"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Confusion matrix
    with col_cm:
        cm_sel  = st.selectbox(
            "Confusion Matrix for:", ["Naive Bayes", "Logistic Regression"],
            key="m3_cm_sel",
        )
        cm_data = nb_m["cm"] if cm_sel == "Naive Bayes" else lr_m["cm"]
        is_dark = colors["bg"] < "#888"
        bg_col  = "#1E293B" if is_dark else "#F8FAFC"
        tk_col  = colors["text"]

        fig_cm, ax = plt.subplots(figsize=(4, 3.5))
        fig_cm.patch.set_facecolor(bg_col)
        ax.set_facecolor(bg_col)
        sns.heatmap(
            cm_data, annot=True, fmt="d", cmap="YlOrRd",
            xticklabels=["Ham", "Spam"],
            yticklabels=["Ham", "Spam"],
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
    # SECTION 3 — BATCH UPLOAD
    # ─────────────────────────────────────────────────────────────────────
    st.divider()
    st.subheader("📂 Batch Prediction (Multiple Messages)")
    st.info("Upload a CSV file to classify multiple messages at once.")

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="m3_upload")
    if uploaded:
        try:
            batch_df = pd.read_csv(uploaded)
            if "message" not in batch_df.columns:
                st.error("❌ CSV must contain a column named 'message'.")
            else:
                st.markdown("##### 📄 Data Preview")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                with st.spinner("Classifying batch…"):
                    results = _batch_predict(
                        batch_df["message"].dropna().tolist(),
                        model_choice, bundle,
                    )
                results_df = pd.DataFrame(results)
                
                spam_count = sum(1 for r in results if r["Prediction"].startswith("🚨"))
                safe_count = len(results) - spam_count
                
                st.success(f"✅ Classified {len(results_df)} messages.")
                st.info(f"**Batch Summary:** Total: {len(results_df)} | 🛑 Spam: {spam_count} | ✅ Safe: {safe_count}")
                st.dataframe(results_df, use_container_width=True)

                # Download button
                csv_bytes = results_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Download Results",
                    data=csv_bytes,
                    file_name="spam_results.csv",
                    mime="text/csv",
                )
        except Exception as e:
            st.error(f"Error reading file: {e}")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 4 — PREDICTION HISTORY
    # ─────────────────────────────────────────────────────────────────────
    if st.session_state["m3_history"]:
        st.divider()
        st.subheader("🕑 Prediction History (last 5)")
        st.dataframe(
            pd.DataFrame(st.session_state["m3_history"]),
            use_container_width=True,
        )


