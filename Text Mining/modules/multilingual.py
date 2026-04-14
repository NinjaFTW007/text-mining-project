import streamlit as st
from deep_translator import GoogleTranslator
from modules.sentiment import _train, _predict, _explain_prediction
from utils.db import log_prediction, log_feedback

def run_multilingual_module(colors: dict):
    # ── Header Banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background-color: {colors["card_bg"]};
         border-top: 4px solid #3B82F6;
         border-left: 1px solid {colors["border"]}; border-right: 1px solid {colors["border"]}; border-bottom: 1px solid {colors["border"]};
         padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.03);'>
      <div style='color: {colors["text"]}; font-size: 1.8rem; font-weight: 800; margin: 0; display:flex; align-items:center;'>
         <span style='margin-right:12px; font-size: 2.2rem;'>🌍</span> Global Multilingual Analyzer
      </div>
      <div style='color: {colors["subtext"]}; margin-top: 8px; font-size: 1.05rem;'>
        Domain: <span style='font-weight:600; color:{colors["text"]};'>Cross-Border Intelligence</span> &nbsp;|&nbsp; Developer: <span style='font-weight:600; color:{colors["text"]};'>OmniText</span>
      </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.info("Paste text in **any language** (e.g. Spanish, Hindi, French). The engine will automatically bridge the language barrier and execute text mining analytics on the native English translation mapping.")

    text_input = st.text_area("Source Text (Any Language):", height=150, placeholder="E.g., Esta película fue absolutamente increíble y me encantó...")
    
    col1, col2 = st.columns([1,3])
    with col1:
        model_choice = st.selectbox("ML Engine", ["Logistic Regression", "Naive Bayes"])
    with col2:
        st.markdown("<br>", unsafe_allow_html=True) # padding
        analyze_btn = st.button("🌐 Translate & Analyze", use_container_width=True)
        
    if analyze_btn:
        if len(text_input.split()) < 3:
            st.warning("Please provide at least 3 words to analyze.")
            return
            
        with st.spinner("Bridging language gap and predicting..."):
            # 1. Translate to English
            translator = GoogleTranslator(source='auto', target='en')
            translated_text = translator.translate(text_input)
            
            # Load ML Bundle from Sentiment module
            bundle = _train()
            
            # Predict
            label, conf, pred_int = _predict(translated_text, model_choice, bundle)
            
        st.markdown("---")
        st.markdown("### 🗺️ Translation Engine Output")
        st.markdown(f"> *{translated_text}*")
        
        st.markdown("### 🤖 Model Sentiment Prediction")
        col_color = "#10B981" if pred_int == 1 else "#EF4444"
        st.markdown(
            f"<div style='background:{col_color}22;border-left:4px solid {col_color};"
            f"padding:16px;border-radius:10px;margin:12px 0'>"
            f"<h3 style='color:{col_color};margin:0'>{label}</h3>"
            f"<p style='margin:6px 0 0;color:{colors['subtext']}'>"
            f"Confidence: <b>{conf*100:.1f}%</b> &nbsp;|&nbsp; Model: <b>{model_choice}</b>"
            f"</p></div>",
            unsafe_allow_html=True,
        )
        
        # Explainability
        st.markdown("#### 🧠 Explainable AI: Diagnostic Sub-Token Impact")
        xai_fig = _explain_prediction(translated_text, model_choice, bundle, colors)
        st.plotly_chart(xai_fig, use_container_width=True)
        
        # Log Database
        row_id = log_prediction("Multilingual Analytics", translated_text[:150]+"...", model_choice, label, conf)
        
        st.markdown(f"<div style='margin-top:20px; background:{colors['input_bg']}; border:1px solid {colors['border']}; padding:10px; border-radius:10px; text-align:center;'>", unsafe_allow_html=True)
        st.markdown("**Was this classification correct? (Continuous Learning Loop)**")
        fc1, fc2, fc3 = st.columns([1,1,1])
        with fc2:
            st.button("👍 Correct", key=f"multi_yes_{row_id}", on_click=lambda id=row_id: log_feedback(id, True))
            st.button("👎 Incorrect", key=f"multi_no_{row_id}", on_click=lambda id=row_id: log_feedback(id, False))
        st.markdown("</div>", unsafe_allow_html=True)
