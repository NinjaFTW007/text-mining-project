"""
analytics.py
─────────────────────────────────────────────────────────────────────────────
Global Analytics Dashboard — Enterprise Feature
Description: Connects to the SQLite database to display live, interactive
telemetry and prediction histories across all modules.
"""

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from utils.db import get_all_logs, clear_logs

def run_analytics_module(colors: dict):
    # ── Header Banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background-color: {colors["card_bg"]};
         border-top: 4px solid #14B8A6;
         border-left: 1px solid {colors["border"]}; border-right: 1px solid {colors["border"]}; border-bottom: 1px solid {colors["border"]};
         padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.03);'>
      <div style='color: {colors["text"]}; font-size: 1.8rem; font-weight: 800; margin: 0; display:flex; align-items:center;'>
         <span style='margin-right:12px; font-size: 2.2rem;'>📊</span> Global Analytics Dashboard
      </div>
      <div style='color: {colors["subtext"]}; margin-top: 8px; font-size: 1.05rem;'>
        <span style='font-weight:600; color:{colors["text"]};'>Enterprise Telemetry & System Logging</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Database Connection ────────────────────────────────────────────────
    df = get_all_logs()

    if df.empty:
        st.info("ℹ️ No predictions logged yet. Go to Sentiment or Spam Detection to generate data.")
        return

    # ── Top Metrics ────────────────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total API Requests", f"{len(df):,}")
    
    avg_conf = df["confidence"].mean() * 100
    col2.metric("Avg. Model Confidence", f"{avg_conf:.1f}%")
    
    top_mod = df["module"].mode().iloc[0] if not df["module"].empty else "N/A"
    col3.metric("Most Active Module", top_mod)

    top_algo = df["model_used"].mode().iloc[0] if not df["model_used"].empty else "N/A"
    col4.metric("Dominant Algorithm", top_algo)

    st.markdown("---")

    # ── Charts ─────────────────────────────────────────────────────────────
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Module Usage Distribution")
        fig_pie = px.pie(
            df, names="module", hole=0.4,
            color_discrete_sequence=["#7C3AED", "#10B981", "#F59E0B"]
        )
        fig_pie.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]}, margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)")
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with c2:
        st.markdown("#### Confidence Density Matrix")
        fig_hist = px.histogram(
            df, x="confidence", color="module", marginal="box",
            nbins=15, opacity=0.8,
            color_discrete_sequence=["#7C3AED", "#10B981", "#F59E0B"]
        )
        fig_hist.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]}, margin=dict(t=20, b=20, l=10, r=10),
            legend=dict(bgcolor="rgba(0,0,0,0)"),
            xaxis=dict(gridcolor=colors["border"]),
            yaxis=dict(gridcolor=colors["border"])
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("---")

    # ── Enterprise Data Grid ───────────────────────────────────────────────
    st.markdown("#### 🗃️ Raw Prediction Telemetry")
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Downoad Full System Report (CSV)",
        data=csv,
        file_name='system_telemetry_report.csv',
        mime='text/csv',
    )
    
    # Render dataframe
    st.dataframe(
        df[["timestamp", "module", "model_used", "prediction", "confidence", "input_text"]],
        use_container_width=True,
        height=400
    )

    if st.button("🗑️ Purge Activity Logs"):
        clear_logs()
        st.toast("✅ Logs purged successfully!")
        st.rerun()
