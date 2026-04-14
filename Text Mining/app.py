"""
app.py
─────────────────────────────────────────────────────────────────────────────
Multi-Domain Text Mining System
─────────────────────────────────────────────────────────────────────────────
Intelligent Text Mining with Explainable Insights

  Module 1 – Sentiment Analysis        (Entertainment)   Developer: Niranjan
  Module 2 – Customer Review Clustering (E-Commerce)     Developer: Vibhu
  Module 3 – Spam Detection             (Communication)  Developer: Gowtham

Run:  streamlit run app.py
─────────────────────────────────────────────────────────────────────────────
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys

# ── Third-party ───────────────────────────────────────────────────────────────
import streamlit as st
import plotly.graph_objects as go
import pandas as pd

# ── Project Path Setup ────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _ROOT)

# ── Module Imports ────────────────────────────────────────────────────────────
from modules.sentiment  import run_sentiment_module
from modules.clustering import run_clustering_module
from modules.spam       import run_spam_module
from utils.db           import init_db


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE CONFIGURATION  (must be first Streamlit call)
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title        = "Text Mining Project",
    page_icon         = "🔍",
    layout            = "wide",
    initial_sidebar_state = "expanded",
)

# Initialise persistent database
init_db()


# ══════════════════════════════════════════════════════════════════════════════
#  THEME ENGINE
# ══════════════════════════════════════════════════════════════════════════════

# Dark-mode palette
_DARK = {
    "bg"      : "#0F172A",
    "card_bg" : "#1E293B",
    "text"    : "#E2E8F0",
    "subtext" : "#94A3B8",
    "accent"  : "#7C3AED",
    "border"  : "#334155",
    "sidebar" : "#1E293B",
    "input_bg": "#0F172A",
    "plot_bg" : "rgba(0,0,0,0)",
}

# Light-mode palette
_LIGHT = {
    "bg"      : "#F2F5F8",
    "card_bg" : "#FAFAFB",
    "text"    : "#2C3E50",
    "subtext" : "#7F8C8D",
    "accent"  : "#10B981", 
    "border"  : "#E0E6ED",
    "sidebar" : "#F7F9FA",
    "input_bg": "#FFFFFF",
    "plot_bg" : "rgba(0,0,0,0)",
}


def _inject_css(c: dict) -> None:
    """Inject dynamic CSS for theme colors."""
    st.markdown(f"""
    <style>
    /* ── Google Font ─────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* ── Root ───────────────────────────────────────────── */
    html, body, [class*="css"] {{
        font-family: 'Inter', sans-serif;
    }}

    /* ── Hide Streamlit Branding ────────────────────────── */
    .stAppDeployButton {{ display: none !important; }}
    footer[data-testid="stFooter"] {{ display: none !important; }}
    #MainMenu {{ display: none !important; }}

    /* ── App background ─────────────────────────────────── */
    .stApp {{
        background-color: {c['bg']} !important;
    }}

    /* ── Sidebar ────────────────────────────────────────── */
    section[data-testid="stSidebar"] {{
        background-color: {c['sidebar']} !important;
        border-right: 1px solid {c['border']} !important;
    }}
    section[data-testid="stSidebar"] * {{
        color: {c['text']} !important;
    }}

    /* ── Main content text ──────────────────────────────── */
    h1, h2, h3, h4, h5, h6 {{
        color: {c['text']} !important;
    }}
    p, li, label, .stMarkdown, span {{
        color: {c['text']} !important;
    }}

    /* ── Text inputs & textarea ─────────────────────────── */
    .stTextArea textarea,
    .stTextInput input {{
        background-color: {c['input_bg']} !important;
        color: {c['text']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        font-size: 0.95rem !important;
    }}
    .stTextArea textarea:focus,
    .stTextInput input:focus {{
        border-color: {c['accent']} !important;
        box-shadow: 0 0 0 2px {c['accent']}44 !important;
    }}

    /* ── Selectbox ──────────────────────────────────────── */
    .stSelectbox > div > div {{
        background-color: {c['input_bg']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        color: {c['text']} !important;
    }}

    /* ── Buttons ────────────────────────────────────────── */
    div[data-testid="stButton"] > button,
    div[data-testid="stDownloadButton"] > button {{
        background: linear-gradient(135deg, {c['accent']}, #2DD4BF) !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        transition: all 0.25s ease !important;
        box-shadow: 0 4px 10px {c['accent']}44 !important;
    }}
    div[data-testid="stButton"] > button:hover,
    div[data-testid="stDownloadButton"] > button:hover {{
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 15px {c['accent']}66 !important;
    }}

    /* ── Metric cards ───────────────────────────────────── */
    [data-testid="metric-container"] {{
        background-color: {c['card_bg']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 12px !important;
        padding: 16px 20px !important;
        transition: box-shadow 0.2s;
    }}
    [data-testid="metric-container"]:hover {{
        box-shadow: 0 4px 16px {c['accent']}33 !important;
    }}
    [data-testid="stMetricValue"] {{
        color: {c['accent']} !important;
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }}
    [data-testid="stMetricLabel"] {{
        color: {c['subtext']} !important;
    }}

    /* ── Expander ───────────────────────────────────────── */
    .streamlit-expanderHeader {{
        background-color: {c['card_bg']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 10px !important;
        color: {c['text']} !important;
    }}
    .streamlit-expanderContent {{
        background-color: {c['card_bg']} !important;
        border: 1px solid {c['border']} !important;
        border-radius: 0 0 10px 10px !important;
    }}

    /* ── Slider ─────────────────────────────────────────── */
    .stSlider [data-baseweb="slider"] {{
        color: {c['accent']} !important;
    }}

    /* ── DataTable ──────────────────────────────────────── */
    /* Removed the global * selector to prevent pure white blob bug */
    div[data-testid="stDataFrame"] {{
        border-radius: 10px;
        overflow: hidden;
        border: 1px solid {c['border']};
    }}

    /* ── Success / Info / Warning banners ───────────────── */
    .stSuccess, .stInfo, .stWarning, .stError {{
        border-radius: 10px !important;
    }}

    /* ── Divider ────────────────────────────────────────── */
    hr {{
        border-color: {c['border']} !important;
        margin: 20px 0 !important;
    }}

    /* ── File uploader ──────────────────────────────────── */
    [data-testid="stFileUploader"] {{
        background-color: {c['card_bg']} !important;
        border: 1px dashed {c['border']} !important;
        border-radius: 10px !important;
        padding: 5px !important;
    }}
    [data-testid="stFileUploadDropzone"] {{
        background-color: {c['card_bg']} !important;
        color: {c['text']} !important;
    }}
    [data-testid="stFileUploadDropzone"] div,
    [data-testid="stFileUploadDropzone"] span,
    [data-testid="stFileUploadDropzone"] small {{
        color: {c['text']} !important;
    }}

    /* ── Sleek Navigation Menu ──────────────────────────── */
    .stRadio div[role="radiogroup"] > label {{
        background: transparent;
        border-radius: 8px;
        padding: 5px 12px;
        margin-bottom: 2px;
        transition: all 0.2s ease;
        cursor: pointer;
    }}
    .stRadio div[role="radiogroup"] > label:hover {{
        background-color: {c['border']}66;
    }}
    /* Hide the radio circles */
    .stRadio div[role="radiogroup"] > label > div:first-child {{
        display: none !important;
    }}
    .stRadio div[role="radiogroup"] > label p {{
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }}

    /* ── Custom card helper class ───────────────────────── */
    .tm-card {{
        background-color: {c['card_bg']};
        border: 1px solid {c['border']};
        border-radius: 14px;
        padding: 22px;
        margin: 8px 0;
        transition: box-shadow 0.2s, transform 0.2s;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }}
    .tm-card:hover {{
        box-shadow: 0 10px 25px rgba(0,0,0,0.1);
        transform: translateY(-2px);
    }}
    /* ── Adjust stForm to look like a popup ─────────────── */
    div[data-testid="stForm"] {{
        background-color: {c["card_bg"]} !important;
        border: 1px solid {c["border"]} !important;
        border-radius: 16px !important;
        padding: 2.5rem !important;
        box-shadow: 0 15px 35px rgba(0,0,0,0.08) !important;
    }}
    </style>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ══════════════════════════════════════════════════════════════════════════════

if "dark_mode" not in st.session_state:
    st.session_state["dark_mode"] = True   # default: dark



# ── Theme Toggle (Top Right) ──────────────────────────────────────────────────
_pad, col_theme = st.columns([9, 1.5])
with col_theme:
    dark_mode = st.toggle("🌙 Dark Mode", value=st.session_state["dark_mode"], key="theme_toggle_main")
    st.session_state["dark_mode"] = dark_mode

# Apply theme colors
colors = _DARK if dark_mode else _LIGHT
_inject_css(colors)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

with st.sidebar:

    # ── Brand ──────────────────────────────────────────────────────────────
    st.markdown(
        f"""
        <div style="display:flex;align-items:center;margin-bottom:20px;margin-top:10px">
            <span style="font-size:1.8rem;margin-right:10px;filter:drop-shadow(0 2px 4px rgba(0,0,0,0.2))">🔍</span>
            <h2 style='margin:0;font-size:1.4rem;font-weight:800;letter-spacing:-0.5px;color:{colors['text']}'>Text Mining System</h2>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    # ── Navigation ─────────────────────────────────────────────────────────
    st.markdown("### 📚 Navigation")
    module = st.radio(
        "Go to:",
        options=[
            "🏠  Home",
            "🎬  Sentiment Analysis",
            "🛍️  Review Clustering",
            "📩  Spam Detection",
        ],
        key="nav_module",
        label_visibility="collapsed",
    )
    st.markdown("---")



    # ── Team Members ───────────────────────────────────────────────────────
    st.markdown("### 👥 Team Members")
    st.markdown(f"""
    <div style='background:{colors["card_bg"]};border:1px solid {colors["border"]};border-radius:12px;padding:14px;margin-bottom:12px;box-shadow:0 4px 6px rgba(0,0,0,0.02)'>
        <div style='display:flex;align-items:center;margin-bottom:12px'>
            <div style='background:{colors["accent"]}22;padding:6px;border-radius:6px;margin-right:12px;font-size:1.1rem'>🎬</div>
            <div style='flex-grow:1'>
                <div style='font-size:0.75rem;color:{colors["subtext"]};text-transform:uppercase;letter-spacing:0.5px'>Sentiment</div>
                <div style='font-weight:600;font-size:0.95rem;color:{colors["text"]}'>Niranjan</div>
            </div>
        </div>
        <div style='display:flex;align-items:center;margin-bottom:12px'>
            <div style='background:{colors["accent"]}22;padding:6px;border-radius:6px;margin-right:12px;font-size:1.1rem'>🛍️</div>
            <div style='flex-grow:1'>
                <div style='font-size:0.75rem;color:{colors["subtext"]};text-transform:uppercase;letter-spacing:0.5px'>Clustering</div>
                <div style='font-weight:600;font-size:0.95rem;color:{colors["text"]}'>Vibhu</div>
            </div>
        </div>
        <div style='display:flex;align-items:center'>
            <div style='background:{colors["accent"]}22;padding:6px;border-radius:6px;margin-right:12px;font-size:1.1rem'>📩</div>
            <div style='flex-grow:1'>
                <div style='font-size:0.75rem;color:{colors["subtext"]};text-transform:uppercase;letter-spacing:0.5px'>Spam</div>
                <div style='font-weight:600;font-size:0.95rem;color:{colors["text"]}'>Gowtham</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("© 2025")





# ══════════════════════════════════════════════════════════════════════════════
#  HOME PAGE
# ══════════════════════════════════════════════════════════════════════════════

def _render_home(colors: dict) -> None:
    acc = colors["accent"]

    # ── Hero Section ───────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='text-align:center;padding:48px 20px 32px'>
        <div style="display:inline-block;padding:8px 20px;border-radius:50px;background:{colors['accent']}15;color:{colors['accent']};font-size:0.85rem;font-weight:700;letter-spacing:1px;text-transform:uppercase;margin-bottom:24px;border:1px solid {colors['accent']}30">
            Intelligent Analytics
        </div>
        <h1 style='font-size:3.5rem;font-weight:800;margin:0 0 16px;letter-spacing:-1px;background:linear-gradient(135deg,{acc},#a855f7,#06b6d4);-webkit-background-clip:text;-webkit-text-fill-color:transparent;'>Text Mining Project</h1>
        <p style='color:{colors["subtext"]};font-size:1.2rem;max-width:680px;
                  margin:0 auto;line-height:1.6'>
            An automated NLP system providing comprehensive insights through Sentiment Analysis, Customer Review Clustering, and Automated Spam Classification.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # ── Module Cards ───────────────────────────────────────────────────────
    c1, c2, c3 = st.columns(3)

    cards = [
        (c1, "#4F0599", "#7C3AED", "#a855f7",
         "🎬", "Sentiment Analysis", "Entertainment",
         "Logistic Regression + Naive Bayes on movie reviews. "
         "Live prediction, token highlighting, and model comparison.",
         "Niranjan"),
        (c2, "#064E3B", "#059669", "#34D399",
         "🛍️", "Review Clustering", "E-Commerce",
         "K-Means clustering on customer reviews with elbow charts, "
         "word clouds, and t-SNE visualisation.",
         "Vibhu"),
        (c3, "#78350F", "#D97706", "#FCD34D",
         "📩", "Spam Detection", "Communication",
         "Naive Bayes + Logistic Regression for SMS spam classification "
         "with batch upload and full metrics.",
         "Gowtham"),
    ]

    for col, g1, g2, g3, icon, title, domain, desc, dev in cards:
        with col:
            st.markdown(f"""
            <div class='tm-card' style='border-top:4px solid {g2}'>
                <div style='font-size:2rem;margin-bottom:8px'>{icon}</div>
                <h3 style='margin:0 0 4px;color:{g2};font-size:1.2rem'>{title}</h3>
                <p style='color:{colors["subtext"]};font-size:0.82rem;margin:0 0 10px'>
                    Domain: <b>{domain}</b>
                </p>
                <p style='color:{colors["text"]};font-size:0.9rem;line-height:1.55;
                          margin:0 0 14px'>
                    {desc}
                </p>
                <div style='background:linear-gradient(90deg,{g1},{g2});
                            color:white;padding:4px 12px;border-radius:20px;
                            display:inline-block;font-size:0.8rem;font-weight:600'>
                    Developer: {dev}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Tech Stack ─────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🛠️ Technology Stack")
    t1, t2, t3, t4, t5 = st.columns(5)
    for col, emoji, name, desc in [
        (t1, "🐍", "Python 3.10+", "Core language"),
        (t2, "⚡", "Streamlit", "Web UI framework"),
        (t3, "🔬", "scikit-learn", "ML algorithms"),
        (t4, "🗄️", "SQLite", "Database Engine"),
        (t5, "📝", "NLTK", "NLP preprocessing"),
    ]:
        with col:
            st.markdown(
                f"<div class='tm-card' style='text-align:center;padding:16px'>"
                f"<div style='font-size:1.8rem'>{emoji}</div>"
                f"<b style='color:{colors['accent']}'>{name}</b><br>"
                f"<small style='color:{colors['subtext']}'>{desc}</small>"
                f"</div>",
                unsafe_allow_html=True,
            )

    # ── Project Dashboard Table ─────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 System Dashboard")

    # High-level System Metrics
    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Active Modules", "3 NLP Pipelines", "Operational")
    mcol2.metric("Total ML Models", "4 Algorithms", "Ready")
    mcol3.metric("System Database", "Connected", "SQLite")
    mcol4.metric("Telemetry Tracking", "Active", "Live Session")
    
    st.markdown("<br>", unsafe_allow_html=True)

    dash_df = pd.DataFrame({
        "Module"     : ["🎬 Sentiment Analysis", "🛍️ Review Clustering", "📩 Spam Detection"],
        "Developer"  : ["👨‍💻 Niranjan", "👨‍💻 Vibhu", "👨‍💻 Gowtham"],
        "Domain"     : ["🎭 Entertainment", "🛒 E-Commerce", "📱 Communication"],
        "Dataset"    : ["📊 Movie Reviews", "📋 Customer Reviews", "💬 SMS Messages"],
        "Algorithms" : ["LR + Naive Bayes", "K-Means (K=2-8)", "NB + Logistic Reg"],
        "Key Feature": ["Live Web Scraping", "t-SNE Projections", "CSV Batch Inference"],
    })
    
    st.dataframe(
        dash_df, 
        use_container_width=True, 
        hide_index=True,
    )

    # ── How to Use ──────────────────────────────────────────────────────────
    st.markdown("---")
    with st.expander("📖 How to Use this Application", expanded=False):
        st.markdown("""
        1. **Use the sidebar** to navigate seamlessly between the specialized ML modules.
        2. **Sentiment Analysis** features **Live Web Scraping** to ingest real-time text context.
        3. **Review Clustering** allows you to discover hidden patterns using unsupervised ML.
        4. **Spam Detection** handles automated threat identification and allows bulk evaluations via CSV uploads.
        """)


# ══════════════════════════════════════════════════════════════════════════════
#  ROUTING
# ══════════════════════════════════════════════════════════════════════════════

if "Home" in module:
    _render_home(colors)
elif "Sentiment" in module:
    run_sentiment_module(colors)
elif "Clustering" in module:
    run_clustering_module(colors)
elif "Spam" in module:
    run_spam_module(colors)
