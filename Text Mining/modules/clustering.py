"""
clustering.py
─────────────────────────────────────────────────────────────────────────────
Module 2 – Customer Review Clustering (E-Commerce Domain)
Developer : Vibhu

Description
───────────
Loads customer review data, vectorises with TF-IDF, reduces dimensions with
TruncatedSVD, then applies K-Means clustering. The Streamlit UI provides:
  • Elbow method graph (inertia vs K)
  • Silhouette score chart
  • Interactive K slider
  • Cluster member display with top keywords
  • t-SNE 2-D scatter plot
  • Word cloud per cluster
"""

# ── Standard Library ──────────────────────────────────────────────────────────
import os
import sys
import random

# ── Third-party ───────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from wordcloud import WordCloud

# ── Scikit-learn ──────────────────────────────────────────────────────────────
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

# ── Local Utilities ───────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _ROOT)
from utils.preprocessing import clean_text

# ── Module Constants ──────────────────────────────────────────────────────────
_DATA_PATH = os.path.join(_ROOT, "data", "customer_reviews.csv")
_MAX_FEAT  = 3_000
_SVD_COMP  = 50
_K_MIN     = 2
_K_MAX     = 8
_RAND      = 42

# Cluster palette
_CLUSTER_COLORS = [
    "#7C3AED", "#10B981", "#F59E0B", "#EF4444",
    "#3B82F6", "#EC4899", "#14B8A6", "#F97316",
]


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def _load_data(file_bytes=None):
    """
    Try to load uploaded file or data/customer_reviews.csv.
    Falls back to synthetic data if file is unavailable.
    """
    try:
        if file_bytes is not None:
            import io
            df = pd.read_csv(io.BytesIO(file_bytes))
            rev_col = next((col for col in df.columns if 'review' in col.lower() or 'text' in col.lower()), df.columns[0])
            df = df[[rev_col]].rename(columns={rev_col: "review"}).dropna()
            df = df[df["review"].astype(str).str.strip().astype(bool)]
            source_name = f"Uploaded CSV ({len(df)} rows)"
            true_labels = [None] * len(df)
        else:
            df = pd.read_csv(_DATA_PATH)
            df = df[["review", "category"]].dropna()
            df = df[df["review"].str.strip().astype(bool)]
            source_name = f"customer_reviews.csv ({len(df)} rows)"
            true_labels = df["category"].tolist()
        
        if len(df) < 20:
            raise ValueError("Dataset too small.")
        return df["review"].astype(str).tolist(), true_labels, source_name
    except Exception:
        return _synthetic_fallback()


def _synthetic_fallback():
    """Generate 150 synthetic e-commerce reviews across three categories."""
    random.seed(_RAND)

    electronics = [
        "excellent battery life phone charges fast great screen display bright",
        "laptop outstanding processor speed memory storage performance exceptional",
        "camera resolution stunning photos video quality amazing device photos",
        "bluetooth headphones noise cancellation sound quality superb wireless",
        "smartwatch heart rate monitor fitness tracking sleep analysis accurate",
        "tablet lightweight portable responsive touchscreen accurate stylus",
        "gaming mouse precision sensor dpi adjustable ergonomic comfortable",
        "keyboard mechanical switches tactile feedback typing experience premium",
        "monitor vivid colours refresh rate gaming performance excellent display",
        "speaker bass crystal audio waterproof outdoor portable bluetooth",
    ]
    clothing = [
        "fabric soft comfortable true size fits perfectly cotton breathable",
        "dress elegant stylish colour vibrant stitching quality durable beautiful",
        "jeans slim fit stretch material comfortable waist flattering size",
        "jacket warm lightweight weather resistant zip pockets stylish design",
        "shirt breathable moisture wicking gym workout performance athletic",
        "shoes comfortable sole cushioning arch support walking comfortable daily",
        "saree beautiful embroidery traditional design festival occasion silk",
        "sweater warm cozy knit pattern winter season comfortable wool",
        "trousers flattering cut premium fabric professional office look",
        "hoodie soft warm casual everyday comfortable relaxed fit",
    ]
    food = [
        "delicious taste fresh ingredients organic spices flavourful healthy",
        "chocolate rich creamy smooth melts texture premium quality dark",
        "coffee beans roasted aroma flavour strong morning perfect blend",
        "snack crunchy healthy low calorie protein nuts seeds wholesome",
        "sauce tangy spicy recipe great pasta pizza versatile flavourful",
        "tea herbal calming sleep relaxation chamomile natural blend soothing",
        "biscuit crispy sweet butter texture packaging fresh crunchy",
        "juice fresh cold pressed vitamins minerals healthy nutritious drink",
        "honey pure natural flavour cooking baking ingredient organic raw",
        "granola crunchy oats nuts fruit yoghurt breakfast healthy morning",
    ]

    reviews, labels = [], []
    for pool, cat in [(electronics, "Electronics"), (clothing, "Clothing"), (food, "Food")]:
        for _ in range(50):
            rev = " ".join(random.sample(pool, k=min(3, len(pool))))
            reviews.append(rev)
            labels.append(cat)

    pairs = list(zip(reviews, labels))
    random.shuffle(pairs)
    rv, lb = zip(*pairs)
    return list(rv), list(lb), "Synthetic E-Commerce Reviews (150 samples)"


# ══════════════════════════════════════════════════════════════════════════════
#  ARTEFACT BUILDING
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def _build_artefacts(file_bytes=None):
    """
    1. Vectorise corpus with TF-IDF
    2. Reduce with TruncatedSVD + Normalizer (LSA)
    3. Pre-compute elbow + silhouette data
    Returns a bundle dict.
    """
    reviews, true_labels, source = _load_data(file_bytes)
    cleaned = [clean_text(r) for r in reviews]

    # TF-IDF
    vec     = TfidfVectorizer(max_features=_MAX_FEAT)
    X_tfidf = vec.fit_transform(cleaned)

    # LSA reduction (TruncatedSVD + L2-normalisation)
    svd       = TruncatedSVD(n_components=_SVD_COMP, random_state=_RAND)
    normalizer= Normalizer(copy=False)
    lsa_pipe  = make_pipeline(svd, normalizer)
    X_lsa     = lsa_pipe.fit_transform(X_tfidf)

    # Pre-compute elbow + silhouette across K range
    Ks, inertias, sils = [], [], []
    for k in range(_K_MIN, _K_MAX + 1):
        km_ = KMeans(n_clusters=k, random_state=_RAND, n_init=10)
        km_.fit(X_lsa)
        Ks.append(k)
        inertias.append(km_.inertia_)
        sils.append(silhouette_score(X_lsa, km_.labels_))

    return {
        "reviews"     : reviews,
        "cleaned"     : cleaned,
        "true_labels" : true_labels,
        "vec"         : vec,
        "X_tfidf"     : X_tfidf,
        "X_lsa"       : X_lsa,
        "vocab"       : vec.get_feature_names_out(),
        "n_total"     : len(reviews),
        "source"      : source,
        "ks"          : Ks,
        "inertias"    : inertias,
        "silhouettes" : sils,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  CLUSTERING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def _run_kmeans(X_lsa: np.ndarray, k: int):
    km = KMeans(n_clusters=k, random_state=_RAND, n_init=10)
    km.fit(X_lsa)
    return km, km.labels_


def _top_terms(art: dict, labels: np.ndarray, k: int, n: int = 8) -> dict[int, list]:
    """Return the top-n TF-IDF terms per cluster."""
    X_tfidf = art["X_tfidf"]
    vocab   = art["vocab"]
    terms   = {}
    for c in range(k):
        mask    = labels == c
        if mask.sum() == 0:
            terms[c] = []
            continue
        centroid = np.asarray(X_tfidf[mask].mean(axis=0)).flatten()
        top_idx  = centroid.argsort()[::-1][:n]
        terms[c] = [vocab[i] for i in top_idx]
    return terms


def _tsne_df(X_lsa: np.ndarray, labels: np.ndarray, reviews: list) -> pd.DataFrame:
    """Compute 2-D t-SNE projection."""
    perp = min(30, len(reviews) - 1)
    tsne = TSNE(n_components=2, random_state=_RAND, perplexity=perp, max_iter=500)
    emb  = tsne.fit_transform(X_lsa)
    return pd.DataFrame({
        "x"      : emb[:, 0],
        "y"      : emb[:, 1],
        "cluster": [f"Cluster {l}" for l in labels],
        "review" : [r[:80] + "…" if len(r) > 80 else r for r in reviews],
    })


def _make_wordcloud(text: str, bg_color: str = "#1E293B") -> plt.Figure:
    """Generate a word cloud figure from text."""
    wc = WordCloud(
        width=500, height=280,
        background_color=bg_color,
        colormap="plasma",
        max_words=60,
    ).generate(text or "no data")
    fig, ax = plt.subplots(figsize=(5, 2.8))
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


# ══════════════════════════════════════════════════════════════════════════════
#  STREAMLIT UI  — MODULE ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def run_clustering_module(colors: dict) -> None:
    """Render the full Clustering module UI."""
    GREEN = "#10B981"

    # ── Header Banner ──────────────────────────────────────────────────────
    st.markdown(f"""
    <div style='background-color: {colors["card_bg"]};
         border-top: 4px solid #10B981;
         border-left: 1px solid {colors["border"]}; border-right: 1px solid {colors["border"]}; border-bottom: 1px solid {colors["border"]};
         padding: 24px 28px; border-radius: 12px; margin-bottom: 24px;
         box-shadow: 0 4px 20px rgba(0,0,0,0.03);'>
      <div style='color: {colors["text"]}; font-size: 1.8rem; font-weight: 800; margin: 0; display:flex; align-items:center;'>
         <span style='margin-right:12px; font-size: 2.2rem;'>🛍️</span> Intelligent Review Clustering System
      </div>
      <div style='color: {colors["subtext"]}; margin-top: 8px; font-size: 1.05rem;'>
        Domain: <span style='font-weight:600; color:{colors["text"]};'>E-Commerce</span> &nbsp;|&nbsp; Developer: <span style='font-weight:600; color:{colors["text"]};'>Vibhu</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    upload_csv = st.file_uploader("📂 Upload Custom Dataset (.csv)", type=["csv"], help="Upload a CSV file containing reviews to cluster.")
    file_bytes = upload_csv.getvalue() if upload_csv else None

    # ── Build artefacts ────────────────────────────────────────────────────
    with st.spinner("Building clustering artefacts…"):
        art = _build_artefacts(file_bytes)

    if upload_csv:
        st.markdown("#### 📄 Data Preview")
        st.dataframe(pd.DataFrame({"Review": art["reviews"]}).head(), use_container_width=True)

    st.success(
        f"✅ Data loaded — **{art['source']}** ({art['n_total']} reviews)."
    )

    # Sidebar info
    with st.sidebar:
        st.markdown("#### ℹ️ Module 2 — Clustering")
        with st.expander("Details", expanded=False):
            st.write(f"**Dataset:** {art['source']}")
            st.write(f"**Samples:** {art['n_total']}")
            st.write(f"**Vectorizer:** TF-IDF ({_MAX_FEAT:,} features)")
            st.write(f"**Reduction:** TruncatedSVD ({_SVD_COMP} components) + L2 norm")
            st.write("**Model:** K-Means")

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 1 — ELBOW & SILHOUETTE
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("### 📉 Choosing K — Elbow & Silhouette")
    col_e, col_s = st.columns(2)

    with col_e:
        fig_elb = px.line(
            x=art["ks"], y=art["inertias"],
            markers=True, title="Elbow Method (Inertia vs K)",
            labels={"x": "Number of Clusters (K)", "y": "Inertia"},
            color_discrete_sequence=[GREEN],
        )
        fig_elb.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]},
            xaxis=dict(gridcolor=colors["border"]),
            yaxis=dict(gridcolor=colors["border"]),
        )
        st.plotly_chart(fig_elb, use_container_width=True)

    with col_s:
        fig_sil = px.line(
            x=art["ks"], y=art["silhouettes"],
            markers=True, title="Silhouette Score vs K",
            labels={"x": "Number of Clusters (K)", "y": "Silhouette Score"},
            color_discrete_sequence=["#7C3AED"],
        )
        fig_sil.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font={"color": colors["text"]},
            xaxis=dict(gridcolor=colors["border"]),
            yaxis=dict(gridcolor=colors["border"]),
        )
        st.plotly_chart(fig_sil, use_container_width=True)

    st.info(
        "📌 **Tip:** Choose K at the 'elbow' of the inertia curve "
        "or where silhouette score peaks."
    )

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 2 — INTERACTIVE K-MEANS
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🚀 Intelligent Review Clustering System")

    k_val = st.slider(
        "Select number of clusters (K):",
        min_value=_K_MIN, max_value=_K_MAX, value=3,
        key="m2_k_slider",
    )

    with st.spinner(f"Running K-Means with K={k_val}…"):
        km, km_labels = _run_kmeans(art["X_lsa"], k_val)
        top_terms     = _top_terms(art, km_labels, k_val)

    # ── Summary box ──
    st.markdown("#### 📌 Cluster Summary")
    sizes = pd.Series(km_labels).value_counts().sort_index()
    largest_cluster = sizes.idxmax()
    sil_score = silhouette_score(art["X_lsa"], km_labels)
    
    dominant_topic = top_terms[largest_cluster][0].capitalize() if top_terms[largest_cluster] else "General"
    st.success(f"**Total Reviews:** {art['n_total']} &nbsp;|&nbsp; **Clusters (K):** {k_val} &nbsp;|&nbsp; **Largest Cluster:** Cluster {largest_cluster} ({sizes.max()} reviews) &nbsp;|&nbsp; **Dominant Topic:** {dominant_topic}")
    
    m_col1, m_col2 = st.columns(2)
    m_col1.metric("Clustering Model", "K-Means + TF-IDF")
    m_col2.metric("Silhouette Score", f"{sil_score:.3f}")

    # ── Distribution Graph ──
    st.markdown("#### 📊 Cluster Distribution")
    pie_fig = px.pie(names=[f"Cluster {i}" for i in sizes.index], values=sizes.values, 
                     color_discrete_sequence=_CLUSTER_COLORS, hole=0.4)
    pie_fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={"color": colors["text"]}, margin=dict(t=10, b=10))
    st.plotly_chart(pie_fig, use_container_width=True)

    # ── Search & Export ──
    st.markdown("---")
    c_srch, c_exp = st.columns([3, 1])
    with c_srch:
        search_term = st.text_input("🔍 Search reviews across clusters:", placeholder="Type a keyword...")
    with c_exp:
        st.markdown("<br>", unsafe_allow_html=True)
        csv_data = pd.DataFrame({"Review": art["reviews"], "Cluster": km_labels}).to_csv(index=False)
        st.download_button("💾 Export to CSV", data=csv_data, file_name="clustered_reviews.csv", mime="text/csv", use_container_width=True)

    # ── Insights ──
    st.markdown("#### 💡 Cluster Insights")
    for c in range(k_val):
        members_idx = [i for i, l in enumerate(km_labels) if l == c]
        
        # Filter samples based on search term
        matches = [art["reviews"][i] for i in members_idx]
        if search_term:
            matches = [m for m in matches if search_term.lower() in m.lower()]
            
        cluster_name = f"{top_terms[c][0].capitalize()} Issues" if top_terms[c] else "General Issues"
        with st.expander(
            f"Cluster {c} → \"{cluster_name}\" ({sizes.get(c, 0)} reviews) "
            f"| Themes: {', '.join(top_terms[c][:3]).capitalize()}",
            expanded=(c == 0),
        ):
            kw_col, wc_col = st.columns([1, 1])
            with kw_col:
                st.markdown("**Top Keywords:**")
                kw_df = pd.DataFrame(
                    {"Keyword": top_terms[c]},
                    index=range(1, len(top_terms[c]) + 1),
                )
                st.dataframe(kw_df, use_container_width=True)

                st.markdown("**Sample Reviews:**")
                samples = matches[:6] # Show up to 6 matches
                txt_col = colors["text"]
                if not samples:
                    st.write(f"*No matching reviews found in Cluster {c}.*")
                else:
                    for s in samples:
                        st.markdown(
                            f"<div style='padding:6px 10px;border-left:3px solid "
                            f"{_CLUSTER_COLORS[c % len(_CLUSTER_COLORS)]};margin:4px 0;"
                            f"color:{txt_col};font-size:0.88rem'>{s}</div>",
                            unsafe_allow_html=True,
                        )
            with wc_col:
                wc_text = " ".join(art["cleaned"][i] for i in members_idx)
                is_dark = colors["bg"] < "#888"
                bg_col  = "#1E293B" if is_dark else "#F1F5F9"
                fig_wc  = _make_wordcloud(wc_text, bg_color=bg_col)
                st.pyplot(fig_wc)
                plt.close(fig_wc)

    # ─────────────────────────────────────────────────────────────────────
    # SECTION 3 — t-SNE VISUALISATION
    # ─────────────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 🗺️ t-SNE 2-D Cluster Visualisation")

    with st.spinner("Computing t-SNE projection…"):
        tsne_df = _tsne_df(art["X_lsa"], km_labels, art["reviews"])

    color_map = {f"Cluster {i}": _CLUSTER_COLORS[i % len(_CLUSTER_COLORS)] for i in range(k_val)}
    fig_tsne  = px.scatter(
        tsne_df, x="x", y="y", color="cluster",
        hover_data={"review": True, "x": False, "y": False},
        color_discrete_map=color_map,
        title="t-SNE — Document Clusters",
        labels={"x": "t-SNE Dim 1", "y": "t-SNE Dim 2"},
    )
    fig_tsne.update_traces(marker=dict(size=8, opacity=0.8))
    fig_tsne.update_layout(
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font={"color": colors["text"]},
        xaxis=dict(gridcolor=colors["border"], zeroline=False),
        yaxis=dict(gridcolor=colors["border"], zeroline=False),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
    )
    st.plotly_chart(fig_tsne, use_container_width=True)


