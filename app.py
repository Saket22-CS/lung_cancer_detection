# ══════════════════════════════════════════════════════════════════════════════
#  LUNG CANCER DETECTION SYSTEM  —  Streamlit App
#  Run:  streamlit run app.py
# ══════════════════════════════════════════════════════════════════════════════

import os, gc, time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
import cv2
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model, Model
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import warnings
warnings.filterwarnings("ignore")

os.environ["TF_USE_LEGACY_KERAS"] = "1"

# ─────────────────────────────────────────────────────────────────────────────
#  PAGE CONFIG  (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LungAI — Cancer Detection",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
#  GLOBAL CSS  — dark medical aesthetic with teal accents
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #080c14;
    color: #e2e8f0;
}
.stApp { background: #080c14; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1321 0%, #0a1628 100%) !important;
    border-right: 1px solid #1e3a5f !important;
    display: block !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"] > div {
    display: block !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"] * { 
    color: #cbd5e1 !important; 
}
[data-testid="collapsedControl"] {
    display: block !important;
    color: #38bdf8 !important;
}

/* ── Hide default streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2.5rem 3rem; max-width: 1400px; }

/* ── Custom scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0d1321; }
::-webkit-scrollbar-thumb { background: #0ea5e9; border-radius: 3px; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, #0d1b2a 0%, #0a2540 40%, #071a35 100%);
    border: 1px solid #1e3a5f;
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 240px; height: 240px;
    background: radial-gradient(circle, rgba(14,165,233,0.12) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 160px;
    background: radial-gradient(ellipse, rgba(6,182,212,0.07) 0%, transparent 70%);
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.6rem;
    font-weight: 700;
    background: linear-gradient(90deg, #38bdf8 0%, #06b6d4 50%, #67e8f9 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0 0 6px 0;
    letter-spacing: -1px;
}
.hero-sub {
    font-size: 1.05rem;
    color: #64748b;
    font-weight: 300;
    letter-spacing: 0.5px;
}
.hero-badges {
    display: flex; gap: 10px; margin-top: 20px; flex-wrap: wrap;
}
.badge {
    background: rgba(14,165,233,0.1);
    border: 1px solid rgba(14,165,233,0.3);
    color: #38bdf8;
    padding: 4px 14px;
    border-radius: 20px;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    letter-spacing: 0.5px;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Space Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    color: #0ea5e9;
    margin: 32px 0 16px 0;
    display: flex; align-items: center; gap: 10px;
}
.section-header::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, #1e3a5f, transparent);
}

/* ── Glass cards ── */
.glass-card {
    background: rgba(13, 27, 42, 0.8);
    backdrop-filter: blur(12px);
    border: 1px solid #1e3a5f;
    border-radius: 14px;
    padding: 24px;
    margin-bottom: 16px;
    transition: border-color 0.3s;
}
.glass-card:hover { border-color: #0ea5e9; }

/* ── Prediction result card ── */
.pred-card {
    border-radius: 16px;
    padding: 28px 24px;
    text-align: center;
    border: 2px solid;
    position: relative;
    overflow: hidden;
}
.pred-card-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    opacity: 0.6;
    margin-bottom: 8px;
}
.pred-card-model {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 14px;
}
.pred-card-class {
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 6px;
}
.pred-card-conf {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
}
.pred-card-label { font-size: 0.78rem; opacity: 0.5; margin-top: 2px; }

/* ── Metric pill ── */
.metric-row {
    display: flex; gap: 12px; flex-wrap: wrap; margin-bottom: 20px;
}
.metric-pill {
    flex: 1; min-width: 100px;
    background: rgba(14,165,233,0.07);
    border: 1px solid rgba(14,165,233,0.2);
    border-radius: 12px;
    padding: 14px 16px;
    text-align: center;
}
.metric-pill-val {
    font-family: 'Space Mono', monospace;
    font-size: 1.5rem;
    font-weight: 700;
    color: #38bdf8;
}
.metric-pill-lbl { font-size: 0.72rem; color: #64748b; margin-top: 3px; }

/* ── Progress bars ── */
.prob-bar-wrap { margin: 6px 0; }
.prob-bar-label {
    display: flex; justify-content: space-between;
    font-size: 0.82rem; margin-bottom: 4px;
}
.prob-bar-bg {
    background: #1e3a5f; border-radius: 4px; height: 10px; overflow: hidden;
}
.prob-bar-fill {
    height: 100%; border-radius: 4px;
    transition: width 0.8s cubic-bezier(.4,0,.2,1);
}

/* ── Status dot ── */
.status-dot {
    display: inline-block;
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #22c55e;
    box-shadow: 0 0 8px #22c55e;
    margin-right: 6px;
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%,100% { opacity: 1; }
    50%      { opacity: 0.4; }
}

/* ── Diagnosis banner ── */
.dx-banner {
    border-radius: 14px;
    padding: 24px 28px;
    border-left: 5px solid;
    margin-top: 20px;
}
.dx-title {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    font-weight: 700;
    margin-bottom: 10px;
}
.dx-body { font-size: 0.92rem; line-height: 1.75; color: #94a3b8; }

/* ── Warning banner ── */
.warn-box {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.3);
    border-radius: 10px;
    padding: 12px 18px;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-top: 16px;
}

/* ── Upload zone styling ── */
[data-testid="stFileUploader"] {
    background: rgba(14,165,233,0.04) !important;
    border: 2px dashed rgba(14,165,233,0.25) !important;
    border-radius: 14px !important;
    padding: 12px !important;
}

/* ── Streamlit widget overrides ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: #0d1b2a !important;
    border-color: #1e3a5f !important;
    color: #e2e8f0 !important;
}
.stSlider .st-bm { background: #0ea5e9 !important; }
div[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    color: #38bdf8 !important;
    font-size: 1.8rem !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    font-family: 'Space Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 1px;
}
.stTabs [aria-selected="true"] {
    color: #38bdf8 !important;
    border-bottom-color: #38bdf8 !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid #1e3a5f;
    gap: 4px;
}
button[kind="primary"] {
    background: linear-gradient(135deg, #0ea5e9, #06b6d4) !important;
    border: none !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.8rem !important;
    letter-spacing: 1px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = ["lung_aca", "lung_n", "lung_scc"]
CLASS_FULL  = {
    "lung_aca": "Lung Adenocarcinoma",
    "lung_n"  : "Normal Lung Tissue",
    "lung_scc": "Squamous Cell Carcinoma",
}
CLASS_EMOJI = {"lung_aca": "🔴", "lung_n": "🟢", "lung_scc": "🟠"}
CLASS_DESC  = {
    "lung_aca": (
        "The most common primary lung malignancy, originating in peripheral glandular "
        "cells. Characterised by acinar, papillary, or micropapillary growth patterns. "
        "Often associated with EGFR/ALK mutations and presents in non-smokers."
    ),
    "lung_n"  : (
        "Healthy lung parenchyma with intact alveolar architecture. No evidence of "
        "abnormal cellular proliferation, nuclear atypia, or structural disruption. "
        "Regular, uniform cell morphology throughout the tissue."
    ),
    "lung_scc": (
        "Arises from bronchial epithelial cells, strongly linked to tobacco exposure. "
        "Characterised by keratin pearls and intercellular bridges. Typically centrally "
        "located near the main bronchi and amenable to surgical resection if detected early."
    ),
}
# Hex colours matched to each class
C_ACA  = "#ef4444"
C_NRM  = "#22c55e"
C_SCC  = "#f97316"
C_TEAL = "#0ea5e9"
CLASS_HEX = {"lung_aca": C_ACA, "lung_n": C_NRM, "lung_scc": C_SCC}

IMG_SIZE = 224

# ─────────────────────────────────────────────────────────────────────────────
#  UTILITY HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def preprocess_image(uploaded):
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    img  = cv2.imdecode(data, cv2.IMREAD_COLOR)
    img  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    norm    = np.expand_dims(resized / 255.0, 0).astype(np.float32)
    return img, resized, norm

def find_last_conv(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
        if hasattr(layer, "layers"):
            for sub in reversed(layer.layers):
                if isinstance(sub, tf.keras.layers.Conv2D):
                    return sub.name
    return None

def get_gradcam(model, img_array, last_conv):
    try:
        grad_model = Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv).output, model.output],
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            idx   = tf.argmax(preds[0])
            score = preds[:, idx]
        grads   = tape.gradient(score, conv_out)
        pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception:
        return None

def overlay_gradcam(orig_img, heatmap, alpha=0.45):
    h = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
    h = cv2.applyColorMap(np.uint8(255 * h), cv2.COLORMAP_INFERNO)
    h = cv2.cvtColor(h, cv2.COLOR_BGR2RGB)
    return cv2.addWeighted(orig_img, 1 - alpha, h, alpha, 0)

def confidence_color(conf):
    if conf >= 90: return C_NRM
    if conf >= 70: return C_TEAL
    if conf >= 50: return "#f59e0b"
    return C_ACA

def make_donut(value, color, size=1.4):
    fig, ax = plt.subplots(figsize=(size, size), facecolor="none")
    ax.set_facecolor("none")
    ax.pie(
        [value, 100 - value],
        colors=[color, "#1e3a5f"],
        startangle=90,
        counterclock=False,
        wedgeprops={"width": 0.32, "edgecolor": "none"},
    )
    ax.text(0, 0, f"{value:.0f}%", ha="center", va="center",
            fontsize=9, fontweight="bold", color=color,
            fontfamily="monospace")
    fig.tight_layout(pad=0)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS  (cached — only loads once)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_all_models(directory):
    loaded = {}

    # ── Rebuild architectures locally (same as training code) ─────────────
    def build_custom_cnn():
        from tensorflow.keras.layers import (Conv2D, MaxPooling2D, Dense, Dropout,
                                              GlobalAveragePooling2D, BatchNormalization)
        model = tf.keras.Sequential([
            Conv2D(32,(3,3),activation='relu',padding='same',input_shape=(224,224,3)),
            BatchNormalization(), Conv2D(32,(3,3),activation='relu',padding='same'),
            MaxPooling2D(2,2),
            Conv2D(64,(3,3),activation='relu',padding='same'),
            BatchNormalization(), Conv2D(64,(3,3),activation='relu',padding='same'),
            MaxPooling2D(2,2),
            Conv2D(128,(3,3),activation='relu',padding='same'),
            BatchNormalization(), Conv2D(128,(3,3),activation='relu',padding='same'),
            MaxPooling2D(2,2),
            GlobalAveragePooling2D(),
            Dense(256,activation='relu'), BatchNormalization(), Dropout(0.5),
            Dense(3,activation='softmax')
        ], name='Custom_CNN')
        return model

    def build_vgg16():
        base  = tf.keras.applications.VGG16(
            weights=None, include_top=False, input_shape=(224,224,3))
        model = tf.keras.Sequential([
            base, tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3,activation='softmax')
        ], name='VGG16')
        return model

    def build_resnet50():
        base  = tf.keras.applications.ResNet50(
            weights=None, include_top=False, input_shape=(224,224,3))
        model = tf.keras.Sequential([
            base, tf.keras.layers.GlobalAveragePooling2D(),
            tf.keras.layers.Dense(256,activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(3,activation='softmax')
        ], name='ResNet50')
        return model

    builders = {
        "Custom CNN": build_custom_cnn,
        "VGG16"     : build_vgg16,
        "ResNet50"  : build_resnet50,
    }
    weight_files = {
        "Custom CNN": "custom_cnn.weights.h5",
        "VGG16"     : "vgg16.weights.h5",
        "ResNet50"  : "resnet50.weights.h5",
    }

    for name, builder in builders.items():
        wpath = os.path.join(directory, weight_files[name])
        if os.path.exists(wpath):
            try:
                model = builder()
                # Build model with dummy pass before loading weights
                _ = model(np.zeros((1,224,224,3), dtype=np.float32),
                          training=False)
                model.load_weights(wpath)
                loaded[name] = model
                st.success(f"✅ {name} loaded")
            except Exception as e:
                st.warning(f"⚠️ {name} failed: {str(e)[:150]}")
        else:
            st.warning(f"⚠️ {name} weights not found → {wpath}")
    return loaded

# ─────────────────────────────────────────────────────────────────────────────
#  SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 8px 0 24px'>
        <div style='font-size:2.8rem'>🫁</div>
        <div style='font-family:"Space Mono",monospace; font-size:0.9rem;
                    color:#38bdf8; letter-spacing:2px'>LUNG AI</div>
        <div style='font-size:0.7rem; color:#475569; margin-top:4px'>
            v2.0 · Deep Learning Suite
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**📁 Model Directory**")
    model_dir = st.text_input(
        label="Model Directory Path",
        value="models",
        label_visibility="hidden"
    )

    st.markdown("---")
    st.markdown("**🧠 Active Models**")
    use_cnn    = st.checkbox("Custom CNN",  value=True)
    use_vgg    = st.checkbox("VGG16",       value=True)
    use_resnet = st.checkbox("ResNet50",    value=True)

    st.markdown("---")
    st.markdown("**🔬 Analysis Options**")
    show_gradcam   = st.checkbox("Grad-CAM Heatmaps",       value=True)
    show_probs     = st.checkbox("Probability Breakdown",    value=True)
    show_agreement = st.checkbox("Model Agreement Panel",    value=True)
    show_donut     = st.checkbox("Confidence Donuts",        value=True)

    st.markdown("---")
    st.markdown("**🎨 Grad-CAM Style**")
    gcam_alpha = st.slider("Overlay Intensity", 0.2, 0.8, 0.45, 0.05)
    gcam_cmap  = st.selectbox(
        "Colormap",
        ["INFERNO", "JET", "HOT", "PLASMA", "TURBO"],
        index=0,
    )
    CMAP_MAP = {
        "INFERNO": cv2.COLORMAP_INFERNO,
        "JET"    : cv2.COLORMAP_JET,
        "HOT"    : cv2.COLORMAP_HOT,
        "PLASMA" : cv2.COLORMAP_PLASMA,
        "TURBO"  : cv2.COLORMAP_TURBO,
    }

    st.markdown("---")
    st.markdown("""
    <div style='font-size:0.72rem; color:#334155; line-height:1.7'>
        Dataset: LC25000 · Lung Histopathology<br>
        Classes: ACA · Normal · SCC<br>
        Framework: TensorFlow 2.x<br><br>
        <span style='color:#ef4444'>⚠ Research use only.</span><br>
        Not a clinical diagnostic tool.
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  LOAD MODELS
# ─────────────────────────────────────────────────────────────────────────────
loading_placeholder = st.empty()
with loading_placeholder.container():
    with st.spinner("Initialising AI models…"):
        all_models = load_all_models(model_dir)
time.sleep(1.5)
loading_placeholder.empty()

active = {}
if use_cnn    and "Custom CNN" in all_models: active["Custom CNN"] = all_models["Custom CNN"]
if use_vgg    and "VGG16"      in all_models: active["VGG16"]      = all_models["VGG16"]
if use_resnet and "ResNet50"   in all_models: active["ResNet50"]   = all_models["ResNet50"]

# ─────────────────────────────────────────────────────────────────────────────
#  HERO BANNER
# ─────────────────────────────────────────────────────────────────────────────
loaded_names = list(all_models.keys())
badge_html   = "".join(f"<span class='badge'>{n}</span>" for n in loaded_names)
badge_html  += f"<span class='badge'>LC25000 Dataset</span>"
badge_html  += f"<span class='badge'>TensorFlow {tf.__version__}</span>"

st.markdown(f"""
<div class='hero-banner'>
    <div class='hero-title'>🫁 LungAI Detection</div>
    <div class='hero-sub'>
        Deep learning–powered histopathology classification &nbsp;·&nbsp;
        <span class='status-dot'></span>
        {len(active)} model{"s" if len(active) != 1 else ""} active
    </div>
    <div class='hero-badges'>{badge_html}</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  MODEL STATUS ROW
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>SYSTEM STATUS</div>", unsafe_allow_html=True)

stat_cols = st.columns(4)
for i, (label, val, icon) in enumerate([
    ("Models Loaded",    str(len(all_models)),  "🔋"),
    ("Models Active",    str(len(active)),       "⚡"),
    ("Input Size",       "224 × 224",            "🖼️"),
    ("Classes",          "3",                    "🏷️"),
]):
    with stat_cols[i]:
        st.markdown(f"""
        <div class='glass-card' style='text-align:center; padding:18px'>
            <div style='font-size:1.6rem'>{icon}</div>
            <div style='font-family:"Space Mono",monospace; font-size:1.4rem;
                        color:#38bdf8; font-weight:700'>{val}</div>
            <div style='font-size:0.75rem; color:#475569; margin-top:2px'>{label}</div>
        </div>
        """, unsafe_allow_html=True)

if not active:
    st.error("❌ No models loaded. Check the directory path in the sidebar.")
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
#  TABS
# ─────────────────────────────────────────────────────────────────────────────
tab_predict, tab_batch, tab_info, tab_about = st.tabs([
    "🔬  PREDICT",
    "📂  BATCH ANALYSIS",
    "ℹ️  CLASS INFO",
    "📊  PROJECT INFO",
])

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 1 — SINGLE IMAGE PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
with tab_predict:
    st.markdown("<div class='section-header'>UPLOAD BIOPSY IMAGE</div>",
                unsafe_allow_html=True)

    up_col, hint_col = st.columns([3, 2])
    with up_col:
        uploaded = st.file_uploader(
            "Drop a lung histopathology image here",
            type=["jpg", "jpeg", "png"],
            label_visibility="collapsed",
        )
    with hint_col:
        st.markdown("""
        <div class='glass-card' style='padding:18px 20px'>
            <div style='font-size:0.7rem; letter-spacing:2px;
                        color:#0ea5e9; font-family:"Space Mono",monospace;
                        margin-bottom:10px'>ACCEPTED INPUTS</div>
            <div style='font-size:0.85rem; color:#94a3b8; line-height:2'>
                ✦ &nbsp;JPEG / PNG histopathology slides<br>
                ✦ &nbsp;Micro-biopsy tissue images<br>
                ✦ &nbsp;LC25000-style magnification<br>
                ✦ &nbsp;Any resolution (resized to 224×224)
            </div>
        </div>
        """, unsafe_allow_html=True)

    # ── If file uploaded ───────────────────────────────────────────────────
    if uploaded:
        uploaded.seek(0)
        orig_img, resized_img, img_norm = preprocess_image(uploaded)

        # ── Analyse with all active models ────────────────────────────────
        results = {}
        progress = st.progress(0, text="Running inference…")
        for i, (name, model) in enumerate(active.items()):
            probs      = model.predict(img_norm, verbose=0)[0]
            pred_idx   = int(np.argmax(probs))
            results[name] = {
                "probs"    : probs,
                "pred_idx" : pred_idx,
                "pred_cls" : CLASS_NAMES[pred_idx],
                "conf"     : float(probs[pred_idx]) * 100,
            }
            progress.progress((i + 1) / len(active), text=f"✅ {name} done")
        time.sleep(0.3)
        progress.empty()

        st.markdown("<div class='section-header'>PREDICTION RESULTS</div>",
                    unsafe_allow_html=True)

        # ── Image + Grad-CAM row ───────────────────────────────────────────
        img_cols = st.columns([1] + [1] * len(active))

        with img_cols[0]:
            st.markdown("""
            <div style='font-size:0.7rem; letter-spacing:2px; color:#64748b;
                        font-family:"Space Mono",monospace; margin-bottom:6px'>
                ORIGINAL
            </div>""", unsafe_allow_html=True)
            st.image(resized_img, width='stretch')

        if show_gradcam:
            for col, (name, model) in zip(img_cols[1:], active.items()):
                last_conv = find_last_conv(model)
                heatmap   = get_gradcam(model, img_norm, last_conv) if last_conv else None
                with col:
                    st.markdown(f"""
                    <div style='font-size:0.7rem; letter-spacing:2px; color:#64748b;
                                font-family:"Space Mono",monospace; margin-bottom:6px'>
                        GRAD-CAM · {name.upper()}
                    </div>""", unsafe_allow_html=True)
                    if heatmap is not None:
                        h    = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
                        hmap = cv2.applyColorMap(np.uint8(255 * h),
                                                  CMAP_MAP[gcam_cmap])
                        hmap = cv2.cvtColor(hmap, cv2.COLOR_BGR2RGB)
                        over = cv2.addWeighted(resized_img, 1 - gcam_alpha,
                                               hmap, gcam_alpha, 0)
                        st.image(over, width='stretch')
                    else:
                        st.image(resized_img, width='stretch')
                        st.caption("Grad-CAM unavailable")

        st.markdown("---")

        # ── Prediction cards ───────────────────────────────────────────────
        card_cols = st.columns(len(active))
        for col, (name, res) in zip(card_cols, results.items()):
            cls  = res["pred_cls"]
            hex_ = CLASS_HEX[cls]
            conf = res["conf"]
            ccol = confidence_color(conf)

            with col:
                st.markdown(f"""
                <div class='pred-card' style='background:rgba(13,27,42,0.9);
                            border-color:{hex_}22;
                            box-shadow:0 0 30px {hex_}18'>
                    <div class='pred-card-title'>MODEL RESULT</div>
                    <div class='pred-card-model'>{name}</div>
                    <div style='font-size:1.8rem; margin:6px 0'>
                        {CLASS_EMOJI[cls]}
                    </div>
                    <div class='pred-card-class' style='color:{hex_}'>
                        {CLASS_FULL[cls]}
                    </div>
                    <div class='pred-card-conf' style='color:{ccol}'>
                        {conf:.1f}%
                    </div>
                    <div class='pred-card-label'>confidence</div>
                </div>
                """, unsafe_allow_html=True)

        # ── Confidence donut row ───────────────────────────────────────────
        if show_donut and len(active) > 0:
            st.markdown("<div class='section-header'>CONFIDENCE DONUTS</div>",
                        unsafe_allow_html=True)
            donut_cols = st.columns(len(active) * len(CLASS_NAMES))
            idx = 0
            for name, res in results.items():
                for ci, cls in enumerate(CLASS_NAMES):
                    with donut_cols[idx]:
                        val = float(res["probs"][ci]) * 100
                        fig = make_donut(val, CLASS_HEX[cls], size=1.3)
                        st.pyplot(fig, width='stretch')
                        plt.close(fig)
                        st.markdown(
                            f"<div style='text-align:center;font-size:0.62rem;"
                            f"color:#475569;margin-top:-8px'>"
                            f"{cls.replace('lung_','').upper()}</div>",
                            unsafe_allow_html=True)
                    idx += 1

        # ── Probability bars ───────────────────────────────────────────────
        if show_probs:
            st.markdown("<div class='section-header'>PROBABILITY BREAKDOWN</div>",
                        unsafe_allow_html=True)
            pb_cols = st.columns(len(active))
            for col, (name, res) in zip(pb_cols, results.items()):
                with col:
                    st.markdown(f"""
                    <div class='glass-card'>
                        <div style='font-family:"Space Mono",monospace;
                                    font-size:0.75rem; color:#0ea5e9;
                                    margin-bottom:14px'>{name}</div>
                    """, unsafe_allow_html=True)
                    for ci, cls in enumerate(CLASS_NAMES):
                        pct = float(res["probs"][ci]) * 100
                        hex_ = CLASS_HEX[cls]
                        st.markdown(f"""
                        <div class='prob-bar-wrap'>
                            <div class='prob-bar-label'>
                                <span style='color:{hex_}'>{CLASS_FULL[cls]}</span>
                                <span style='font-family:"Space Mono",monospace;
                                             color:{hex_}'>{pct:.1f}%</span>
                            </div>
                            <div class='prob-bar-bg'>
                                <div class='prob-bar-fill'
                                     style='width:{pct}%;
                                            background:linear-gradient(90deg,{hex_}88,{hex_})'></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    st.markdown("</div>", unsafe_allow_html=True)

        # ── Model agreement ────────────────────────────────────────────────
        if show_agreement and len(active) > 1:
            st.markdown("<div class='section-header'>MODEL AGREEMENT</div>",
                        unsafe_allow_html=True)
            preds_list = [r["pred_cls"] for r in results.values()]
            unique     = set(preds_list)
            agree      = len(unique) == 1

            agree_color = "#22c55e" if agree else "#f97316"
            agree_icon  = "✅" if agree else "⚠️"
            agree_text  = (
                f"All {len(active)} models agree on <b style='color:{CLASS_HEX[preds_list[0]]}'>"
                f"{CLASS_FULL[preds_list[0]]}</b>"
                if agree else
                "Models have differing predictions — review individual results."
            )

            st.markdown(f"""
            <div class='glass-card' style='border-color:{agree_color}44'>
                <div style='font-size:1.1rem; font-weight:600; margin-bottom:10px'>
                    {agree_icon} &nbsp;{agree_text}
                </div>
                <div style='display:flex; gap:16px; flex-wrap:wrap; margin-top:10px'>
            """, unsafe_allow_html=True)

            vote_html = ""
            for name, res in results.items():
                cls  = res["pred_cls"]
                hex_ = CLASS_HEX[cls]
                vote_html += (
                    f"<div style='background:rgba(13,27,42,0.9); border:1px solid {hex_}55;"
                    f"border-radius:10px; padding:10px 18px; text-align:center'>"
                    f"<div style='font-size:0.7rem; color:#64748b; font-family:"
                    f'"Space Mono",monospace\'>{name}</div>'
                    f"<div style='color:{hex_}; font-weight:600; margin-top:4px'>"
                    f"{CLASS_FULL[cls]}</div>"
                    f"<div style='font-family:\"Space Mono\",monospace; color:{hex_}; font-size:0.9rem'>"
                    f"{res['conf']:.1f}%</div></div>"
                )

            st.markdown(vote_html + "</div></div>", unsafe_allow_html=True)

        # ── Diagnosis card ─────────────────────────────────────────────────
        best_name = "ResNet50" if "ResNet50" in results else list(results.keys())[0]
        best_cls  = results[best_name]["pred_cls"]
        best_hex  = CLASS_HEX[best_cls]

        st.markdown("<div class='section-header'>CLINICAL CONTEXT</div>",
                    unsafe_allow_html=True)
        st.markdown(f"""
        <div class='dx-banner'
             style='background:rgba(13,27,42,0.85); border-color:{best_hex}'>
            <div class='dx-title' style='color:{best_hex}'>
                {CLASS_EMOJI[best_cls]}&nbsp; {CLASS_FULL[best_cls]}
            </div>
            <div class='dx-body'>{CLASS_DESC[best_cls]}</div>
        </div>
        <div class='warn-box'>
            ⚠️ &nbsp; This system is intended for research and educational purposes only.
            Results must not be used for clinical diagnosis without review by a qualified
            medical professional.
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── Empty state ────────────────────────────────────────────────────
        st.markdown("""
        <div style='text-align:center; padding:80px 20px; color:#1e3a5f'>
            <div style='font-size:5rem; margin-bottom:16px'>🫁</div>
            <div style='font-family:"Space Mono",monospace; font-size:1rem;
                        color:#334155; letter-spacing:2px'>
                UPLOAD AN IMAGE TO BEGIN
            </div>
            <div style='font-size:0.85rem; color:#1e3a5f; margin-top:10px'>
                Supported: JPG · PNG · Histopathology biopsy slides
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 2 — BATCH ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_batch:
    st.markdown("<div class='section-header'>BATCH IMAGE ANALYSIS</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        Upload multiple biopsy images at once and get predictions for all of them
        from the best available model (ResNet50 preferred).
    </div>
    """, unsafe_allow_html=True)

    batch_files = st.file_uploader(
        "Upload multiple images",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    batch_model_name = (
        "ResNet50" if "ResNet50" in active
        else ("VGG16" if "VGG16" in active
              else (list(active.keys())[0] if active else None))
    )

    if batch_files and batch_model_name:
        model_b  = active[batch_model_name]
        st.markdown(f"""
        <div style='font-size:0.8rem; color:#64748b; margin-bottom:16px'>
            Running with: &nbsp;<span style='color:#38bdf8;
            font-family:"Space Mono",monospace'>{batch_model_name}</span>
            &nbsp;·&nbsp; {len(batch_files)} images
        </div>
        """, unsafe_allow_html=True)

        bar   = st.progress(0)
        rows  = []
        thumb_cols = st.columns(min(len(batch_files), 6))

        for i, f in enumerate(batch_files):
            f.seek(0)
            _, img_r, img_n = preprocess_image(f)
            probs   = model_b.predict(img_n, verbose=0)[0]
            pi      = int(np.argmax(probs))
            rows.append({
                "File"      : f.name,
                "Prediction": CLASS_FULL[CLASS_NAMES[pi]],
                "Confidence": f"{probs[pi]*100:.1f}%",
                "ACA %"     : f"{probs[0]*100:.1f}",
                "Normal %"  : f"{probs[1]*100:.1f}",
                "SCC %"     : f"{probs[2]*100:.1f}",
            })
            if i < 6:
                with thumb_cols[i]:
                    st.image(img_r, caption=f.name[:12], width='stretch')
                    col_hex = CLASS_HEX[CLASS_NAMES[pi]]
                    st.markdown(
                        f"<div style='text-align:center; font-size:0.7rem;"
                        f"color:{col_hex}; font-family:\"Space Mono\",monospace'>"
                        f"{CLASS_NAMES[pi].replace('lung_','').upper()}</div>",
                        unsafe_allow_html=True)
            bar.progress((i + 1) / len(batch_files))

        bar.empty()
        df = pd.DataFrame(rows)
        st.markdown("<div class='section-header'>RESULTS TABLE</div>",
                    unsafe_allow_html=True)
        st.dataframe(df, width='stretch',
                     column_config={
                         "Confidence": st.column_config.TextColumn("Confidence"),
                     })

        # Summary pie
        pred_counts = df["Prediction"].value_counts()
        fig, ax = plt.subplots(figsize=(5, 3.5), facecolor="#0d1b2a")
        ax.set_facecolor("#0d1b2a")
        colors_pie = [CLASS_HEX.get(
            next((k for k,v in CLASS_FULL.items() if v == lbl), "lung_n"), C_TEAL)
            for lbl in pred_counts.index]
        ax.pie(pred_counts.values, labels=pred_counts.index, colors=colors_pie,
               autopct="%1.0f%%", startangle=90,
               wedgeprops={"edgecolor": "#0d1b2a", "linewidth": 2},
               textprops={"color": "#e2e8f0", "fontsize": 9})
        ax.set_title("Batch Distribution", color="#94a3b8", fontsize=10,
                     fontfamily="monospace")
        c1, c2 = st.columns([1, 2])
        with c1:
            st.pyplot(fig)
            plt.close(fig)
        with c2:
            st.markdown("<div class='section-header'>BATCH SUMMARY</div>",
                        unsafe_allow_html=True)
            for lbl, cnt in pred_counts.items():
                pct = cnt / len(batch_files) * 100
                hex_ = CLASS_HEX.get(
                    next((k for k,v in CLASS_FULL.items() if v == lbl), "lung_n"), C_TEAL)
                st.markdown(f"""
                <div style='margin:8px 0'>
                    <div style='display:flex; justify-content:space-between;
                                font-size:0.85rem; margin-bottom:4px'>
                        <span style='color:{hex_}'>{lbl}</span>
                        <span style='font-family:"Space Mono",monospace;
                                     color:{hex_}'>{cnt} &nbsp;({pct:.0f}%)</span>
                    </div>
                    <div style='background:#1e3a5f; border-radius:4px; height:8px'>
                        <div style='width:{pct}%; height:100%; border-radius:4px;
                                    background:{hex_}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Download CSV
        csv = df.to_csv(index=False)
        st.download_button("⬇ Download Results CSV", csv,
                           "batch_results.csv", "text/csv",
                           width='stretch')

    elif not active:
        st.warning("No models active. Enable models in the sidebar.")

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 3 — CLASS INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_info:
    st.markdown("<div class='section-header'>CANCER CLASS REFERENCE</div>",
                unsafe_allow_html=True)

    for cls in CLASS_NAMES:
        hex_  = CLASS_HEX[cls]
        emoji = CLASS_EMOJI[cls]
        st.markdown(f"""
        <div class='glass-card' style='border-color:{hex_}33; margin-bottom:16px'>
            <div style='display:flex; align-items:center; gap:12px; margin-bottom:12px'>
                <div style='font-size:2rem'>{emoji}</div>
                <div>
                    <div style='font-size:1.15rem; font-weight:700; color:{hex_}'>
                        {CLASS_FULL[cls]}
                    </div>
                    <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
                                color:#475569; letter-spacing:2px'>
                        {cls.upper()}
                    </div>
                </div>
            </div>
            <div style='font-size:0.9rem; color:#94a3b8; line-height:1.8'>
                {CLASS_DESC[cls]}
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>ABOUT THE DATASET</div>",
                unsafe_allow_html=True)
    st.markdown("""
    <div class='glass-card'>
        <div style='font-size:0.9rem; color:#94a3b8; line-height:2'>
            <b style='color:#38bdf8'>LC25000</b> is a publicly available dataset of
            25,000 color histopathology images across 5 classes (lung and colon tissue).
            This project uses the <b style='color:#e2e8f0'>3 lung classes</b> only:<br><br>
            ✦ &nbsp;<b style='color:#ef4444'>lung_aca</b> — 5,000 adenocarcinoma slides<br>
            ✦ &nbsp;<b style='color:#22c55e'>lung_n</b> &nbsp;&nbsp;— 5,000 normal lung slides<br>
            ✦ &nbsp;<b style='color:#f97316'>lung_scc</b> — 5,000 squamous cell carcinoma slides<br><br>
            Images are 768×768 px RGB, resized to <b style='color:#e2e8f0'>224×224</b>
            for model input. The dataset is balanced across all classes.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
#  TAB 4 — PROJECT INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("<div class='section-header'>MODEL ARCHITECTURE SUMMARY</div>",
                unsafe_allow_html=True)

    arch_data = {
        "Model"     : ["Custom CNN", "VGG16", "ResNet50"],
        "Type"      : ["Scratch", "Transfer Learning", "Transfer Learning"],
        "Base"      : ["—", "VGG16 (ImageNet)", "ResNet50 (ImageNet)"],
        "LR"        : ["1e-3", "1e-4", "1e-4"],
        "Frozen"    : ["None", "Conv base", "Conv base"],
        "Expected Acc": ["88–93%", "92–96%", "94–98%"],
    }
    st.dataframe(pd.DataFrame(arch_data), width='stretch', hide_index=True)

    st.markdown("<div class='section-header'>PIPELINE OVERVIEW</div>",
                unsafe_allow_html=True)

    steps = [
        ("01", "DATA", "LC25000 lung subset · 15,000 images · 3 classes"),
        ("02", "PREPROCESS", "Resize 224×224 · Normalise /255 · Augmentation"),
        ("03", "PIPELINE", "tf.data + prefetch + parallel decode · 64 batch"),
        ("04", "TRAIN", "EarlyStopping · ReduceLROnPlateau · ModelCheckpoint"),
        ("05", "EVALUATE", "Confusion matrix · ROC-AUC · F1 · Grad-CAM"),
        ("06", "DEPLOY", "Streamlit app · Batch inference · CSV export"),
    ]
    pipe_cols = st.columns(3)
    for i, (num, title, desc) in enumerate(steps):
        with pipe_cols[i % 3]:
            st.markdown(f"""
            <div class='glass-card' style='padding:16px 18px; margin-bottom:12px'>
                <div style='font-family:"Space Mono",monospace; font-size:0.65rem;
                            color:#0ea5e9; letter-spacing:3px'>{num} · {title}</div>
                <div style='font-size:0.82rem; color:#94a3b8;
                            margin-top:6px; line-height:1.6'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>TECH STACK</div>",
                unsafe_allow_html=True)
    tech_cols = st.columns(4)
    techs = [
        ("🧠", "TensorFlow", tf.__version__),
        ("🐍", "Python",     "3.10+"),
        ("📊", "Streamlit",  "1.x"),
        ("🔬", "OpenCV",     "4.x"),
    ]
    for col, (icon, name, ver) in zip(tech_cols, techs):
        with col:
            st.markdown(f"""
            <div class='glass-card' style='text-align:center; padding:18px'>
                <div style='font-size:2rem'>{icon}</div>
                <div style='font-weight:600; color:#e2e8f0; margin-top:6px'>{name}</div>
                <div style='font-family:"Space Mono",monospace; font-size:0.72rem;
                            color:#0ea5e9; margin-top:4px'>{ver}</div>
            </div>
            """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='border-top:1px solid #1e3a5f; margin-top:40px; padding-top:20px;
            text-align:center; font-size:0.75rem; color:#334155'>
    <span style='font-family:"Space Mono",monospace; color:#0ea5e9'>LungAI</span>
    &nbsp;·&nbsp; Deep Learning Major Project
    &nbsp;·&nbsp; LC25000 Dataset
    &nbsp;·&nbsp; Built with TensorFlow + Streamlit
    <br><br>
    <span style='color:#1e3a5f'>
        ⚠ Not intended for clinical use · Research & educational purposes only
    </span>
</div>
""", unsafe_allow_html=True)
