import streamlit as st
import numpy as np
import joblib
import io
import os
import gdown
import tensorflow as tf
from PIL import Image
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="ASD Emotion Recognition",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500&display=swap');
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0f1117;
    color: #e8e8e8;
}
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
.title-block { text-align: center; padding: 2rem 0 1rem 0; }
.title-block h1 { font-size: 2.4rem; color: #f0f0f0; }
.title-block p  { color: #888; font-size: 0.95rem; font-weight: 300; }
.upload-label {
    font-size: 0.78rem; color: #888;
    letter-spacing: 0.1em; text-transform: uppercase; margin-bottom: 0.4rem;
}
.result-card {
    background: linear-gradient(135deg, #1a1d27, #1e2235);
    border: 1px solid #2e3248; border-radius: 16px;
    padding: 2rem; text-align: center; margin-top: 1.5rem;
}
.emotion-label { font-family: 'DM Serif Display', serif; font-size: 3rem; margin: 0.5rem 0; }
.confidence-text { color: #888; font-size: 0.9rem; margin-bottom: 1.5rem; }
.bar-row { display: flex; align-items: center; margin: 6px 0; gap: 10px; }
.bar-label { width: 80px; text-align: right; font-size: 0.82rem; color: #aaa; flex-shrink: 0; }
.bar-track { flex: 1; background: #2a2d3e; border-radius: 999px; height: 8px; overflow: hidden; }
.bar-fill  { height: 100%; border-radius: 999px; background: linear-gradient(90deg, #4f6ef7, #a78bfa); }
.bar-pct   { width: 40px; font-size: 0.78rem; color: #666; }
.stButton > button {
    background: linear-gradient(135deg, #4f6ef7, #7c3aed);
    color: white; border: none; border-radius: 10px;
    padding: 0.65rem 2.5rem; font-size: 1rem;
    font-family: 'DM Sans', sans-serif; font-weight: 500;
    width: 100%; cursor: pointer;
}
.stButton > button:hover { opacity: 0.88; }
.divider { border: none; border-top: 1px solid #2e3248; margin: 1.5rem 0; }
.info-pill {
    display: inline-block; background: #1e2235;
    border: 1px solid #2e3248; border-radius: 999px;
    padding: 0.25rem 0.9rem; font-size: 0.78rem; color: #888; margin: 0.2rem;
}
#MainMenu {visibility: hidden;} footer {visibility: hidden;} header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
EMOTION_EMOJI = {
    'anger':    '😠',
    'Natural':  '😐',
    'joy':      '😄',
    'sadness':  '😢',
    'surprise': '😲',
    'fear':     '😨',
}
EMOTION_COLOR = {
    'anger':    '#ef4444',
    'Natural':  '#94a3b8',
    'joy':      '#f59e0b',
    'sadness':  '#60a5fa',
    'surprise': '#a78bfa',
    'fear':     '#34d399',
}

# ─────────────────────────────────────────────
# DOWNLOAD MODELS FROM GOOGLE DRIVE
# Replace each YOUR_XXXXX_ID with your actual Google Drive file IDs
# ─────────────────────────────────────────────
MODEL_FILES = {
    "encoder_model.h5":    "https://drive.google.com/uc?id=1YR7c3Gb2-7E3MTWvhpPRnzO05J7kVR-1",
    "classifier_model.h5": "https://drive.google.com/uc?id=1AHt13E9l9Lh8Fwu3-LkchZB2PNCxbcmz",
    "scaler.pkl":           "https://drive.google.com/uc?id=1wfZCAjTHSLFICFD-choQ3q2Qzpoi360q",
    "label_encoder.pkl":    "https://drive.google.com/uc?id=1IddPZVNtqDG3FGBcxBqMWrl4WqSiIrcw",
}

def download_models():
    for filename, url in MODEL_FILES.items():
        if not os.path.exists(filename):
            with st.spinner(f"Downloading {filename}..."):
                gdown.download(url, filename, quiet=False)

download_models()

# ─────────────────────────────────────────────
# LOAD MODELS — cached so they load only once
# ─────────────────────────────────────────────
@st.cache_resource
def load_vgg():
    base = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
    # GlobalAveragePooling2D — must match how you trained
    out = tf.keras.layers.GlobalAveragePooling2D()(base.output)
    m   = Model(inputs=base.input, outputs=out)
    m.trainable = False
    return m

@st.cache_resource
def load_pipeline():
    encoder = load_model("encoder_model.h5")    # encoder only, not full autoencoder
    clf     = load_model("classifier_model.h5")
    scaler  = joblib.load("scaler.pkl")
    le      = joblib.load("label_encoder.pkl")  # fixed filename
    return encoder, clf, scaler, le

# ─────────────────────────────────────────────
# FEATURE EXTRACTION — uses PIL, no cv2 needed
# ─────────────────────────────────────────────
def extract_features(uploaded_file, vgg_model):
    uploaded_file.seek(0)                              # reset file pointer before reading
    img = Image.open(io.BytesIO(uploaded_file.read())).convert("RGB")
    img = img.resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)                        # VGG19 standard preprocessing
    features = vgg_model.predict(arr, verbose=0)
    return features.flatten()

# ─────────────────────────────────────────────
# UI — Title
# ─────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>🧠 ASD Emotion Recognition</h1>
    <p>Upload a hand gesture image and a facial expression image to detect emotion</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FILE UPLOADERS
# ─────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="upload-label">✋ Hand Gesture</div>', unsafe_allow_html=True)
    hand_file = st.file_uploader("", type=["jpg","jpeg","png"], key="hand",
                                  label_visibility="collapsed")
    if hand_file:
        st.image(hand_file, use_container_width=True, caption="Hand image")

with col2:
    st.markdown('<div class="upload-label">😊 Facial Expression</div>', unsafe_allow_html=True)
    face_file = st.file_uploader("", type=["jpg","jpeg","png"], key="face",
                                  label_visibility="collapsed")
    if face_file:
        st.image(face_file, use_container_width=True, caption="Face image")

st.markdown('<hr class="divider">', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PREDICTION
# ─────────────────────────────────────────────
if hand_file and face_file:
    if st.button("🔍 Predict Emotion"):
        with st.spinner("Analysing..."):
            try:
                vgg                      = load_vgg()
                encoder, clf, scaler, le = load_pipeline()

                hand_feat = extract_features(hand_file, vgg)
                face_feat = extract_features(face_file, vgg)

                # Fusion order: [hand, face] — must match training
                fused  = np.concatenate([hand_feat, face_feat]).reshape(1, -1)
                scaled = scaler.transform(fused)

                # Use encoder.predict(), NOT autoencoder.predict()
                encoded = encoder.predict(scaled, verbose=0)
                probs   = clf.predict(encoded, verbose=0)[0]

                pred_idx   = int(np.argmax(probs))
                pred_label = le.inverse_transform([pred_idx])[0]
                confidence = float(probs[pred_idx])
                all_probs  = {cls: float(p) for cls, p in zip(le.classes_, probs)}

                emoji = EMOTION_EMOJI.get(pred_label, '🎭')
                color = EMOTION_COLOR.get(pred_label, '#4f6ef7')

                st.markdown(f"""
                <div class="result-card">
                    <div style="font-size:1rem;color:#888;letter-spacing:0.1em;text-transform:uppercase;">
                        Detected Emotion
                    </div>
                    <div class="emotion-label" style="color:{color};">
                        {emoji} {pred_label.capitalize()}
                    </div>
                    <div class="confidence-text">{confidence*100:.1f}% confidence</div>
                """, unsafe_allow_html=True)

                for cls, prob in sorted(all_probs.items(), key=lambda x: x[1], reverse=True):
                    pct = prob * 100
                    st.markdown(f"""
                    <div class="bar-row">
                        <div class="bar-label">{cls}</div>
                        <div class="bar-track">
                            <div class="bar-fill" style="width:{pct:.1f}%;"></div>
                        </div>
                        <div class="bar-pct">{pct:.1f}%</div>
                    </div>
                    """, unsafe_allow_html=True)

                st.markdown("</div>", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Prediction failed: {e}")
else:
    st.markdown("""
    <div style="text-align:center; color:#555; padding:1rem;">
        Upload both images above to enable prediction
    </div>
    """, unsafe_allow_html=True)

st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<div style="text-align:center;">
    <span class="info-pill">VGG19 Feature Extraction</span>
    <span class="info-pill">Sparse Autoencoder</span>
    <span class="info-pill">SpAuCNN Classifier</span>
    <span class="info-pill">6 Emotion Classes</span>
</div>
""", unsafe_allow_html=True)