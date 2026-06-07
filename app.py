import streamlit as st
import tensorflow as tf
import numpy as np
import json
import os
from PIL import Image

# ─── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="🌿 Plant Disease Predictor",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #2e7d32;
        text-align: center;
        margin-bottom: 0.2rem;
    }
    .subtitle {
        text-align: center;
        color: #555;
        font-size: 1rem;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f1f8e9;
        border-left: 5px solid #66bb6a;
        border-radius: 8px;
        padding: 1.2rem 1.5rem;
        margin-top: 1.5rem;
    }
    .healthy-box {
        background: #e8f5e9;
        border-left: 5px solid #2e7d32;
    }
    .disease-box {
        background: #fff8e1;
        border-left: 5px solid #f9a825;
    }
    .confidence-bar-label {
        font-size: 0.85rem;
        color: #444;
    }
    .footer {
        text-align: center;
        color: #aaa;
        font-size: 0.8rem;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─── Constants ─────────────────────────────────────────────────────────────────
MODEL_PATH        = "plant_disease_prediction_model.h5"
CLASS_INDICES_PATH = "class_indices.json"
IMG_SIZE          = 224

# ─── Load Model & Class Indices ────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None

    # ── Attempt 1: standard Keras load ────────────────────────────────────────
    try:
        return tf.keras.models.load_model(MODEL_PATH)
    except Exception:
        pass  # falls through on Keras version mismatch (batch_shape / optional)

    # ── Attempt 2: rebuild exact architecture + load weights only ─────────────
    #
    # TF 2.15 Keras cannot deserialise InputLayer configs saved by older Keras
    # versions ("Unrecognized keyword arguments: ['batch_shape', 'optional']").
    # The fix: reconstruct the CNN from scratch (matches the training notebook)
    # and call load_weights(), which reads tensors directly without touching the
    # saved config at all.
    #
    num_classes = 38  # class_indices.json has keys 0-37
    if os.path.exists(CLASS_INDICES_PATH):
        with open(CLASS_INDICES_PATH, "r") as _f:
            num_classes = len(json.load(_f))

    mdl = tf.keras.Sequential()
    mdl.add(tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu",
        input_shape=(IMG_SIZE, IMG_SIZE, 3)
    ))
    mdl.add(tf.keras.layers.MaxPooling2D(2, 2))
    mdl.add(tf.keras.layers.Conv2D(64, (3, 3), activation="relu"))
    mdl.add(tf.keras.layers.MaxPooling2D(2, 2))
    mdl.add(tf.keras.layers.Flatten())
    mdl.add(tf.keras.layers.Dense(256, activation="relu"))
    mdl.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    # load_weights() reads tensor values by position — version-agnostic
    mdl.load_weights(MODEL_PATH)
    return mdl

@st.cache_data(show_spinner=False)
def load_class_indices():
    if not os.path.exists(CLASS_INDICES_PATH):
        return {}
    with open(CLASS_INDICES_PATH, "r") as f:
        return json.load(f)

model         = load_model()
class_indices = load_class_indices()

# ─── Helper Functions ──────────────────────────────────────────────────────────
def preprocess_image(img: Image.Image) -> np.ndarray:
    img = img.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.array(img, dtype="float32") / 255.0
    return np.expand_dims(arr, axis=0)

def predict(img: Image.Image):
    arr   = preprocess_image(img)
    probs = model.predict(arr, verbose=0)[0]
    idx   = int(np.argmax(probs))
    label = class_indices.get(str(idx), "Unknown")
    conf  = float(probs[idx])
    return label, conf, probs

def format_label(raw: str) -> tuple[str, str]:
    """Return (plant_name, condition) from 'Plant___Condition' format."""
    parts = raw.replace("___", "|").replace("_", " ").split("|")
    if len(parts) == 2:
        return parts[0].strip(), parts[1].strip()
    return raw, ""

def top_k_predictions(probs, k=5):
    indices = np.argsort(probs)[::-1][:k]
    return [(class_indices.get(str(i), "Unknown"), float(probs[i])) for i in indices]

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/color/96/leaf.png", width=64)
    st.markdown("### ℹ️ About")
    st.markdown(
        "This app uses a **CNN trained on the PlantVillage dataset** to detect "
        "plant diseases from leaf images.\n\n"
        "**Supported plants include:** Apple, Grape, Corn, Tomato, Potato, "
        "Peach, Pepper, Strawberry, Squash, and more."
    )
    st.markdown("---")
    st.markdown("**How to use:**")
    st.markdown("1. Upload a clear leaf photo\n2. Click **Predict**\n3. View results & confidence")
    st.markdown("---")
    st.markdown("**Model Input:** 224×224 RGB image")
    if class_indices:
        st.markdown(f"**Classes:** {len(class_indices)} disease categories")

# ─── Main UI ───────────────────────────────────────────────────────────────────
st.markdown('<div class="main-title">🌿 Plant Disease Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload a leaf image to instantly diagnose plant health</div>', unsafe_allow_html=True)

# Model / class-indices warnings
if model is None:
    st.error(
        "⚠️ **Model file not found.**  \n"
        f"Place `{MODEL_PATH}` in the same directory as `app.py` and restart."
    )
    st.stop()

if not class_indices:
    st.warning(
        f"⚠️ `{CLASS_INDICES_PATH}` not found. Predictions will show raw index numbers."
    )

# ─── Upload Section ────────────────────────────────────────────────────────────
uploaded = st.file_uploader(
    "Choose a leaf image (JPG / PNG / JPEG)",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear, well-lit image of a single leaf for best results.",
)

if uploaded:
    img = Image.open(uploaded)

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        st.image(img, caption="Uploaded image", use_container_width=True)

    with col2:
        st.markdown("#### Image Details")
        st.markdown(f"- **Filename:** `{uploaded.name}`")
        st.markdown(f"- **Size:** {img.size[0]} × {img.size[1]} px")
        st.markdown(f"- **Mode:** {img.mode}")
        st.markdown(f"- **File size:** {uploaded.size / 1024:.1f} KB")

    st.divider()

    # ─── Predict Button ────────────────────────────────────────────────────────
    if st.button("🔍  Predict Disease", use_container_width=True, type="primary"):
        with st.spinner("Analysing leaf…"):
            label, conf, probs = predict(img)

        plant, condition = format_label(label)
        is_healthy = "healthy" in condition.lower()

        box_cls = "healthy-box" if is_healthy else "disease-box"
        icon    = "✅" if is_healthy else "🔬"
        status  = "Healthy" if is_healthy else "Disease Detected"

        st.markdown(f"""
        <div class="result-box {box_cls}">
            <h3 style="margin:0 0 0.4rem 0;">{icon} {status}</h3>
            <p style="margin:0;font-size:1.1rem;">
                <strong>Plant:</strong> {plant}<br>
                <strong>Condition:</strong> {condition}
            </p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("#### Confidence")
        st.progress(conf, text=f"{conf * 100:.1f}%")

        # Top-5 predictions chart
        st.markdown("#### Top 5 Predictions")
        top5 = top_k_predictions(probs, k=min(5, len(class_indices)))

        chart_data = {
            "Class": [format_label(lbl)[1] or lbl for lbl, _ in top5],
            "Confidence (%)": [round(c * 100, 2) for _, c in top5],
        }

        import pandas as pd
        df = pd.DataFrame(chart_data)
        st.bar_chart(df.set_index("Class"))

        # Advice section
        st.markdown("#### 💡 Recommendations")
        if is_healthy:
            st.success(
                "Your plant appears healthy! Keep up good watering habits, "
                "ensure adequate sunlight, and inspect leaves regularly."
            )
        else:
            st.warning(
                f"**{condition}** detected on **{plant}**.  \n"
                "Consider consulting an agronomist. Common treatments include "
                "targeted fungicides/pesticides, removing infected leaves, "
                "improving air circulation, and reducing leaf wetness."
            )

# ─── Placeholder when no image uploaded ───────────────────────────────────────
else:
    st.info("👆 Upload a leaf image above to get started.")
    with st.expander("📷 Example diseases this model can detect"):
        examples = [
            "Apple – Cedar Apple Rust", "Apple – Black Rot", "Apple – Scab",
            "Grape – Black Rot", "Grape – Leaf Blight", "Grape – Powdery Mildew",
            "Corn – Common Rust", "Corn – Northern Leaf Blight",
            "Tomato – Late Blight", "Tomato – Bacterial Spot",
            "Potato – Early Blight", "Potato – Late Blight",
        ]
        st.markdown("\n".join(f"- {e}" for e in examples))

# ─── Footer ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Built with TensorFlow · Streamlit · PlantVillage Dataset</div>',
    unsafe_allow_html=True,
)