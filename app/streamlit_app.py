"""
streamlit_app.py
================
Front-end for the AI Agent for Garbage Classification and Smart Recycling
Recommendation. Run with:

    streamlit run app/streamlit_app.py

This file contains ONLY UI/display logic. All classification and
recommendation logic lives in src/predict.py and src/recycling.py.
"""

import os
import sys
import time

import numpy as np
import streamlit as st
from PIL import Image

# Allow `from src...` imports when running via `streamlit run app/streamlit_app.py`
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src import config

st.set_page_config(
    page_title="AI Agent for Garbage Classification & Smart Recycling",
    page_icon="♻️",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Cached model loading — loaded once per server process, not per request
# ---------------------------------------------------------------------------
@st.cache_resource
def load_predictor():
    from src.predict import GarbagePredictor
    if not os.path.exists(config.BEST_MODEL_PATH):
        return None
    return GarbagePredictor(config.BEST_MODEL_PATH)


def load_gradcam(predictor):
    from src.gradcam import GradCAM
    try:
        return GradCAM(predictor.model, predictor.model_name)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("♻️ AI Agent for Garbage Classification and Smart Recycling Recommendation")
st.write(
    "Upload a garbage image and receive AI-powered classification and "
    "recycling guidance."
)

predictor = load_predictor()

if predictor is None:
    st.error(
        f"No trained model found at `{config.BEST_MODEL_PATH}`.\n\n"
        "Train the models in the Colab notebook first "
        "(`notebooks/garbage_classification_colab.ipynb`), then copy the "
        "resulting `best_model.pth` into the `models/` folder."
    )
    st.stop()

st.success(
    f"Model loaded: **{predictor.model_name}** · "
    f"{len(predictor.class_names)} classes · input size {predictor.image_size}px"
)

# ---------------------------------------------------------------------------
# Sidebar — model info (Section: Model Information)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("ℹ️ Model Information")
    st.write(f"**Model used:** {predictor.model_name}")
    st.write(f"**Number of classes:** {len(predictor.class_names)}")
    st.write(f"**Classes:** {', '.join(predictor.class_names)}")

    comparison_csv = config.MODEL_COMPARISON_CSV
    if os.path.exists(comparison_csv):
        import pandas as pd
        df = pd.read_csv(comparison_csv)
        best_row = df.iloc[0]
        st.write(f"**Test Accuracy:** {best_row['Test Accuracy']*100:.2f}%")
        st.write(f"**F1 Score:** {best_row['F1 Score']:.4f}")
        st.dataframe(df, use_container_width=True)
    else:
        st.info("Run model_comparison.csv generation in the notebook to see full metrics here.")

    st.markdown("---")
    threshold = st.slider(
        "Confidence threshold", min_value=0.0, max_value=1.0,
        value=config.CONFIDENCE_THRESHOLD, step=0.05,
        help="Predictions below this confidence will be flagged as uncertain."
    )
    show_gradcam = st.checkbox("Show Grad-CAM explanation", value=True)


# ---------------------------------------------------------------------------
# Image upload (Section: Image Upload)
# ---------------------------------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload a garbage image", type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info("👆 Upload an image (JPG, JPEG, or PNG) to get started.")
    st.stop()

# --- Error handling: invalid/corrupted/empty upload (Section 29) ---
try:
    image = Image.open(uploaded_file).convert("RGB")
except Exception as e:
    st.error(f"Could not read this image file — it may be corrupted or in an unsupported format. ({e})")
    st.stop()

col1, col2 = st.columns([1, 1.3])

with col1:
    st.subheader("📷 Uploaded Image")
    st.image(image, use_container_width=True)

# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------
with st.spinner("Classifying..."):
    result = predictor.predict_pil_image(image, confidence_threshold=threshold)

with col2:
    st.subheader("🧠 AI Prediction")

    if result["below_confidence_threshold"]:
        st.warning(result["message"])
        st.write(f"Best guess: **{result['predicted_class']}** ({result['confidence']*100:.1f}% confidence)")
    else:
        st.metric("Predicted Category", result["predicted_class"].capitalize(),
                   f"{result['confidence']*100:.1f}% confidence")

    st.write("**Top-3 Predictions:**")
    for i, pred in enumerate(result["top_predictions"], start=1):
        st.write(f"{i}. {pred['class'].capitalize()} — {pred['confidence']*100:.1f}%")
        st.progress(min(pred["confidence"], 1.0))

    st.caption(f"Inference time: {result['inference_time_ms']:.1f} ms")

# ---------------------------------------------------------------------------
# Recycling Recommendation
# ---------------------------------------------------------------------------
st.markdown("---")

if result["recycling_recommendation"] is not None:
    rec = result["recycling_recommendation"]

    st.subheader("♻️ Recycling Recommendation")

    r_col1, r_col2 = st.columns(2)

    with r_col1:
        recyclable_label = (
            "✅ Recyclable" if rec["recyclable"] is True
            else "🚫 Not typically recyclable" if rec["recyclable"] is False
            else "❓ Unknown — check local guidelines"
        )
        st.write(f"**Recyclability:** {recyclable_label}")

        st.write("**Disposal Method:**")
        st.write(rec["disposal_method"])

        if rec["instructions"]:
            st.write("**Instructions:**")
            for step in rec["instructions"]:
                st.write(f"- {step}")

    with r_col2:
        if rec["reuse_ideas"]:
            st.write("**♻️ Reuse Ideas:**")
            for idea in rec["reuse_ideas"]:
                st.write(f"- {idea}")

        st.write("**🌍 Environmental Impact:**")
        st.write(rec["environmental_impact"])

        st.write("**⚠️ Safety:**")
        st.write(rec["safety"])
else:
    st.info("No recycling recommendation shown — classification confidence was below the threshold.")

# ---------------------------------------------------------------------------
# Grad-CAM explainability (Section 30)
# ---------------------------------------------------------------------------
if show_gradcam and not result["below_confidence_threshold"]:
    st.markdown("---")
    st.subheader("🔍 Model Explainability (Grad-CAM)")
    st.caption(
        "Grad-CAM highlights the image regions that most influenced this "
        "prediction. It is an approximate visual explanation, not proof of "
        "the model's exact reasoning."
    )

    try:
        import torch
        from src.gradcam import GradCAM, overlay_heatmap

        gradcam = load_gradcam(predictor)
        if gradcam is not None:
            tensor = predictor.transform(image).unsqueeze(0)
            class_idx = predictor.class_names.index(result["predicted_class"])
            cam, _ = gradcam.generate(tensor, class_idx=class_idx)

            resized_original = image.resize((predictor.image_size, predictor.image_size))
            overlay = overlay_heatmap(np.array(resized_original), cam)

            g_col1, g_col2 = st.columns(2)
            with g_col1:
                st.image(resized_original, caption="Original", use_container_width=True)
            with g_col2:
                st.image(overlay, caption="Grad-CAM Heatmap", use_container_width=True)
        else:
            st.info("Grad-CAM is not available for this model architecture.")
    except Exception as e:
        st.info(f"Grad-CAM could not be generated for this image. ({e})")

st.markdown("---")
st.caption(
    "AI Agent for Garbage Classification and Smart Recycling Recommendation · "
    "Built with PyTorch and Streamlit."
)
