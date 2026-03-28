from __future__ import annotations

import os

import numpy as np
import streamlit as st
from PIL import Image

from src.pipeline.run_pipeline import run_pipeline

DEFAULT_CHECKPOINT = "models/noise_classifier_best.pt"
LABEL_PRETTY = {
    "gaussian": "Gaussian",
    "salt_pepper": "Salt & Pepper",
    "speckle": "Speckle",
    "periodic": "Periodic",
}
METHOD_PRETTY = {
    "gaussian": "Non-local Means",
    "salt_pepper": "Median Filter",
    "speckle": "Bilateral Filter",
    "periodic": "FFT Notch Filter",
}


st.set_page_config(page_title="AI Noise Classifier", layout="centered")


def pretty_label(label: str) -> str:
    return LABEL_PRETTY.get(label, label.replace("_", " ").title())


def pretty_method(label: str) -> str:
    return METHOD_PRETTY.get(label, "Adaptive Denoising")


def ensure_uint8_for_display(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        if np.issubdtype(arr.dtype, np.floating) and arr.size and float(arr.max()) <= 1.0 and float(arr.min()) >= 0.0:
            arr = arr * 255.0
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[:, :, 0]
    return arr


def load_image_to_np(uploaded_file) -> tuple[Image.Image, np.ndarray]:
    """Load uploaded image as PIL and RGB ndarray."""
    pil_img = Image.open(uploaded_file).convert("RGB")
    rgb = np.array(pil_img)
    return pil_img, rgb


def render_styles() -> None:
    st.markdown(
        """
        <style>
        .stApp {
            font-family: "Inter", "Segoe UI", sans-serif;
            background: #0B1220;
            color: #E5E7EB;
        }
        section.main, header, footer {
            background: #0B1220 !important;
        }
        footer { visibility: hidden; }
        .block-container {
            padding-top: 2.4rem;
            padding-bottom: 1.5rem;
            max-width: 920px;
        }
        .ui-muted { color: #9CA3AF; font-size: 0.95rem; }
        .ui-card {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem 1rem 0.85rem 1rem;
            background: rgba(17,24,39,0.65);
            color: #E5E7EB;
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
        }
        .ui-badge {
            display: inline-block;
            border: 1px solid rgba(59,130,246,0.25);
            border-radius: 999px;
            font-size: 0.75rem;
            padding: 0.18rem 0.55rem;
            color: #93C5FD;
            margin-bottom: 0.55rem;
            background: rgba(59,130,246,0.12);
        }
        .ui-title {
            font-size: 1.65rem;
            font-weight: 700;
            margin: 0.25rem 0 0.2rem 0;
            color: #E5E7EB;
        }
        .prediction-label {
            font-size: 28px;
            font-weight: 700;
            color: #F9FAFB;
            margin: 0.2rem 0 0.35rem 0;
        }
        .stCaption {
            color: #A7B0BE !important;
            font-weight: 500 !important;
        }
        .stProgress > div > div > div > div {
            background-color: #3B82F6;
        }
        .stProgress > div > div {
            background-color: rgba(255,255,255,0.1);
        }
        ::selection {
            background: #3B82F6;
            color: #0B1220;
        }
        ::-moz-selection {
            background: #3B82F6;
            color: #0B1220;
        }
        .ui-empty {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 16px;
            padding: 1rem;
            color: #9CA3AF;
            background: rgba(17,24,39,0.65);
            box-shadow: 0 10px 28px rgba(0, 0, 0, 0.28);
        }
        .stMarkdown, .stMarkdown p, .stMarkdown div, label {
            color: #E5E7EB !important;
        }
        section[data-testid="stSidebar"] {
            background: #0F172A !important;
            border-right: 1px solid rgba(255,255,255,0.08);
        }
        div[data-testid="stFileUploader"] {
            background: rgba(17,24,39,0.65) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 16px !important;
            padding: 0.4rem !important;
        }
        div[data-testid="stFileUploader"] section {
            background: transparent !important;
            border: 1px dashed rgba(59,130,246,0.35) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stFileUploader"] button {
            background: #111827 !important;
            color: #E5E7EB !important;
            border: 1px solid rgba(255,255,255,0.12) !important;
        }
        div[data-testid="stFileUploader"] button:hover {
            background: #3B82F6 !important;
            color: #0B1220 !important;
            border-color: rgba(59,130,246,0.8) !important;
        }
        div[data-testid="stAlert"] {
            background: rgba(17,24,39,0.85) !important;
            color: #E5E7EB !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stExpander"] {
            background: rgba(17,24,39,0.65) !important;
            border: 1px solid rgba(255,255,255,0.08) !important;
            border-radius: 12px !important;
        }
        div[data-testid="stImage"] img {
            border: 1px solid rgba(255,255,255,0.08);
            border-radius: 12px;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


render_styles()
st.markdown('<div class="ui-title">AI Driven Noise Reduction</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="ui-muted">Upload an image to classify noise type and run adaptive denoising.</div>',
    unsafe_allow_html=True,
)

# Sidebar
st.sidebar.header("Settings")
default_checkpoint = os.getenv("NOISE_CLASSIFIER_CHECKPOINT", DEFAULT_CHECKPOINT)
checkpoint_path = st.sidebar.text_input("Checkpoint path", value=default_checkpoint)
device = st.sidebar.selectbox("Device", options=["cpu", "cuda"], index=0)
st.sidebar.caption("Model yüklenemezse sistem otomatik olarak heuristic fallback kullanır.")

left_col, right_col = st.columns(2, gap="large")
uploaded = left_col.file_uploader("Upload image", type=["png", "jpg", "jpeg", "bmp"])

if uploaded is None:
    left_col.markdown('<div class="ui-empty">No image uploaded yet. Please select an image to continue.</div>', unsafe_allow_html=True)
    right_col.markdown('<div class="ui-empty">Denoised output will appear here after upload.</div>', unsafe_allow_html=True)
    st.stop()

_, rgb_img = load_image_to_np(uploaded)
left_col.markdown("##### Uploaded Image")
left_col.image(ensure_uint8_for_display(rgb_img), use_container_width=True)

predicted_label = "gaussian"
denoised_image = rgb_img

with st.spinner("Detecting noise and applying denoising..."):
    try:
        result = run_pipeline(
            rgb_img,
            checkpoint_path=checkpoint_path.strip() or None,
            device=device,
        )
        predicted_label = result.predicted_noise
        denoised_image = result.denoised_image
    except Exception as exc:
        st.error(f"Pipeline failed: {exc}")

right_col.markdown("##### Denoised Output")
right_col.image(ensure_uint8_for_display(denoised_image), use_container_width=True)

st.markdown(
    (
        '<div class="ui-card"><span class="ui-badge">Prediction</span>'
        f'<div class="prediction-label">{pretty_label(predicted_label)}</div>'
        f'<div class="ui-muted">Denoising method: {pretty_method(predicted_label)}</div>'
        "</div>"
    ),
    unsafe_allow_html=True,
)
