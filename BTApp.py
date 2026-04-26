# ==============================
# IMPORTS
# ==============================
import tensorflow as tf
from PIL import Image, ImageOps
import streamlit as st
import numpy as np
import os
import requests

# ==============================
# PAGE CONFIG
# ==============================
st.set_page_config(page_title="Brain Tumor Detection", layout="centered")

st.title("🧠 Brain Tumor Classification")
st.header("CNN-based MRI Analysis")
st.write("Upload a brain MRI scan to detect Tumor or Healthy")

# ==============================
# DOWNLOAD MODEL FROM DRIVE
# ==============================
MODEL_URL = "https://drive.google.com/uc?id=1usdJ6gRoWjyKXgiuP8g7XcnWcNmpWq4o"
MODEL_PATH = "BrainTumor.h5"

@st.cache_resource
def load_model_from_drive():
    if not os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, "wb") as f:
            f.write(requests.get(MODEL_URL).content)
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model_from_drive()

# ==============================
# PREDICTION FUNCTION
# ==============================
def import_and_classify(img):
    data = np.ndarray(shape=(1, 224, 224, 3), dtype=np.float32)

    size = (224, 224)
    image = ImageOps.fit(img, size, Image.Resampling.LANCZOS)
    image = image.convert("RGB")   # 🔥 Fix grayscale issue

    image_array = np.asarray(image)
    normalized_image_array = (image_array.astype(np.float32) / 127.0) - 1

    data[0] = normalized_image_array

    prediction = model.predict(data)
    return np.argmax(prediction)

# ==============================
# FILE UPLOAD
# ==============================
uploaded_file = st.file_uploader("📤 Upload MRI Image", type=("jpg","png","jpeg"))

st.write("Accepted formats: JPG, PNG, JPEG")

# ==============================
# SAMPLE IMAGE (SAFE)
# ==============================
if st.checkbox("Show sample MRI"):
    sample_path = "Cmd Test Images/Yes1.JPG"
    if os.path.exists(sample_path):
        Image_1 = Image.open(sample_path)
        st.image(Image_1, width=300, caption="Sample MRI scan")
    else:
        st.warning("Sample image not found")

# ==============================
# PREDICTION BUTTON
# ==============================
if st.button("🔍 Check Results"):
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, width=400, caption="Uploaded MRI")

        st.write("Analyzing...")

        label = import_and_classify(image)

        if label == 1:
            st.error("🛑 Brain Tumor Detected")
        else:
            st.success("✅ Healthy Brain")

    else:
        st.warning("Please upload an image first")
