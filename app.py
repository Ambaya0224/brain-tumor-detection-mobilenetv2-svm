import streamlit as st
import numpy as np
import joblib
from PIL import Image
import tensorflow as tf

# Load models
feature_extractor = tf.keras.models.load_model("feature_extractor.keras")
svm_model = joblib.load("svm_model.pkl")

# Load and reverse label map
label_map = np.load("label_map.npy", allow_pickle=True).item()
inv_label_map = {v: k for k, v in label_map.items()}

IMG_SIZE = (180, 180)

st.title("Brain Tumor Detection (MobileNetV2 + SVM)")
st.write("Upload an MRI image to classify tumor type.")

uploaded_file = st.file_uploader("Choose MRI Image", type=['jpg', 'jpeg', 'png'])

if uploaded_file:
    img = Image.open(uploaded_file).convert("RGB")
    st.image(img, caption="Uploaded MRI", width=300)

    # Preprocess
    img_resized = img.resize(IMG_SIZE)
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Extract features
    features = feature_extractor.predict(img_array)

    # Predict using SVM
    pred = svm_model.predict(features)[0]

    # Get label name from reversed map
    class_name = inv_label_map[pred]

    st.subheader("Prediction:")
    st.write(f"**{class_name}**")

    st.success("Classification completed successfully!")