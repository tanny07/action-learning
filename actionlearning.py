import streamlit as st
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib

# set page layout
st.set_page_config(
    page_title="Explainable AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Image Classification")
st.sidebar.subheader("Input")
models_list = ["VGG16", "Inception"]
network = st.sidebar.selectbox("Select the Model", models_list)


MODELS = {
    "VGG16": VGG16,
    "Inception": InceptionV3,
}

# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)


if uploaded_file:
    bytes_data = uploaded_file.read()
    inputShape = (224, 224)
    preprocess = imagenet_utils.preprocess_input
    if network in ("Inception"):
        inputShape = (299, 299)
        preprocess = preprocess_input
    Network = MODELS[network]
    model = Network(weights="imagenet")

    image = Image.open(BytesIO(bytes_data))
    image = image.convert("RGB")
    image = image.resize(inputShape)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess(image)

    preds = model.predict(image)
    predictions = imagenet_utils.decode_predictions(preds)
    imagenetID, label, prob = predictions[0][0]

    st.image(bytes_data, caption=[f"{label} {prob*100:.2f}"])
    st.subheader(f"Top Predictions from {network}")
    st.dataframe(
        pd.DataFrame(
            predictions[0], columns=["Network", "Classification", "Confidence"]
        )
    )

