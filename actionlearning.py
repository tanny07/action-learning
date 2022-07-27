import streamlit as st
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import pandas as pd
import urllib
from config import config
import requests
import json
import matplotlib.pyplot as plt

def get_explanations(file):
    files = {'upload_file': file}
    res = requests.post(config['API']['URL']+'/explain', files=files)
    data = []
    if res.status_code == 200:
        data = res.json()
    return data

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




# component to upload images
uploaded_file = st.sidebar.file_uploader(
    "Choose an image to classify", type=["jpg", "jpeg", "png"]
)


if uploaded_file:
    pred = get_explanations(uploaded_file.getvalue())
    val = json.loads(pred)
    heatmap = np.array(val['heatmap_lime'])
    heatmap_custom = np.array(val['cie_inspiriation'])
    fig, ax = plt.subplots(2,1)
    ax[0].imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
    # ax[0].colorbar()
    ax[1].imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap_custom.max(), vmax = heatmap_custom.max())
    # ax[1].colorbar()
    st.write(fig)
    # bytes_data = uploaded_file.read()
    # inputShape = (224, 224)
    # preprocess = imagenet_utils.preprocess_input
    # if network in ("Inception"):
    #     inputShape = (299, 299)
    #     preprocess = preprocess_input
    # Network = MODELS[network]
    # model = Network(weights="imagenet")

    # image = Image.open(BytesIO(bytes_data))
    # image = image.convert("RGB")
    # image = image.resize(inputShape)
    # image = img_to_array(image)
    # image = np.expand_dims(image, axis=0)
    # image = preprocess(image)

    # preds = model.predict(image)
    # predictions = imagenet_utils.decode_predictions(preds)
    # imagenetID, label, prob = predictions[0][0]

    # st.image(bytes_data, caption=[f"{label} {prob*100:.2f}"])
    # st.subheader(f"Top Predictions from {network}")
    # st.dataframe(
    #     pd.DataFrame(
    #         predictions[0], columns=["Network", "Classification", "Confidence"]
    #     )
    # )

