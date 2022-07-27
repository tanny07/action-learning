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
from streamlit_option_menu import option_menu

# ---- SETUP ----
def get_explanations(file):
    files = {'upload_file': file}
    res = requests.post(config['API']['URL']+'/explain', files=files)
    data = []
    if res.status_code == 200:
        data = res.json()
    return data

# ---- set page layout ----
st.set_page_config(
    page_title="Explainable AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Image Classification with Explainers ")

# ---- Navigation bar ---- 
selected = option_menu(None, ["Home", "Benchmark", "Creations"], 
    icons=['house', 'cast', 'command'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
selected

# ---- Home Page Function ----
if selected == 'Home':
    st.write("""
    # ABOUT THE ACTION LEARNING PROJECT 
    ## We have created this website using Streamlit and FastApi
    ## In the Benchmark tab you can use 2 pre-trained models which are VGG16 and Inception
    ## In the Creation tab we have created our own explainers to eplain how the model in classifying the image 
    ## In 'Benchmark' and 'Creation' tab, the results include classification score and explainer's heatmap  
    ## Professors : Prof. Bill Manos and Prof. Alaa Bhakti
    ## Contributors : Rahul Jaikishen, Utsav Pandey and Tanmay MONDKAR
    """)



# ---- Benchmark Function ----
def benchmark():

    st.sidebar.subheader("Upload an Image file")
    models_list = ["VGG16", "Inception"]
    network = st.sidebar.selectbox("Select the Model", models_list)
    uploaded_file = st.sidebar.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    check_benchmark = st.sidebar.checkbox("Submit")

    if check_benchmark:
        bytes_data = uploaded_file.read()
        st.image(bytes_data)
        pred = get_explanations(uploaded_file.getvalue())
        val = json.loads(pred)
        st.subheader(f"Top Predictions from {network}")
        st.write(val['score'],val['prediction'])
        heatmap = np.array(val['heatmap_lime'])
        heatmap_custom = np.array(val['cie_inspiriation'])
        width = st.sidebar.slider("plot width", 1, 20, 3)
        height = st.sidebar.slider("plot height", 1, 20, 1)
        fig, ax = plt.subplots(figsize=(width ,height))
        plt.title("Lime Explainer Heatmap", fontsize = 20)
        ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap.max(), vmax = heatmap.max())
        ax.axis('off')
        st.write(fig)

# ---- Own Created Explainer Function ----
def own_creation():

    st.sidebar.subheader("Upload an Image file")
    models_list = ["VGG16", "Inception"]
    network = st.sidebar.selectbox("Select the Model", models_list)
    uploaded_file = st.sidebar.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    check = st.sidebar.checkbox("Submit")

    if check:
        bytes_data = uploaded_file.read()
        st.image(bytes_data)
        pred = get_explanations(uploaded_file.getvalue())
        val = json.loads(pred)
        st.subheader(f"Top Predictions from {network}")
        st.write(val['score'],val['prediction'])
        heatmap = np.array(val['heatmap_lime'])
        heatmap_custom = np.array(val['cie_inspiriation'])
        width = st.sidebar.slider("plot width", 1, 20, 3)
        height = st.sidebar.slider("plot height", 1, 20, 1)
        fig, ax = plt.subplots(figsize=(width ,height))
        plt.title("Custom Created Explainer", fontsize = 20)
        ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap_custom.max(), vmax = heatmap_custom.max())
        ax.axis('off')
        st.write(fig)

# Navigating to pages
if selected == 'Benchmark':
    benchmark()

if selected == 'Creations':
    own_creation()