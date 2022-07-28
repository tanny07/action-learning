from imp import reload
from optparse import check_builtin
from turtle import width
from typing_extensions import Self
from sqlalchemy import column
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
    page_title="Explainable AI in Deep Learning Models for Computer Vision",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title("Explainable AI in Deep Learning Models for Computer Vision ")

# ---- Navigation bar ---- 
selected = option_menu(None, ["Home", "Benchmark", "Creations"], 
    icons=['house', 'cast', 'command'], 
    menu_icon="cast", default_index=0, orientation="horizontal")
#selected

# ---- Home Page Function ----
if selected == 'Home':
    st.write("""
    # ABOUT THE ACTION LEARNING PROJECT 
    ## We have created this website using Streamlit and FastApi
    ## In the Benchmark tab you can use 2 pre-trained models which are VGG16 and Inception
    ## In the Creation tab we have created our own explainers to eplain how the model in classifying the image 
    ## In 'Benchmark' and 'Creation' tab, the results include classification score and explainer's heatmap  
    ## Professors : Prof. Bill MANOS and Prof. Alaa BHAKTI
    ## Contributors : Rahul JAIKISHEN, Utsav PANDEY and Tanmay MONDKAR
    """)



# ---- Benchmark Function ----
def benchmark():
    st.subheader("Upload an Image file")
    col1, col2 = st.columns(2)
    with col1 :         
        models_list = ["VGG16", "Inception"]
        network = st.selectbox("Select the Model", models_list)
    with col2 :
        uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)
 
    with col1: clear_benchmark = st.button("Clear Screen")
    with col2: check_benchmark = st.button("Submit")
    
    if check_benchmark:

        pred = get_explanations(uploaded_file.getvalue())
        val = json.loads(pred)
        st.subheader(f"Top Classification from {network}")
        st.write(val['score'],val['prediction'], column=('score','classification'))

        bytes_data = uploaded_file.read()
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(bytes_data, width=650)

        with col2:
            st.subheader("Lime Explainer")
            heatmap = np.array(val['heatmap_lime'])
            heatmap_custom = np.array(val['cie_inspiriation'])
            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap_custom.max(), vmax = heatmap.max())
            ax.axis('off')
            st.write(fig)

        if clear_benchmark :
            st.stop()

# ---- Own Created Explainer Function ----
def own_creation():

    st.subheader("Upload an Image file")
    col1, col2 = st.columns(2)
    with col1 :         
        models_list = ["VGG16", "Inception"]
        network = st.selectbox("Select the Model", models_list)
    with col2 :
        uploaded_file = st.file_uploader("Choose an image to classify", type=["jpg", "jpeg", "png"])
    
    col1, col2 = st.columns(2)    
    with col1: clear_creation = st.button("Clear Screen")
    with col2: check_creation = st.button("Submit")

    if check_creation:

        pred = get_explanations(uploaded_file.getvalue())
        val = json.loads(pred)
        st.subheader(f"Top Classification from {network}")
        st.write(val['score'],val['prediction'], column= ('score','classification'))

        bytes_data = uploaded_file.read()
        col1,col2 = st.columns(2)
        with col1:
            st.subheader("Original Image")
            st.image(bytes_data)

        with col2:
            st.subheader("Custom Create Explainer")
            heatmap = np.array(val['heatmap_lime'])
            heatmap_custom = np.array(val['cie_inspiriation'])            
            fig, ax = plt.subplots()
            ax.imshow(heatmap, cmap = 'RdBu', vmin  = -heatmap_custom.max(), vmax = heatmap_custom.max())
            ax.axis('off')
            st.write(fig)

        if clear_creation :
            st.stop()

# Navigating to pages
if selected == 'Benchmark':
    benchmark()

if selected == 'Creations':
    own_creation()