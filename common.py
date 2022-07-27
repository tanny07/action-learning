from configparser import NoOptionError
import matplotlib.pyplot as plt
import uvicorn
from fastapi import FastAPI, File,Query, UploadFile, Request
from starlette.responses import RedirectResponse
from tensorflow.keras.applications import InceptionV3 
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.applications.vgg16 import decode_predictions as decode_predictions_vgg16


from PIL import Image
import numpy as np
from io import BytesIO
import cv2
from typing import Union
import cv2
import json

def transform_img_inception(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
      # out.append(x)
    return x

def transform_img_vgg16(img):
    x = np.expand_dims(img, axis=0)
    x = preprocess_input(x)
    return x

def get_predictions(model_name,model,img_arr, last_prediction = ''):
  if model_name == 'VGG16':
    img = transform_img_vgg16(img_arr)
    preds = model.predict(img)
    predictions = decode_predictions_vgg16(preds)[0]
  elif model_name == 'Inception_V3':
    img = transform_img_inception(img_arr) 
    preds = model.predict(img)
    predictions = decode_predictions(preds)[0]
  top_prediction, top_pred_score, prediction_type = get_scores(predictions,last_prediction)
  return top_prediction, top_pred_score, prediction_type

def get_scores(predictions,last_prediction):
  top_prediction, top_pred_score, prediction_type = last_prediction, 0, 0
  # 0 - initial state
  # 1 - image modified and prediction still in first position
  # 2 - image modified and prediction no more in first position
  # 3 - first image prediction
  for index, x in enumerate(predictions):
    label = x[1]
    score = x[2]
    if last_prediction != '' and last_prediction == label:
      if index == 0:
        top_prediction, top_pred_score, prediction_type = label, score, 1
      else:
        top_prediction, top_pred_score, prediction_type = label, score, 2
    else:
        top_prediction, top_pred_score, prediction_type = label, score, 3
    
    if prediction_type != 0:
      break
  return top_prediction, top_pred_score, prediction_type

def evaluate_pixels(row_start,row_to,column_start,column_to,frame,model_name,model,prediction):
    # blur = cv2.GaussianBlur(frame[row_start:row_to,column_start:column_to],(3,3),0)
    frame[row_start:row_to,column_start:column_to] = (0,0,0)
    top_prediction, top_pred_score, prediction_type = get_predictions(model_name,model,frame,prediction)
    return top_prediction, top_pred_score, prediction_type