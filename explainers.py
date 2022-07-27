import os,sys
import lime
from lime import lime_image
import numpy as np
from tensorflow.keras.applications.inception_v3 import preprocess_input
from common import get_predictions, evaluate_pixels

def get_explanations(explanation_type,images,model,model_name):
    if explanation_type == 'LIME':
        x = np.expand_dims(images, axis=0)
        x = preprocess_input(images)
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(x.astype('double'), model.predict, top_labels=5, hide_color=0, num_samples=10)
        ind =  explanation.top_labels[0]
        print(ind)
        #Map each explanation weight to the corresponding superpixel
        dict_heatmap = dict(explanation.local_exp[ind])
        heatmap = np.vectorize(dict_heatmap.get)(explanation.segments) 
        return heatmap
    elif explanation_type == 'CIE_INSPIRATION':
        prediction, prediction_score, prediction_type = get_predictions(model_name,model,images)
        mask = np.zeros(shape=(224, 224))
        row_start = 0
        window_size = 56
        frame_no = 0
        while row_start <=168:
            column_start = 0
            while column_start <=168:
                frame = images.copy()
                # frame = np.asarray(img_copy)
                top_prediction, top_pred_score, prediction_type = evaluate_pixels(row_start,row_start+window_size,column_start,column_start+window_size,frame,model_name,model,prediction)
                print('Prediction for block ',frame_no+1,' - ',top_prediction, top_pred_score, prediction_type)
                diff = top_pred_score - prediction_score
                # normalize_diff = 
                mask[row_start:row_start+window_size,column_start:column_start+window_size] = diff
                column_start = column_start + window_size
                frame_no = frame_no+1
            row_start = row_start + window_size
        return mask
