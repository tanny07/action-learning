from configparser import NoOptionError
from matplotlib import image
import uvicorn
from fastapi import FastAPI, File,Query, UploadFile
from starlette.responses import RedirectResponse
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import imagenet_utils
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image
import numpy as np
from io import BytesIO
import cv2

#from application.components import predict, read_imagefile

app_desc = """<h2>Try this app by uploading any image with `predict/image`</h2>"""

app = FastAPI(title='Tensorflow FastAPI Starter Pack', description=app_desc)

@app.get('/')
async def root():
    return {"message" : "please upload an image in the uploader"}



#@app.get('/select_model')
#async def get_model(_q: str = Query("Select Model for Classification", enum=["VGG16", "Inception"])):
#    return {"selected": _q}

@app.post("/classification/image")
async def upload_image(image_file: UploadFile = File(...)):
    kidarhoo = {}
   
    if image_file:
    #    kidarhoo = cv2.imread(image_file)

        bytes_data = await image_file.file.read()
    # inputShape = (224, 224)
    # preprocess = imagenet_utils.preprocess_input
    # network = "Inception"

    # MODELS = {
    # "VGG16": VGG16,
    # "Inception": InceptionV3,
    # }

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
    # magenetID, label, prob = predictions[0][0] 
    return {'value' : bytes_data} 
    #return {'value': predictions}
   

    

#@app.get('/selections')
#async def selection_and( _1 : str= Query("VGG16", enum = ["VGG16", "Inception"]), _2 : str = Query("Response", enum = ['target', 'response']):

#    return({"task": _1, "target": _2})


if __name__ == "__main__":
    uvicorn.run(app, debug=True)