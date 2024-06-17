import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from PIL import Image

# flask --app main.py --debug run

import sys

app = Flask(__name__)
UPLOAD_FOLDER = './images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ADJUST THIS TO MODEL LOCATION ON YOUR LOCAL FILES!!!!!!!!!!!!!!
model = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")

@app.route("/")
def index():
    return "Endpoints: POST /predict, GET /status"

@app.route("/predict", methods=["POST"])
def predict():
    try: 
        req_image = request.files['image']
        req_image = Image.open(req_image).convert("RGB")
        req_image = req_image.resize((224, 224))

        image_array = image.img_to_array(req_image)

        image_array = np.expand_dims(image_array, axis=0) # (1, 244, 244, 3)

        image_array = preprocess_input(image_array)

        pred = model.predict(image_array)
        decoded_pred = decode_predictions(pred, top=10)[0]
        custom_print(decoded_pred)

        # result = [{"class": pred[1], "description": pred[2], "probability": float(pred[2])} for pred in decoded_pred]
        result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred]

        custom_print("")
        custom_print("")
        custom_print("")
        custom_print("")
        for pred in decoded_pred:
            custom_print(pred)

        return jsonify({"predictions": result})

        # debugs to check if image file is actually being requested and exists
        # image.save(os.path.join(UPLOAD_FOLDER, secure_filename(image.filename)))

        # return "PREDICT POST"
    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/status")
def status():
    return "Predict server is up."

if (__name__ == "__main__"):
    app.run()



def custom_print(toPrint):
    print(toPrint, file=sys.stdout)