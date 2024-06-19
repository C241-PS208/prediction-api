import os
from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras._tf_keras.keras.preprocessing import image
from keras._tf_keras.keras.applications.vgg16 import preprocess_input, decode_predictions, VGG16
from PIL import Image
import random
from hair_recommendation import hair_recommendation

# flask --app main.py --debug run

import sys

app = Flask(__name__)
app.json.sort_keys = False

UPLOAD_FOLDER = './images'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# ADJUST THIS TO MODEL LOCATION ON YOUR LOCAL FILES!!!!!!!!!!!!!!
# model = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")

# VINSEN
# model = VGG16(weights="/Users/vinsensiusferdinando/Desktop/predict/boyvgg.h5")
# model_boy = VGG16(weights="/Users/vinsensiusferdinando/Desktop/predict/boyvgg.h5")
# # model_girl = VGG16(weights="/Users/vinsensiusferdinando/Desktop/predict/girlvgg.h5")
# model_hairtype = VGG16(weights="/Users/vinsensiusferdinando/Desktop/predict/hairtypesvgg.h5")

# BINTANG
# model = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")
# model_boy = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")
# model_girl = VGG16(weights="D:/Semester 6/Bangkit/predict/girlvgg.h5")
# model_hairtype = VGG16(weights="D:/Semester 6/Bangkit/predict/hairtypesvgg.h5")

# VM
# model = VGG16(weights="./models/boymodified.h5")
# model_boy = VGG16(weights="./models/boymodified.h5")
# model_girl = VGG16(weights=".models/girlFacevgg13.h5")
model = tf.keras.models.load_model("./models/boymodified.h5")
model_boy = tf.keras.models.load_model("./models/boymodified.h5")
model_girl = tf.keras.models.load_model("./models/girlFacevgg13.h5")

model_hairtype = VGG16(weights="models/hairtypesvgg.h5")

male_face_shape_array = ["ovale", "rectangle", "round", "square"]
female_face_shape_array = ["oblong", "heart", "square", "ovale", "round"]
hair_type_array = ["straight", "wavy", "curly"]

@app.route("/")
def index():
    return "Endpoints: POST /predict, GET /status"

@app.route("/predict", methods=["POST"])
def predict():
    try: 
        # Gender Request Body
        gender = request.form.get('gender')

        # image processing
        req_image = request.files['image']
        req_image = Image.open(req_image).convert("RGB")
        req_image = req_image.resize((224, 224))
        image_array = image.img_to_array(req_image)
        image_array = np.expand_dims(image_array, axis=0) # (1, 244, 244, 3)
        image_array = preprocess_input(image_array)

        predicted_classes = []

        # FACE SHAPE PREDICTION
        if gender == 'male':
            face_shape = random.choice(male_face_shape_array)
            pred = model_boy.predict(image_array)
            # decoded_pred = decode_predictions(pred, top=10)[0]
            predicted_class_indices = np.argmax(pred, axis=1)

            # Ensure predicted class indices are within range of classes list
            for idx in predicted_class_indices:
                if 0 <= idx < len(male_face_shape_array):
                    predicted_classes.append(male_face_shape_array[idx])
                else:
                    predicted_classes.append("Unknown")  # Handle out-of-range indices gracefully
        elif gender == 'female':
            face_shape = random.choice(female_face_shape_array)
            pred = model_girl.predict(image_array)
            # decoded_pred = decode_predictions(pred, top=10)[0]
            predicted_class_indices = np.argmax(pred, axis=1)
            print(pred)

            # Ensure predicted class indices are within range of classes list
            # TODO only resulting in 4 indices, needed 5 because girls have 5 face shapes
            for idx in predicted_class_indices:
                if 0 <= idx < len(female_face_shape_array):
                    predicted_classes.append(female_face_shape_array[idx])
                else:
                    predicted_classes.append("Unknown")  # Handle out-of-range indices gracefully
        else:
            return jsonify({"error": "Gender is not valid"})

        # FACE SHAPE RESULT
        face_shape_result = predicted_classes[0]

        # HAIR TYPE RANDOMIZER 
        hair_type = random.choice(hair_type_array)



        # HAIR TYPE PREDICTION
        # hair_pred = model_hairtype.predict(image_array)
        # decoded_pred_hair = decode_predictions(hair_pred, top=10)[0]

        # RECOMMENDATION GENERATOR
        hair_recomendations = hair_recommendation(gender, face_shape_result, hair_type)

        # RESULT JSON
        # result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred]
        # hair_result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred_hair]

        return jsonify({"face_type": face_shape_result, "hair_type": hair_type, "recommendations" : hair_recomendations})
        # return jsonify({"pred": predicted_classes[0]})

    except Exception as e:
        print(str(e), file=sys.stdout)
        # custom_print(str(e))
        return jsonify({"error": str(e)})

@app.route("/status")
def status():
    return "Predict server is up."

if (__name__ == "__main__"):
    app.run(host="0.0.0.0", port="5000")

def make_prediction(model, image_array):
    pred = model.predict(image_array)
    decoded_pred = decode_predictions(pred, top=5)
    # custom_print(decoded_pred)
    result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred]
    return result



def custom_print(toPrint):
    print(toPrint, file=sys.stdout)