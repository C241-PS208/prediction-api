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

# flask --app main.py --debug run

import sys

app = Flask(__name__)
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
model = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")
model_boy = VGG16(weights="D:/Semester 6/Bangkit/predict/boyvgg.h5")
# model_girl = VGG16(weights="D:/Semester 6/Bangkit/predict/girlvgg.h5")
model_hairtype = VGG16(weights="D:/Semester 6/Bangkit/predict/hairtypesvgg.h5")


male_face_shape_array = ["ovale", "rectangle", "square", "round"]
female_face_shape_array = ["oblong", "heart", "square", "ovale", "round"]
hair_type_array = ["straight", "wavy", "curly"]

@app.route("/")
def index():
    return "Endpoints: POST /predict, GET /status"

@app.route("/predict", methods=["POST"])
def predict():
    try: 
        # Gender Request Parameter
        gender = request.form.get('gender')

        req_image = request.files['image']
        req_image = Image.open(req_image).convert("RGB")
        req_image = req_image.resize((224, 224))

        image_array = image.img_to_array(req_image)

        image_array = np.expand_dims(image_array, axis=0) # (1, 244, 244, 3)

        image_array = preprocess_input(image_array)
        
        if gender == 'male':
            face_shape = random.choice(male_face_shape_array)
            pred = model_boy.predict(image_array)
            decoded_pred = decode_predictions(pred, top=10)[0]

        elif gender == 'female':
            face_shape = random.choice(female_face_shape_array)
            pred = model.predict(image_array)
            decoded_pred = decode_predictions(pred, top=10)[0]
        else:
            return jsonify({"error": "Gender is not valid"})
        
        hair_type = random.choice(hair_type_array)

        hair_pred = model_hairtype.predict(image_array)
        decoded_pred_hair = decode_predictions(hair_pred, top=10)[0]

        hair_result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred_hair]
        custom_print(hair_result)

        hair_recomendations = hair_recomendation(gender, face_shape, hair_type)

        # result = [{"class": pred[1], "description": pred[2], "probability": float(pred[2])} for pred in decoded_pred]
        result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred]

        return jsonify({
                        "face_type": face_shape,
                        "hair_type": hair_type,
                        "recommendations" : hair_recomendations})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route("/status")
def status():
    return "Predict server is up."

if (__name__ == "__main__"):
    app.run()

def make_prediction(model, image_array):
    pred = model.predict(image_array)
    decoded_pred = decode_predictions(pred, top=5)
    custom_print(decoded_pred)
    result = [{"class": pred[1], "probability": float(pred[2])} for pred in decoded_pred]
    return result

def hair_recomendation(gender, face_shape, hair_type):
    combinations = {
        ('male', 'round', 'straight'): [
            {
                'hairstyle': 'Undercut with Comb Over',
                'description': 'Provides a clean and sharp look that contrasts with the jawline. The sides are kept short while the top is left longer and combed over for a polished appearance.',
                'photo_url': 'URL to Undercut with Comb Over photo'
            },
            {
                'hairstyle': 'Side Part',
                'description': 'Adds asymmetry, softening the square features. The hair is parted to one side, creating a classic and sophisticated look.',
                'photo_url': 'URL to Side Part photo'
            },
            {
                'hairstyle': 'Textured Crop',
                'description': 'Adds volume and texture, reducing the boxiness of a square face. The hair is cut in short, choppy layers that add movement.',
                'photo_url': 'URL to Textured Crop photo'
            }
        ],
        ('male', 'round', 'wavy'): [
        {
            'hairstyle': 'Messy Quiff',
            'description': 'Adds height and texture, drawing attention upwards. This style is slightly tousled for a casual yet stylish appearance.',
            'photo_url': 'URL to Messy Quiff photo'
        },
        {
            'hairstyle': 'Wavy Side Swept',
            'description': 'Balances the strong jawline with soft waves. The hair is swept to one side, creating a relaxed and sophisticated look.',
            'photo_url': 'URL to Wavy Side Swept photo'
        },
        {
            'hairstyle': 'Curly Fringe',
            'description': 'Softens the angles with natural wave patterns. The fringe is left longer and styled to create a wavy, textured look.',
            'photo_url': 'URL to Curly Fringe photo'
        }
        ],
        ('male', 'round', 'curly'): [
            {
                'hairstyle': 'Curly Top with Tapered Sides',
                'description': 'Keeps the curls manageable and stylish. The sides are tapered short while the top is left longer to showcase natural curls.',
                'photo_url': 'URL to Curly Top with Tapered Sides photo'
            },
            {
                'hairstyle': 'Short Afro',
                'description': 'Maintains the natural curl pattern, emphasizing the round shape without overwhelming it. This style keeps the curls tight and neat.',
                'photo_url': 'URL to Short Afro photo'
            },
            {
                'hairstyle': 'Curly Undercut',
                'description': 'Contrasts the tight curls on top with short sides. This style is trendy and highlights the natural curls.',
                'photo_url': 'URL to Curly Undercut photo'
            }
        ],
        ('male', 'rectangle', 'straight'): [
            {
                'hairstyle': 'Pompadour',
                'description': 'Adds volume on top, balancing the face length. This style is characterized by its height and slicked-back appearance.',
                'photo_url': 'URL to Pompadour photo'
            },
            {
                'hairstyle': 'Side Part with Fade',
                'description': 'Adds width to the face, making it look less elongated. The fade creates a clean contrast between the sides and the longer top.',
                'photo_url': 'URL to Side Part with Fade photo'
            },
            {
                'hairstyle': 'Buzz Cut',
                'description': 'Creates a balanced and uniform look. This ultra-short haircut is easy to maintain and emphasizes facial features.',
                'photo_url': 'URL to Buzz Cut photo'
            }
        ],
        ('male', 'rectangle', 'wavy'): [
            {
                'hairstyle': 'Wavy Pompadour',
                'description': 'Combines volume and texture, ideal for a rectangular face. This style keeps the natural wave while adding height.',
                'photo_url': 'URL to Wavy Pompadour photo'
            },
            {
                'hairstyle': 'Loose Waves with Tapered Sides',
                'description': 'Keeps the focus on the hair without elongating the face. The sides are tapered short while the top remains wavy.',
                'photo_url': 'URL to Loose Waves with Tapered Sides photo'
            },
            {
                'hairstyle': 'Shaggy Layers',
                'description': 'Adds width and softens the long face shape. The hair is cut in layers to enhance the natural wave and add volume.',
                'photo_url': 'URL to Shaggy Layers photo'
            }
        ],
        ('male', 'rectangle', 'curly'): [
            {
                'hairstyle': 'Curly Pompadour',
                'description': 'Adds height and dimension without elongating the face. The curls are styled upwards and back.',
                'photo_url': 'URL to Curly Pompadour photo'
            },
            {
                'hairstyle': 'Medium Length Curls',
                'description': 'Provides volume and balance to the face. The curls are left at medium length for a natural, voluminous look.',
                'photo_url': 'URL to Medium Length Curls photo'
            },
            {
                'hairstyle': 'Curly Fringe with Undercut',
                'description': 'Keeps curls manageable and adds width to the face. The undercut contrasts with the curly fringe on top.',
                'photo_url': 'URL to Curly Fringe with Undercut photo'
            }
        ],
        ('male', 'ovale', 'straight'): [
            {
                'hairstyle': 'Classic Taper',
                'description': 'A timeless cut that complements the balanced proportions. The sides are tapered short, and the top is left slightly longer.',
                'photo_url': 'URL to Classic Taper photo'
            },
            {
                'hairstyle': 'Textured Pompadour',
                'description': 'Adds volume without elongating the face. The hair is styled upwards and back with added texture.',
                'photo_url': 'URL to Textured Pompadour photo'
            },
            {
                'hairstyle': 'Crew Cut',
                'description': 'Simple and neat, maintaining the natural balance. This short haircut is easy to maintain and suits an oval face well.',
                'photo_url': 'URL to Crew Cut photo'
            }
        ],
        ('male', 'ovale', 'wavy'): [
            {
                'hairstyle': 'Wavy Fringe',
                'description': 'Adds texture and keeps the face looking balanced. The fringe is left longer and styled to enhance the natural wave.',
                'photo_url': 'URL to Wavy Fringe photo'
            },
            {
                'hairstyle': 'Tousled Waves',
                'description': 'Enhances the natural wave pattern for a casual look. The hair is left at medium length and tousled for texture.',
                'photo_url': 'URL to Tousled Waves photo'
            },
            {
                'hairstyle': 'Medium Length Shag',
                'description': 'Softens the features and adds texture. The hair is cut in layers to enhance the wave and create a shaggy look.',
                'photo_url': 'URL to Medium Length Shag photo'
            }
        ],
        ('male', 'ovale', 'curly'): [
            {
                'hairstyle': 'Curly Caesar Cut',
                'description': 'Keeps curls tight and manageable. The hair is cut short with a slight fringe.',
                'photo_url': 'URL to Curly Caesar Cut photo'
            },
            {
                'hairstyle': 'Curly Quiff',
                'description': 'Adds height and texture, complementing the oval shape. The curls are styled upwards and back.',
                'photo_url': 'URL to Curly Quiff photo'
            },
            {
                'hairstyle': 'Medium Length Curls with Tapered Sides',
                'description': 'Balances the face with volume and control. The sides are tapered short while the curls are left at medium length.',
                'photo_url': 'URL to Medium Length Curls with Tapered Sides photo'
            }
        ],
        ('male', 'square', 'straight'): [
            {
                'hairstyle': 'Pompadour',
                'description': 'Adds volume on top, balancing the face length. This style is characterized by its height and slicked-back appearance.',
                'photo_url': 'URL to Pompadour photo'
            },
            {
                'hairstyle': 'Side Part with Fade',
                'description': 'Adds width to the face, making it look less elongated. The fade creates a clean contrast between the sides and the longer top.',
                'photo_url': 'URL to Side Part with Fade photo'
            },
            {
                'hairstyle': 'Buzz Cut',
                'description': 'Creates a balanced and uniform look. This ultra-short haircut is easy to maintain and emphasizes facial features.',
                'photo_url': 'URL to Buzz Cut photo'
            }
        ],
        ('male', 'square', 'wavy'): [
            {
                'hairstyle': 'Wavy Pompadour',
                'description': 'Combines volume and texture, ideal for a rectangular face. This style keeps the natural wave while adding height.',
                'photo_url': 'URL to Wavy Pompadour photo'
            },
            {
                'hairstyle': 'Loose Waves with Tapered Sides',
                'description': 'Keeps the focus on the hair without elongating the face. The sides are tapered short while the top remains wavy.',
                'photo_url': 'URL to Loose Waves with Tapered Sides photo'
            },
            {
                'hairstyle': 'Shaggy Layers',
                'description': 'Adds width and softens the long face shape. The hair is cut in layers to enhance the natural wave and add volume.',
                'photo_url': 'URL to Shaggy Layers photo'
            }
        ],
        ('male', 'square', 'curly'): [
            {
                'hairstyle': 'Curly Pompadour',
                'description': 'Adds height and dimension without elongating the face. The curls are styled upwards and back.',
                'photo_url': 'URL to Curly Pompadour photo'
            },
            {
                'hairstyle': 'Medium Length Curls',
                'description': 'Provides volume and balance to the face. The curls are left at medium length for a natural, voluminous look.',
                'photo_url': 'URL to Medium Length Curls photo'
            },
            {
                'hairstyle': 'Curly Fringe with Undercut',
                'description': 'Keeps curls manageable and adds width to the face. The undercut contrasts with the curly fringe on top.',
                'photo_url': 'URL to Curly Fringe with Undercut photo'
            }
        ],
        ('female', 'square', 'straight'): [
            {
                'hairstyle': 'Long Layers',
                'description': 'Adds softness and movement, reducing the angularity of a square face. The layers create a cascading effect that flatters the jawline.',
                'photo_url': 'URL to Long Layers photo'
            },
            {
                'hairstyle': 'Blunt Bob',
                'description': 'Emphasizes the sharp features of a square face with a sleek and sophisticated look. The length is usually just below the chin, enhancing the jawline.',
                'photo_url': 'URL to Blunt Bob photo'
            },
            {
                'hairstyle': 'Side-Swept Bangs',
                'description': 'Softens the forehead and adds a touch of asymmetry to balance the square shape. The bangs are swept to one side, creating a gentle frame for the face.',
                'photo_url': 'URL to Side-Swept Bangs photo'
            }
        ],
        ('female', 'square', 'wavy'): [
            {
                'hairstyle': 'Beach Waves',
                'description': 'Adds a soft, tousled look that breaks up the strong lines of a square face. The natural waves create volume and movement.',
                'photo_url': 'URL to Beach Waves photo'
            },
            {
                'hairstyle': 'Shaggy Lob',
                'description': 'Combines layers and waves for a textured look that adds dimension and softness to a square face. The lob length falls around the collarbone.',
                'photo_url': 'URL to Shaggy Lob photo'
            },
            {
                'hairstyle': 'Wavy Pixie Cut',
                'description': 'Adds a playful and chic element to the square face. The waves create a soft texture that contrasts with the structured shape.',
                'photo_url': 'URL to Wavy Pixie Cut photo'
            }
        ],
        ('female', 'square', 'curly'): [
            {
                'hairstyle': 'Curly Bob',
                'description': 'The curls add volume and texture, softening the angles of a square face. The bob length enhances the jawline while adding movement.',
                'photo_url': 'URL to Curly Bob photo'
            },
            {
                'hairstyle': 'Curly Shag',
                'description': 'Features layers and curls for a voluminous and textured look. This style breaks up the strong lines of a square face, adding softness.',
                'photo_url': 'URL to Curly Shag photo'
            },
            {
                'hairstyle': 'Long Curly Layers',
                'description': 'Adds length and volume, drawing attention away from the jawline. The curls create a soft and romantic look.',
                'photo_url': 'URL to Long Curly Layers photo'
            }
        ],
        ('female', 'heart', 'straight'): [
            {
                'hairstyle': 'Long Layers with Side Part',
                'description': 'Adds volume at the bottom to balance the wider forehead and narrower chin. The side part softens the forehead.',
                'photo_url': 'URL to Long Layers with Side Part photo'
            },
            {
                'hairstyle': 'Chin-Length Bob',
                'description': 'Balances the face shape by adding volume around the chin area. This length is perfect for highlighting the jawline.',
                'photo_url': 'URL to Chin-Length Bob photo'
            },
            {
                'hairstyle': 'Curtain Bangs',
                'description': 'Softens the forehead and draws attention to the eyes. The bangs part in the middle and frame the face nicely.',
                'photo_url': 'URL to Curtain Bangs photo'
            }
        ],
        ('female', 'heart', 'wavy'): [
            {
                'hairstyle': 'Textured Lob',
                'description': 'Adds body and movement, softening the chin and balancing the forehead. The waves create a relaxed and stylish look.',
                'photo_url': 'URL to Textured Lob photo'
            },
            {
                'hairstyle': 'Wavy Bob with Bangs',
                'description': 'Adds volume around the chin and softens the forehead with bangs. The waves add texture and dimension.',
                'photo_url': 'URL to Wavy Bob with Bangs photo'
            },
            {
                'hairstyle': 'Boho Waves',
                'description': 'Creates a soft, tousled look that balances the heart-shaped face. The waves add a carefree and feminine touch.',
                'photo_url': 'URL to Boho Waves photo'
            }
        ],
        ('female', 'heart', 'curly'): [
            {
                'hairstyle': 'Curly Bob',
                'description': 'Adds volume and balances the face by drawing attention away from the forehead. The curls create a playful and chic look.',
                'photo_url': 'URL to Curly Bob photo'
            },
            {
                'hairstyle': 'Curly Lob',
                'description': 'Adds length and volume, creating a balanced look. The curls enhance the overall volume and add a fun texture.',
                'photo_url': 'URL to Curly Lob photo'
            },
            {
                'hairstyle': 'Curly Pixie',
                'description': 'Highlights the natural curl pattern and creates a voluminous top, balancing the face shape. The sides are kept short for contrast.',
                'photo_url': 'URL to Curly Pixie photo'
            }
        ],
        ('female', 'ovale', 'straight'): [
            {
                'hairstyle': 'Long Layers',
                'description': 'Adds softness and movement, reducing the angularity of a square face. The layers create a cascading effect that flatters the jawline.',
                'photo_url': 'URL to Long Layers photo'
            },
            {
                'hairstyle': 'Blunt Bob',
                'description': 'Emphasizes the sharp features of a square face with a sleek and sophisticated look. The length is usually just below the chin, enhancing the jawline.',
                'photo_url': 'URL to Blunt Bob photo'
            },
            {
                'hairstyle': 'Side-Swept Bangs',
                'description': 'Softens the forehead and adds a touch of asymmetry to balance the square shape. The bangs are swept to one side, creating a gentle frame for the face.',
                'photo_url': 'URL to Side-Swept Bangs photo'
            }
        ],
        ('female', 'ovale', 'wavy'): [
            {
                'hairstyle': 'Beach Waves',
                'description': 'Adds a soft, tousled look that breaks up the strong lines of a square face. The natural waves create volume and movement.',
                'photo_url': 'URL to Beach Waves photo'
            },
            {
                'hairstyle': 'Shaggy Lob',
                'description': 'Combines layers and waves for a textured look that adds dimension and softness to a square face. The lob length falls around the collarbone.',
                'photo_url': 'URL to Shaggy Lob photo'
            },
            {
                'hairstyle': 'Wavy Pixie Cut',
                'description': 'Adds a playful and chic element to the square face. The waves create a soft texture that contrasts with the structured shape.',
                'photo_url': 'URL to Wavy Pixie Cut photo'
            }
        ],
        ('female', 'ovale', 'curly'): [
            {
                'hairstyle': 'Curly Bob',
                'description': 'The curls add volume and texture, softening the angles of a square face. The bob length enhances the jawline while adding movement.',
                'photo_url': 'URL to Curly Bob photo'
            },
            {
                'hairstyle': 'Curly Shag',
                'description': 'Features layers and curls for a voluminous and textured look. This style breaks up the strong lines of a square face, adding softness.',
                'photo_url': 'URL to Curly Shag photo'
            },
            {
                'hairstyle': 'Long Curly Layers',
                'description': 'Adds length and volume, drawing attention away from the jawline. The curls create a soft and romantic look.',
                'photo_url': 'URL to Long Curly Layers photo'
            }
        ],
        ('female', 'oblong', 'straight'): [
            {
                'hairstyle': 'Long Layers',
                'description': 'Adds volume and movement, breaking up the length of an oblong face. The layers start around the chin, creating a balanced look.',
                'photo_url': 'URL to Long Layers photo'
            },
            {
                'hairstyle': 'Blunt Bangs',
                'description': 'Shortens the appearance of an oblong face by covering the forehead. Blunt bangs add a chic and youthful touch.',
                'photo_url': 'URL to Blunt Bangs photo'
            },
            {
                'hairstyle': 'Shoulder-Length Cut',
                'description': 'Adds width around the face and draws attention to the collarbone, balancing the long face shape.',
                'photo_url': 'URL to Shoulder-Length Cut photo'
            }
        ],
        ('female', 'oblong', 'wavy'): [
            {
                'hairstyle': 'Wavy Lob',
                'description': 'Adds volume and texture, breaking up the length of an oblong face. The waves create a soft and feminine look.',
                'photo_url': 'URL to Wavy Lob photo'
            },
            {
                'hairstyle': 'Side-Swept Waves',
                'description': 'Adds volume and movement to one side, creating an asymmetrical look that shortens the face.',
                'photo_url': 'URL to Side-Swept Waves photo'
            },
            {
                'hairstyle': 'Beach Waves',
                'description': 'Adds a relaxed and tousled look, breaking up the length and adding softness.',
                'photo_url': 'URL to Beach Waves photo'
            }
        ],
        ('female', 'oblong', 'curly'): [
            {
                'hairstyle': 'Curly Bob',
                'description': 'Adds volume and width, balancing the length of an oblong face. The curls create a playful and chic look.',
                'photo_url': 'URL to Curly Bob photo'
            },
            {
                'hairstyle': 'Curly Lob',
                'description': 'Adds length and volume, breaking up the vertical lines of an oblong face. The curls create a soft and voluminous look.',
                'photo_url': 'URL to Curly Lob photo'
            },
            {
                'hairstyle': 'Long Curly Layers',
                'description': 'Adds volume and movement, drawing attention away from the length of the face. The layers create a soft and flowing look.',
                'photo_url': 'URL to Long Curly Layers photo'
            }
        ],
        ('female', 'round', 'straight'): [
            {
                'hairstyle': 'Long Layers',
                'description': 'Adds length and balances the width of a round face. The layers start around the chin, creating a slimming effect.',
                'photo_url': 'URL to Long Layers photo'
            },
            {
                'hairstyle': 'Blunt Cut',
                'description': 'Adds structure and sharp lines, contrasting the roundness of the face. The length usually falls below the chin.',
                'photo_url': 'URL to Blunt Cut photo'
            },
            {
                'hairstyle': 'Side-Swept Bangs',
                'description': 'Adds asymmetry and draws attention to the eyes, creating a slimming effect. The bangs are swept to one side.',
                'photo_url': 'URL to Side-Swept Bangs photo'
            }
        ],
        ('female', 'round', 'wavy'): [
            {
                'hairstyle': 'Textured Lob',
                'description': 'Adds volume and movement, balancing the width of a round face. The waves create a soft and stylish look.',
                'photo_url': 'URL to Textured Lob photo'
            },
            {
                'hairstyle': 'Beach Waves',
                'description': 'Adds a tousled and relaxed look, breaking up the roundness of the face. The waves create a soft and feminine look.',
                'photo_url': 'URL to Beach Waves photo'
            },
            {
                'hairstyle': 'Shaggy Bob',
                'description': 'Adds layers and waves for a textured look that balances the width of a round face. The shaggy style creates volume and movement.',
                'photo_url': 'URL to Shaggy Bob photo'
            }
        ],
        ('female', 'round', 'curly'): [
            {
                'hairstyle': 'Curly Bob',
                'description': 'Adds volume and texture, balancing the width of a round face. The curls create a playful and chic look.',
                'photo_url': 'URL to Curly Bob photo'
            },
            {
                'hairstyle': 'Curly Shag',
                'description': 'Features layers and curls for a voluminous and textured look. This style adds height and breaks up the roundness of the face.',
                'photo_url': 'URL to Curly Shag photo'
            },
            {
                'hairstyle': 'Long Curly Layers',
                'description': 'Adds length and volume, drawing attention away from the width of the face. The curls create a soft and flowing look.',
                'photo_url': 'URL to Long Curly Layers photo'
            }
        ]
    }

    key = (gender, face_shape, hair_type)
    if key in combinations:
        return(combinations[key])
    else:
        return("Combination not found.")

def custom_print(toPrint):
    print(toPrint, file=sys.stdout)