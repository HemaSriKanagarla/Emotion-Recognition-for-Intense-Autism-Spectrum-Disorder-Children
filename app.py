import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from flask import Flask, request, jsonify
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg19 import VGG19, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from PIL import Image
import gdown

app = Flask(__name__)

# -----------------------------------
# Download models from Google Drive
# -----------------------------------

if not os.path.exists("encoder_model.h5"):
    gdown.download(
        "https://drive.google.com/uc?id=19jL0RvDoXiYkO0eQnWiAFcU5G_vK-NEk",
        "encoder_model.h5",
        quiet=False
    )

if not os.path.exists("classifier_model.h5"):
    gdown.download(
        "https://drive.google.com/uc?id=1Rm0PtfvIe0SAonAxjFqFoLir1imE4Dnl",
        "classifier_model.h5",
        quiet=False
    )

if not os.path.exists("class_labels.npy"):
    gdown.download(
        "https://drive.google.com/uc?id=1KS1AQeX9w79p8TnZeZAAi6DrAJ3PiKKP",
        "class_labels.npy",
        quiet=False
    )

# -----------------------------------
# Load Models
# -----------------------------------

encoder = load_model("encoder_model.h5")
classifier = load_model("classifier_model.h5")
class_labels = np.load("class_labels.npy", allow_pickle=True)

# -----------------------------------
# Load VGG19 Feature Extractor
# -----------------------------------

base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224,224,3))
x = tf.keras.layers.Flatten()(base_model.output)
vgg_model = Model(inputs=base_model.input, outputs=x)

def extract_features(img):
    img = img.resize((224,224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    features = vgg_model.predict(img_array)
    return features

# -----------------------------------
# Routes
# -----------------------------------

@app.route("/")
def home():
    return "ASD Hand Gesture Model Running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"})

    file = request.files["file"]
    img = Image.open(file).convert("RGB")

    features = extract_features(img)
    encoded = encoder.predict(features)
    prediction = classifier.predict(encoded)

    predicted_class = class_labels[np.argmax(prediction)]

    return jsonify({"prediction": str(predicted_class)})

# -----------------------------------
# Run App
# -----------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)