import streamlit as st
import numpy as np
import cv2
import joblib

from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from tensorflow.keras.models import Model
import os
import gdown

st.title("Autistic Children Emotion Recognition")
st.write("Upload Face Image and Hand Gesture Image to Predict Emotion")
# Download encoder model
if not os.path.exists("encoder_model.h5"):
    gdown.download(
        "https://drive.google.com/uc?id=1KS1AQeX9w79p8TnZeZAAi6DrAJ3PiKKP",
        "encoder_model.h5",
        quiet=False
    )

# -----------------------------
# Load models
# -----------------------------
classifier = load_model("classifier_model.h5")
autoencoder = load_model("encoder_model.h5")
scaler = joblib.load("scaler.pkl")
emotion_labels = joblib.load("emotion_label.pkl")

# -----------------------------
# Load VGG19 feature extractor
# -----------------------------
base_model = VGG19(weights='imagenet', include_top=False)

vgg_model = Model(
    inputs=base_model.input,
    outputs=base_model.output
)

# -----------------------------
# Upload images
# -----------------------------
face_file = st.file_uploader("Upload Face Image", type=["jpg","jpeg","png"])
hand_file = st.file_uploader("Upload Hand Gesture Image", type=["jpg","jpeg","png"])


# -----------------------------
# Preprocess image
# -----------------------------
def preprocess_image(uploaded_file):

    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img = cv2.resize(img,(224,224))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    img = preprocess_input(img)

    img = np.expand_dims(img, axis=0)

    return img


# -----------------------------
# Feature extraction
# -----------------------------
def extract_features(uploaded_file):

    img = preprocess_image(uploaded_file)

    features = vgg_model.predict(img)

    features = features.flatten().reshape(1,-1)

    return features


# -----------------------------
# Prediction
# -----------------------------
if st.button("Predict Emotion"):

    if face_file is None or hand_file is None:
        st.warning("Please upload both images")

    else:

        face_features = extract_features(face_file)
        hand_features = extract_features(hand_file)

        fused_features = np.concatenate([face_features, hand_features], axis=1)

        fused_scaled = scaler.transform(fused_features)

        encoded = autoencoder.predict(fused_scaled)

        prediction = classifier.predict(encoded)

        predicted_index = np.argmax(prediction)

        emotion = emotion_labels[predicted_index]

        st.success(f"Predicted Emotion: {emotion}")
