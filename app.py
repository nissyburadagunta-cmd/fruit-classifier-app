import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json

# Load the trained model and class indices
@st.cache_resource
def load_model_and_classes():
    model = tf.keras.models.load_model("fruit_classifier_mobilenetv2.h5")
    with open("class_indices.json", "r") as f:
        class_indices = json.load(f)
    index_to_class = {v: k for k, v in class_indices.items()}
    return model, index_to_class

model, index_to_class = load_model_and_classes()

# Streamlit UI
st.title("🍎 Fruit Classifier")
st.write("Upload a fruit image and the model will predict its class!")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img_resized = image.resize((100, 100))
    img_array = np.array(img_resized) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    predictions = model.predict(img_array)
    predicted_index = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    predicted_label = index_to_class[predicted_index]

    st.success(f"**Prediction:** {predicted_label} ({confidence:.2f}%)")

