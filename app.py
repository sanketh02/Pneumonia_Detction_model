import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image

# Load Model
@st.cache_resource
def load_trained_model():
    model = load_model("model/Pneumonia_detection_model.h5")
    return model

model = load_trained_model()

# App Title
st.title("🩺 Pneumonia Detection using CNN")
st.write("Upload a Chest X-ray image to detect Pneumonia.")

# File Upload
uploaded_file = st.file_uploader(
    "Choose an X-ray image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Display uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Convert image to numpy array
    image = np.array(image)

    # Resize image
    image_resized = cv2.resize(image, (128, 128))

    # Normalize image
    image_scaled = image_resized / 255.0

    # Reshape for model
    image_reshaped = np.reshape(image_scaled, (1, 128, 128, 3))

    # Prediction
    prediction = model.predict(image_reshaped)
    confidence = float(np.max(prediction))
    predicted_label = np.argmax(prediction)

    st.write("### Prediction Result:")

    if predicted_label == 1:
        st.error(f"⚠ The person is suffering from Pneumonia")
    else:
        st.success(f"✅ The person is Normal")

    st.write(f"Confidence: {confidence:.2f}")