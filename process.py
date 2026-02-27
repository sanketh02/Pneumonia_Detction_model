import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2 
print(cv2.__version__)  # to perform image operation
from PIL import Image   # image processing library
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import filedialog

model = load_model("model\\Pneumonia_detection_model.h5")
print("Model loaded successfully!")

# Take input image path
# Open file dialog
root = tk.Tk()
root.withdraw()

input_image_path = filedialog.askopenfilename(
    title="Select Chest X-ray Image",
    filetypes=[("Image Files", "*.png *.jpg *.jpeg")]
)


# Read image
input_image = cv2.imread(input_image_path)

# Check if image loaded properly
if input_image is None:
    print("Error: Image not found. Please check the path.")
    exit()

# Display image using OpenCV
cv2.imshow("Input Image", input_image)
cv2.waitKey(0)   # Wait until any key is pressed
cv2.destroyAllWindows()

# Resize image
input_image = cv2.resize(input_image, (128, 128))

# Normalize image
input_image_scaled = input_image / 255.0

# Reshape image for model
input_image_reshaped = np.reshape(input_image_scaled, [1, 128, 128, 3])

#load model


# Prediction
prediction = model.predict(input_image_reshaped)

print("Raw Prediction:", prediction)

# Get class label
input_predict_label = np.argmax(prediction)

if input_predict_label == 1:
    print("The person is suffering from Pneumonia")
else:
    print("The person is Normal")