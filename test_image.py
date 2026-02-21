# test_image.py
# SmartCampus ID Detector - Predict single image
# Author: Ani

import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import os

# --- Paths ---
model_path = os.path.join('models', 'cnn_model.h5')  # Use CNN model for demo
test_img_path = 'test.jpg'  # Replace with your image filename

# --- Load trained model ---
model = load_model(model_path)
print(f"\n✅ Loaded model from {model_path}")

# --- Load and preprocess image ---
img = image.load_img(test_img_path, target_size=(128,128))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0) / 255.0

# --- Predict ---
pred = model.predict(x)
label = "With ID" if pred[0][0] < 0.5 else "Without ID"
confidence = pred[0][0]

print(f"\n🧠 Prediction: {label} (confidence: {confidence:.3f})")
