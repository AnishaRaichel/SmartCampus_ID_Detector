# main.py
# SmartCampus ID Detector - ANN vs CNN
# Author: Ani

import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# --- Setup directories ---
data_dir = r"D:\SmartCampus_ID_Detector\data"  # use raw string r"" for Windows paths
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

# --- Load and preprocess data ---
datagen = ImageDataGenerator(rescale=1./255)

# Use all images as training data (sufficient for demo with tiny dataset)
train_data = datagen.flow_from_directory(
    data_dir,
    target_size=(128,128),
    batch_size=1,
    class_mode='binary',
    shuffle=True
)

# For demo purposes, the same dataset as "validation"
val_data = train_data

# --- 1️ Traditional ANN Model ---
ann = Sequential([
    Flatten(input_shape=(128,128,3)),
    Dense(128, activation='relu'),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

ann.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("\n🔹 Training ANN Model...")
ann_hist = ann.fit(train_data, validation_data=val_data, epochs=5)
ann.save(os.path.join(model_dir, 'ann_model.h5'))
print("✅ ANN model saved!")

# --- Display ANN Metrics ---
ann_train_acc = ann_hist.history['accuracy'][-1]
ann_val_acc = ann_hist.history['val_accuracy'][-1]
print(f"\n📈 ANN Final Training Accuracy: {ann_train_acc*100:.2f}%")
print(f"📊 ANN Final Validation Accuracy: {ann_val_acc*100:.2f}%")

# --- 2️ CNN Model (Recent Model) ---
cnn = Sequential([
    Conv2D(16, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2,2),
    Conv2D(32, (3,3), activation='relu'),
    MaxPooling2D(2,2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

cnn.compile(optimizer=Adam(0.001), loss='binary_crossentropy', metrics=['accuracy'])
print("\n🔹 Training CNN Model...")
cnn_hist = cnn.fit(train_data, validation_data=val_data, epochs=5)
cnn.save(os.path.join(model_dir, 'cnn_model.h5'))
print("✅ CNN model saved!")

# --- Display CNN Metrics ---
cnn_train_acc = cnn_hist.history['accuracy'][-1]
cnn_val_acc = cnn_hist.history['val_accuracy'][-1]
print(f"\n📈 CNN Final Training Accuracy: {cnn_train_acc*100:.2f}%")
print(f"📊 CNN Final Validation Accuracy: {cnn_val_acc*100:.2f}%")

# --- 📊 Compare results visually ---
plt.figure(figsize=(7,4))
plt.plot(ann_hist.history['accuracy'], label='ANN Train Acc')
plt.plot(cnn_hist.history['accuracy'], label='CNN Train Acc')
plt.title("Performance Comparison: ANN vs CNN")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
