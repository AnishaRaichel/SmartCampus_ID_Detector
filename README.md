# 🏫 SmartCampus ID Detector

An intelligent computer vision system that detects students without ID cards in group images using deep learning.  
The project combines **YOLOv8** for face/person detection and an **ensemble of ANN and CNN models** for ID classification.

---

## 🚀 Features
- Detects multiple individuals in a single frame.
- Identifies students without ID cards using a hybrid ANN-CNN model.
- Lightweight and GUI-based — built with **Tkinter** for easy interaction.
- Uses **YOLOv8** for robust region detection.

---

## 🧠 Project Structure
- 📁 data/ → Training dataset (images)
- 📁 models/ → Saved ANN & CNN models
- 📁 static/, templates/ → UI assets (optional if used)
- 📁 test/ → Sample test images
- 📄 main.py → Model training and comparison (ANN vs CNN)
- 📄 new_ui.py → GUI app for real-time ID detection
- 📄 requirements.txt → Dependencies list

- 📄 README.md → Project documentation
