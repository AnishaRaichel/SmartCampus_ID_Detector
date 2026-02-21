# Smart Campus ID Card Detection System

An AI-based computer vision system that detects whether individuals in a group image are wearing ID cards. The system uses YOLOv8 for person detection and a combination of CNN and ANN models for classification.

---

## Overview

This project identifies people in images and determines if they are wearing ID cards. It is designed for smart campus environments to automate ID compliance monitoring.

---
## Demo


---

## Features

- Detects multiple people in a single image
- Uses YOLOv8 for person detection
- Extracts relevant regions (chest area) for ID detection
- Applies CNN and ANN models for classification
- Uses ensemble learning to improve accuracy
- Provides a graphical interface using Tkinter
- Displays the number of people without ID cards

---

## Tech Stack

- Python
- OpenCV
- TensorFlow / Keras
- YOLOv8 (Ultralytics)
- NumPy
- Tkinter

---

## Project Structure
smartcampus-id-detector/
│── models/ # Trained models
│── data/ # Dataset
│── static/ # UI assets
│── templates/ # Templates (if used)
│── test/ # Test scripts
│── main.py # Core logic
│── new_ui.py # GUI application
│── yolov8n.pt # YOLO model weights
│── requirements.txt # Dependencies
│── README.md

---

## How It Works

1. The user uploads an image
2. YOLOv8 detects all persons in the image
3. The chest region of each person is extracted
4. The extracted region is passed to:
   - CNN model
   - ANN model
5. Predictions from both models are combined using an ensemble approach
6. The system classifies each person as:
   - With ID
   - Without ID

---

## Model Details

- YOLOv8: Detects persons in images
- CNN: Performs image-based classification
- ANN: Performs feature-based classification
- Ensemble: Combines predictions to improve accuracy

---

## Output

- Bounding boxes around detected individuals
- Classification labels for each person
- Count of individuals not wearing ID cards

---

## Future Enhancements

- Real-time video processing
- Web-based interface (Flask or Streamlit)
- Integration with attendance systems
- Face recognition module
