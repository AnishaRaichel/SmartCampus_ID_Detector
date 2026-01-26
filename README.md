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
📁 data/ → Training dataset (images)
📁 models/ → Saved ANN & CNN models
📁 static/, templates/ → UI assets (optional if used)
📁 test/ → Sample test images
📄 main.py → Model training and comparison (ANN vs CNN)
📄 new_ui.py → GUI app for real-time ID detection
📄 requirements.txt → Dependencies list
📄 README.md → Project documentation

---

## Demo
![SmartCampus ID Detector Demo](https://private-user-images.githubusercontent.com/218777513/540464183-c931e08b-3ea4-497a-99e9-641c711b1772.gif?jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmF3LmdpdGh1YnVzZXJjb250ZW50LmNvbSIsImtleSI6ImtleTUiLCJleHAiOjE3Njk0MjU1NTMsIm5iZiI6MTc2OTQyNTI1MywicGF0aCI6Ii8yMTg3Nzc1MTMvNTQwNDY0MTgzLWM5MzFlMDhiLTNlYTQtNDk3YS05OWU5LTY0MWM3MTFiMTc3Mi5naWY_WC1BbXotQWxnb3JpdGhtPUFXUzQtSE1BQy1TSEEyNTYmWC1BbXotQ3JlZGVudGlhbD1BS0lBVkNPRFlMU0E1M1BRSzRaQSUyRjIwMjYwMTI2JTJGdXMtZWFzdC0xJTJGczMlMkZhd3M0X3JlcXVlc3QmWC1BbXotRGF0ZT0yMDI2MDEyNlQxMTAwNTNaJlgtQW16LUV4cGlyZXM9MzAwJlgtQW16LVNpZ25hdHVyZT01MzU4ODczMzllNzViODRmNmFlMTRlYzRmN2ZhMGViMzU4NDgxMzE4NjZkOTY1NjZkNDM4ZThhYmI4N2E2NWUxJlgtQW16LVNpZ25lZEhlYWRlcnM9aG9zdCJ9.0-aaHfh6v0ahnEsRuVP3_PQJWZx0FpkdHDfaULkYH-Q)
