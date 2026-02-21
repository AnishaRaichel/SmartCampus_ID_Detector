# group_test_ui.py
import cv2, os, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk

# ---------------- CONFIG ----------------
ANN_PATH = os.path.join('models', 'ann_model.h5')
CNN_PATH = os.path.join('models', 'cnn_model.h5')

INPUT_SIZE = (128,128)
CHEST_TOP = 0.28
CHEST_BOTTOM = 0.62
SIDE_MARGIN = 0.12
ENSEMBLE_WEIGHT_CNN = 0.7
THRESH = 0.40

# ---------------- LOAD MODELS ----------------
ann = load_model(ANN_PATH)
cnn = load_model(CNN_PATH)
yolo = YOLO("yolov8n.pt")
print("✅ Loaded ANN, CNN, and YOLO models.")

# ---------------- HELPERS ----------------
def clamp(a, lo, hi): 
    return max(lo, min(hi, a))

def process_image(filepath):
    img = cv2.imread(filepath)
    if img is None:
        messagebox.showerror("Error", "Cannot read the selected image.")
        return

    orig = img.copy()
    results = yolo(filepath)
    boxes = results[0].boxes.xyxy.cpu().numpy()

    no_id_count = 0

    for i, bbox in enumerate(boxes):
        x1, y1, x2, y2 = map(int, bbox)
        w, h = x2 - x1, y2 - y1

        cx1 = x1 + int(w * SIDE_MARGIN)
        cx2 = x2 - int(w * SIDE_MARGIN)
        cy1 = y1 + int(h * CHEST_TOP)
        cy2 = y1 + int(h * CHEST_BOTTOM)

        cx1 = clamp(cx1, 0, img.shape[1]-1)
        cx2 = clamp(cx2, 0, img.shape[1]-1)
        cy1 = clamp(cy1, 0, img.shape[0]-1)
        cy2 = clamp(cy2, 0, img.shape[0]-1)

        chest = img[cy1:cy2, cx1:cx2]
        if chest.shape[0] < 20 or chest.shape[1] < 20:
            chest = img[y1:y2, x1:x2]

        chest_resized = cv2.resize(chest, INPUT_SIZE)
        x = image.img_to_array(chest_resized) / 255.0
        x = np.expand_dims(x, axis=0)

        ann_pred = ann.predict(x, verbose=0)[0][0]
        cnn_pred = cnn.predict(x, verbose=0)[0][0]
        ensemble_score = ENSEMBLE_WEIGHT_CNN * cnn_pred + (1-ENSEMBLE_WEIGHT_CNN) * ann_pred

        label = "With ID" if ensemble_score < THRESH else "Without ID"
        color = (0,255,0) if label == "With ID" else (0,0,255)
        if label == "Without ID":
            no_id_count += 1

        cv2.rectangle(orig, (x1,y1), (x2,y2), color, 2)
        cv2.putText(orig, f"{label} ({ensemble_score:.2f})",
                    (x1, max(10, y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    # Convert for Tkinter
    img_rgb = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_pil.thumbnail((600, 500))
    img_tk = ImageTk.PhotoImage(img_pil)

    result_label.config(text=f"{no_id_count} person{'s' if no_id_count != 1 else ''} detected without ID card")
    image_label.config(image=img_tk)
    image_label.image = img_tk  # Keep reference


def browse_file():
    file_path = filedialog.askopenfilename(
        title="Select Group Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")]
    )
    if file_path:
        process_image(file_path)

# ---------------- GUI ----------------
root = tk.Tk()
root.title("ID Card Detector")
root.geometry("800x650")
root.configure(bg="#2e2e2e")

title_label = tk.Label(root, text="ID Card Detection", font=("Arial", 20, "bold"), fg="white", bg="#2e2e2e")
title_label.pack(pady=15)

browse_btn = tk.Button(root, text="📁 Upload Group Image", font=("Arial", 14), command=browse_file, bg="#ff8c00", fg="white", padx=15, pady=5)
browse_btn.pack(pady=10)

result_label = tk.Label(root, text="Upload an image to start.", font=("Arial", 14), fg="white", bg="#2e2e2e")
result_label.pack(pady=15)

image_label = tk.Label(root, bg="#2e2e2e")
image_label.pack(pady=10)

root.mainloop()
