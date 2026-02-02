from flask import Flask, render_template, request
import os, cv2, numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from ultralytics import YOLO
from werkzeug.utils import secure_filename

# ---------------- CONFIG ----------------
UPLOAD_FOLDER = 'static/uploads'
OUTPUT_FOLDER = 'static/output'

ANN_PATH = os.path.join('models', 'ann_model.h5')
CNN_PATH = os.path.join('models', 'cnn_model.h5')
YOLO_PATH = 'yolov8n.pt'

INPUT_SIZE = (128, 128)
CHEST_TOP = 0.28
CHEST_BOTTOM = 0.62
SIDE_MARGIN = 0.12
ENSEMBLE_WEIGHT_CNN = 0.7
THRESH = 0.40

# ---------------- INIT APP ----------------
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# ---------------- LOAD MODELS ----------------
ann = load_model(ANN_PATH)
cnn = load_model(CNN_PATH)
yolo = YOLO(YOLO_PATH)

print("✅ Models loaded")

# ---------------- HELPERS ----------------
def clamp(v, lo, hi):
    return max(lo, min(hi, v))

# ---------------- ROUTES ----------------
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    output_image = None

    if request.method == 'POST':
        file = request.files.get('file')
        if not file or file.filename == '':
            return render_template('index.html', result="No file selected")

        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        img = cv2.imread(filepath)
        orig = img.copy()

        results = yolo(filepath)
        boxes = results[0].boxes.xyxy.cpu().numpy()

        no_id_count = 0

        for bbox in boxes:
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
            if chest.size == 0:
                chest = img[y1:y2, x1:x2]

            chest = cv2.resize(chest, INPUT_SIZE)
            x = image.img_to_array(chest) / 255.0
            x = np.expand_dims(x, axis=0)

            ann_pred = ann.predict(x, verbose=0)[0][0]
            cnn_pred = cnn.predict(x, verbose=0)[0][0]
            score = ENSEMBLE_WEIGHT_CNN * cnn_pred + (1 - ENSEMBLE_WEIGHT_CNN) * ann_pred

            label = "With ID" if score < THRESH else "Without ID"
            color = (0, 255, 0) if label == "With ID" else (0, 0, 255)

            if label == "Without ID":
                no_id_count += 1

            cv2.rectangle(orig, (x1, y1), (x2, y2), color, 2)
            cv2.putText(orig, f"{label} ({score:.2f})",
                        (x1, max(20, y1-10)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        output_name = "output_" + filename
        output_path = os.path.join(OUTPUT_FOLDER, output_name)
        cv2.imwrite(output_path, orig)

        result = f"{no_id_count} person(s) detected without ID card"
        output_image = output_name

    return render_template('index.html', result=result, output_image=output_image)

# ---------------- RUN ----------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
