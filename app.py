from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# --- Initialize Flask ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# --- Load your trained model ---
model = load_model('models/cnn_model.h5')

# --- Define routes ---
@app.route('/', methods=['GET', 'POST'])
def index():
    result = None
    filename = None

    if request.method == 'POST':
        file = request.files['file']
        if file:
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            file.save(filepath)

            # --- Preprocess Image ---
            img = image.load_img(filepath, target_size=(128,128))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0) / 255.0

            # --- Predict ---
            pred = model.predict(x)
            label = "With ID" if pred[0][0] < 0.5 else "Without ID"
            result = "✅ Person is wearing ID card" if label == "With ID" else "🚫 Person is not wearing ID card"

    return render_template('index.html', filename=filename, result=result)

# --- Run the app ---
if __name__ == "__main__":
    app.run(debug=True)
