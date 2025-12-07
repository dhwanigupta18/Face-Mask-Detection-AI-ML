from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import base64
import io
from PIL import Image
import os

app = Flask(__name__)

# Load your trained model
MODEL_PATH = os.path.join("models", "mask_detector.h5")
model = tf.keras.models.load_model(MODEL_PATH)

CLASS_NAMES = ["Mask", "No Mask"]  # as we used binary classifier

def predict_image_from_base64(data_url: str):
    # data_url format: "data:image/jpeg;base64,AAAA..."
    if "," in data_url:
        header, encoded = data_url.split(",", 1)
    else:
        encoded = data_url

    img_bytes = base64.b64decode(encoded)
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    img = img.resize((224, 224))

    img_arr = np.array(img).astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)  # shape (1, 224, 224, 3)

    pred = model.predict(img_arr)[0][0]  # sigmoid output

    if pred < 0.5:
        label = "Mask"
        confidence = float(1 - pred)
    else:
        label = "No Mask"
        confidence = float(pred)

    return label, confidence

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    if not data or "image" not in data:
        return jsonify({"error": "No image received"}), 400

    try:
        label, confidence = predict_image_from_base64(data["image"])
        return jsonify({"label": label, "confidence": round(confidence, 3)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # debug=True sirf local development ke liye
    app.run(host="0.0.0.0", port=5000, debug=True)
