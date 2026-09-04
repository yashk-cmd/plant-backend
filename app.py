"""
Plant Pathology Identification System — Backend API
-----------------------------------------------------
Wraps a trained Keras/TensorFlow CNN (ResNet / MobileNet / VGG16 /
EfficientNet trained on PlantVillage) behind a REST endpoint that the
AI Studio frontend can call.

Run locally:
    pip install -r requirements.txt
    python app.py

Then POST an image to: http://localhost:5000/predict
"""

import io
import os

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS
from PIL import Image
from tensorflow.keras.models import load_model

# ----------------------------------------------------------------------
# CONFIG — edit these to match your actual trained model
# ----------------------------------------------------------------------

MODEL_PATH = "plant_disease_model.h5"      # TODO: path to your saved model
IMAGE_SIZE = (224, 224)                    # TODO: match your model's input size
                                            # (EfficientNetB2/B3 etc. use larger sizes)

# TODO: replace with your actual class labels, in the exact order your
# model's output layer was trained with (check train_generator.class_indices
# or similar from your training script — order MUST match exactly).
CLASS_NAMES = [
    "Pepper__bell___Bacterial_spot",
    "Pepper__bell___healthy",
    "Potato___Early_blight",
    "Potato___Late_blight",
    "Potato___healthy",
    "Tomato_Bacterial_spot",
    "Tomato_Early_blight",
    "Tomato_Late_blight",
    "Tomato_Leaf_Mold",
    "Tomato_Septoria_leaf_spot",
    "Tomato_Spider_mites_Two_spotted_spider_mite",
    "Tomato__Target_Spot",
    "Tomato__Tomato_YellowLeaf__Curl_Virus",
    "Tomato__Tomato_mosaic_virus",
    "Tomato_healthy",
]

# Below this confidence, we refuse to give a diagnosis rather than
# guess — this is your main defense against random/non-leaf images,
# since the model itself has no "not a leaf" class.
CONFIDENCE_THRESHOLD = 0.70

# ----------------------------------------------------------------------

app = Flask(__name__)
CORS(app)  # allow requests from your AI Studio frontend's domain

print("Loading model...")
model = load_model(MODEL_PATH)
print("Model loaded.")


def allowed_file(filename: str) -> bool:
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    return ext in {"jpg", "jpeg"}


def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """Load, resize, and normalize the image the same way it was
    preprocessed during training. Adjust the normalization line below
    to match your training pipeline (e.g. rescale=1./255, or the
    specific preprocess_input() for ResNet/MobileNet/EfficientNet)."""
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMAGE_SIZE)
    arr = np.array(img).astype("float32")

    # TODO: match this to your training preprocessing exactly.
    # Example for simple rescale:
    arr = arr / 255.0
    # Example if you used a Keras applications preprocess_input instead:
    # from tensorflow.keras.applications.resnet50 import preprocess_input
    # arr = preprocess_input(arr)

    return np.expand_dims(arr, axis=0)


def is_likely_leaf_image(arr: np.ndarray) -> bool:
    """Lightweight placeholder guardrail to catch obviously non-leaf
    images before they hit the disease model. This is NOT a trained
    classifier — it's a rough heuristic (dominant green/vegetation
    color ratio) to filter out extreme non-plant cases.

    For a real solution, train a small binary "leaf vs not-leaf"
    classifier and swap it in here — the confidence threshold on the
    main model is your primary and more reliable guardrail.
    """
    img = (arr[0] * 255).astype("uint8")
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    green_dominant = (g.astype(int) > r.astype(int)) & (g.astype(int) > b.astype(int))
    green_ratio = green_dominant.mean()
    return green_ratio > 0.15  # very loose threshold, tune or replace


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "No file selected"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Only JPG/JPEG files are supported"}), 400

    try:
        file_bytes = file.read()
        processed = preprocess_image(file_bytes)
    except Exception:
        return jsonify({"error": "Could not read image file"}), 400

    if not is_likely_leaf_image(processed):
        return jsonify({
            "result": "rejected",
            "message": "Unable to detect a plant leaf in this image. "
                        "Please upload a clear photo of a single leaf."
        }), 200

    predictions = model.predict(processed)[0]
    top_idx = int(np.argmax(predictions))
    confidence = float(predictions[top_idx])

    if confidence < CONFIDENCE_THRESHOLD:
        return jsonify({
            "result": "rejected",
            "message": "Unable to confidently identify a plant disease "
                        "in this image. Please upload a clearer photo "
                        "of a single leaf.",
            "confidence": round(confidence * 100, 2)
        }), 200

    disease_name = CLASS_NAMES[top_idx]

    return jsonify({
        "result": "success",
        "disease": disease_name,
        "confidence": round(confidence * 100, 2),
        # TODO: pull from a dictionary of descriptions per class if you
        # want a short explanation shown alongside the result
        "description": f"Detected: {disease_name.replace('___', ' - ').replace('_', ' ')}"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
