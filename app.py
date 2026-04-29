from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import numpy as np
import tflite_runtime.interpreter as tflite
import json
import io
import os
import requests as req_lib

app = Flask(__name__)
CORS(app)

# ── Charger le modèle et les classes au démarrage ──────────────────
MODEL_PATH      = 'model/skin_model.tflite'
CLASS_INFO_PATH = 'class_info.json'

print("⏳ Chargement du modèle...")
interpreter = tflite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details  = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(CLASS_INFO_PATH) as f:
    class_info = json.load(f)

CLASS_NAMES = class_info['class_names']
IMAGE_SIZE  = tuple(class_info['image_size'])  # (256, 256)

print(f"✅ Modèle chargé — Classes : {CLASS_NAMES}")


# ── Fonctions utilitaires ───────────────────────────────────────────
def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    image = image.resize(IMAGE_SIZE)
    image = np.array(image, dtype='float32') / 255.0
    image = np.expand_dims(image, axis=0)  # [1, 256, 256, 3]
    return image

def run_inference(image):
    interpreter.set_tensor(input_details[0]['index'], image)
    interpreter.invoke()
    return interpreter.get_tensor(output_details[0]['index'])


# ── Routes ──────────────────────────────────────────────────────────
@app.route('/', methods=['GET'])
def home():
    return jsonify({
        "api":      "Skin Condition Classifier",
        "version":  "1.0",
        "classes":  CLASS_NAMES,
        "endpoint": "POST /predict  →  envoyer une image"
    })


@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "Aucune image reçue. Envoyer un fichier avec la clé 'image'"}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({"error": "Fichier vide"}), 400

    allowed = {'jpg', 'jpeg', 'png', 'webp'}
    ext = file.filename.rsplit('.', 1)[-1].lower()
    if ext not in allowed:
        return jsonify({"error": f"Format non supporté. Utiliser : {allowed}"}), 400

    try:
        image        = preprocess_image(file.read())
        predictions  = run_inference(image)
        probabilities = predictions[0].tolist()

        predicted_idx   = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(probabilities[predicted_idx] * 100, 2)

        all_probs = {
            CLASS_NAMES[i]: round(probabilities[i] * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "prediction":        predicted_class,
            "confidence":        f"{confidence}%",
            "all_probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/predict-url', methods=['POST'])
def predict_from_url():
    data = request.get_json()
    if not data or 'image_url' not in data:
        return jsonify({"error": "image_url manquant"}), 400

    try:
        response = req_lib.get(data['image_url'], timeout=10)
        response.raise_for_status()

        image         = preprocess_image(response.content)
        predictions   = run_inference(image)
        probabilities = predictions[0].tolist()

        predicted_idx   = int(np.argmax(probabilities))
        predicted_class = CLASS_NAMES[predicted_idx]
        confidence      = round(probabilities[predicted_idx] * 100, 2)

        all_probs = {
            CLASS_NAMES[i]: round(probabilities[i] * 100, 2)
            for i in range(len(CLASS_NAMES))
        }

        return jsonify({
            "prediction":        predicted_class,
            "confidence":        confidence,
            "confidence_str":    f"{confidence}%",
            "all_probabilities": all_probs
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok", "model_loaded": True})


# ── Lancement ───────────────────────────────────────────────────────
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', 'false').lower() == 'true'
    app.run(host='0.0.0.0', port=port, debug=debug)