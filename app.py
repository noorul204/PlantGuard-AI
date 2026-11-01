from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request, render_template
from PIL import Image

app = Flask(__name__)

MODEL_PATH = Path(__file__).resolve().parent / "models" / "plant_disease_model_v2.keras"

CLASS_NAMES = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
]

DISEASE_INFO = {
    "Potato___Early_blight": {
        "title": "Potato - Early Blight",
        "about": "Early blight is commonly caused by the fungus Alternaria solani. It can cause dark spots on potato leaves.",
        "symptoms": [
            "Dark brown spots on older leaves.",
            "Spots may have concentric, ring-like patterns.",
            "Yellowing around affected areas."
        ],
        "prevention": [
    "Avoid wetting leaves while watering.",
    "Provide enough spacing for good airflow.",
    "Remove infected plant debris.",
    "Use healthy seed potatoes and rotate crops where possible."
],
"care_recommendations": [
    "Remove badly affected leaves carefully.",
    "Avoid watering the leaves directly.",
    "Keep the plant area clean.",
    "Monitor nearby potato plants for similar symptoms."
]
    },
    "Potato___healthy": {
        "title": "Potato - Healthy",
        "about": "The model classified this image as a healthy potato leaf. This does not guarantee that the entire plant is disease-free.",
        "symptoms": [
            "No obvious early-blight symptoms detected by the model."
        ],
        "prevention": [
    "Water appropriately and avoid prolonged leaf wetness.",
    "Maintain good airflow between plants.",
    "Inspect leaves regularly for spots or yellowing.",
    "Keep the growing area free of infected plant debris."
],
"care_recommendations": [
    "Continue regular watering without waterlogging the soil.",
    "Ensure the plant receives adequate sunlight.",
    "Inspect leaves regularly for changes.",
    "Maintain proper spacing between plants."
]
    },
    "Tomato___Early_blight": {
        "title": "Tomato - Early Blight",
        "about": "Early blight is commonly associated with Alternaria solani. It can affect tomato leaves, stems, and sometimes fruit.",
        "symptoms": [
            "Dark spots, often with ring-like patterns.",
            "Yellowing around the spots.",
            "Older leaves may turn yellow and dry out."
        ],
        "prevention": [
    "Water at the base of the plant when possible.",
    "Leave enough space between plants for airflow.",
    "Remove infected leaves and plant debris carefully.",
    "Rotate crops where practical."
],
"care_recommendations": [
    "Remove severely affected lower leaves carefully.",
    "Water at the base of the plant.",
    "Avoid handling healthy plants immediately after infected plants.",
    "Monitor the plant for spreading spots."
]
    },
    "Tomato___healthy": {
        "title": "Tomato - Healthy",
        "about": "The model classified this image as a healthy tomato leaf. Other diseases or problems may still be present.",
        "symptoms": [
            "No obvious early-blight symptoms detected by the model."
        ],
        "prevention": [
    "Water near the soil rather than over the leaves.",
    "Provide good spacing and airflow.",
    "Check plants regularly for spots, yellowing, or wilting.",
    "Maintain good garden hygiene."
],
"care_recommendations": [
    "Continue regular watering at the soil level.",
    "Provide adequate sunlight and airflow.",
    "Support the plant as it grows.",
    "Inspect leaves and stems regularly."
]
    }
}

model = None


def load_model():
    global model

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

    model = tf.keras.models.load_model(str(MODEL_PATH))


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "Please upload an image."}), 400

    uploaded_file = request.files["image"]

    if not uploaded_file.filename:
        return jsonify({"error": "No image selected."}), 400

    try:
        with Image.open(uploaded_file.stream) as img:
            img = img.convert("RGB").resize((224, 224))
            image_array = np.asarray(img, dtype=np.float32)

        image_array = np.expand_dims(image_array, axis=0)
        predictions = model.predict(image_array, verbose=0)[0]
        predicted_index = int(np.argmax(predictions))
        predicted_class = CLASS_NAMES[predicted_index]

        return jsonify({
            "predicted_class": predicted_class,
            "confidence": round(float(predictions[predicted_index]) * 100, 2),
            "message": "AI prediction only; verify the result independently.",
            "disease_info": DISEASE_INFO[predicted_class]
        })

    except Exception as error:
        print(f"Prediction error: {error}")
        return jsonify({
            "error": "Could not process this image. Please upload a valid plant-leaf image."
        }), 400


if __name__ == "__main__":
    load_model()
    print("PlantGuard-AI model loaded successfully!")
    app.run(host="0.0.0.0", port=5000, debug=False)
