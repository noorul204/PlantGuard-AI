from pathlib import Path
import sys
import numpy as np
import tensorflow as tf

BASE_DIR = Path("/mnt/c/Users/ELCOT/Desktop/PlantGuard-AI")
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model_v2.keras"

if len(sys.argv) < 2:
    print("Usage: python predict_plant.py <image_path>")
    sys.exit(1)

IMAGE_PATH = Path(sys.argv[1])

if not MODEL_PATH.exists():
    print("Model file not found:", MODEL_PATH)
    sys.exit(1)

if not IMAGE_PATH.is_file():
    print("Image file not found:", IMAGE_PATH)
    sys.exit(1)

print("Loading trained model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("Preparing leaf image...")
image = tf.keras.utils.load_img(
    IMAGE_PATH,
    target_size=(224, 224)
)
image_array = tf.keras.utils.img_to_array(image)
image_array = np.expand_dims(image_array, axis=0)

print("Predicting plant disease...")
predictions = model.predict(image_array, verbose=0)[0]

class_names = [
    "Potato___Early_blight",
    "Potato___healthy",
    "Tomato___Early_blight",
    "Tomato___healthy",
]

predicted_index = int(np.argmax(predictions))
predicted_class = class_names[predicted_index]
confidence = float(predictions[predicted_index]) * 100

print("\n--- PlantGuard-AI Prediction ---")
print("Image:", IMAGE_PATH.name)
print("Result:", predicted_class.replace("___", " - ").replace("_", " "))
print(f"Model confidence: {confidence:.2f}%")

