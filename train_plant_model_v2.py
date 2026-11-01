
from pathlib import Path
import tensorflow as tf

BASE_DIR = Path("/mnt/c/Users/ELCOT/Desktop/PlantGuard-AI")
TRAIN_DIR = BASE_DIR / "plant_data" / "train"
VAL_DIR = BASE_DIR / "plant_data" / "validation"
MODEL_PATH = BASE_DIR / "models" / "plant_disease_model_v2.keras"

IMAGE_SIZE = (224, 224)
BATCH_SIZE = 16
SEED = 42

print("Loading training images...")

train_ds = tf.keras.utils.image_dataset_from_directory(
    TRAIN_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=SEED
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    VAL_DIR,
    image_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = train_ds.class_names
print("Classes:", class_names)

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)

print("Loading pretrained MobileNetV2...")

base_model = tf.keras.applications.MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False

inputs = tf.keras.Input(shape=(224, 224, 3))
x = tf.keras.applications.mobilenet_v2.preprocess_input(inputs)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(
    len(class_names), activation="softmax"
)(x)

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("\nStarting training...")

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=8
)

model.save(str(MODEL_PATH))

print("\nTraining completed!")
print("Candidate model saved to:", MODEL_PATH)
print("Class names:", class_names)
