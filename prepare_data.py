import os
import random
import shutil

# Original PlantVillage dataset
SOURCE_DIR = "dataset/PlantVillage-Dataset/raw/color"

# Our prepared dataset
TRAIN_DIR = "plant_data/train"
VALIDATION_DIR = "plant_data/validation"

# The 4 classes we selected
CLASSES = [
    "Tomato___healthy",
    "Tomato___Early_blight",
    "Potato___healthy",
    "Potato___Early_blight"
]

# Number of images for each class
TRAIN_COUNT = 120
VALIDATION_COUNT = 32

# Make the selection reproducible
random.seed(42)

for class_name in CLASSES:
    source_class = os.path.join(SOURCE_DIR, class_name)
    train_class = os.path.join(TRAIN_DIR, class_name)
    validation_class = os.path.join(VALIDATION_DIR, class_name)

    # Get image files
    images = [
        file for file in os.listdir(source_class)
        if file.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    # Shuffle images randomly
    random.shuffle(images)

    # Select images
    train_images = images[:TRAIN_COUNT]
    validation_images = images[TRAIN_COUNT:TRAIN_COUNT + VALIDATION_COUNT]

    # Copy training images
    for image in train_images:
        shutil.copy2(
            os.path.join(source_class, image),
            os.path.join(train_class, image)
        )

    # Copy validation images
    for image in validation_images:
        shutil.copy2(
            os.path.join(source_class, image),
            os.path.join(validation_class, image)
        )

    print(f"{class_name}:")
    print(f"  Training images: {len(train_images)}")
    print(f"  Validation images: {len(validation_images)}")

print("\nDataset preparation completed successfully!")