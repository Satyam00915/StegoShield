import os
import cv2
import numpy as np
import random
import pandas as pd
from skimage.util import random_noise

# Define Paths
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
IMAGE_FOLDER = os.path.join(PROJECT_DIR, "dataset/images/preprocessed")
CLEAN_FOLDER = os.path.join(PROJECT_DIR, "dataset/images/clean")
STEGO_FOLDER = os.path.join(PROJECT_DIR, "dataset/images/stego")
LABELS_FILE = os.path.join(PROJECT_DIR, "dataset/images/labels.csv")

# Ensure output directories exist
os.makedirs(CLEAN_FOLDER, exist_ok=True)
os.makedirs(STEGO_FOLDER, exist_ok=True)

# Get all images
image_files = [f for f in os.listdir(IMAGE_FOLDER) if f.endswith(".png") or f.endswith(".jpg")]
random.shuffle(image_files)

# Define Payload Functions
def embed_lsb(image, payload):
    """Embeds binary payload in the least significant bit of an image."""
    bin_payload = ''.join(format(byte, '08b') for byte in payload)
    index = 0
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            for k in range(3):  # RGB channels
                if index < len(bin_payload):
                    image[i, j, k] = (image[i, j, k] & 254) | int(bin_payload[index])
                    index += 1
    return image

def embed_noise(image):
    """Adds high-frequency noise to the image."""
    return (random_noise(image, mode="gaussian", var=0.01) * 255).astype(np.uint8)

# Generate Stego Images and Labels
labels = []
for i, file_name in enumerate(image_files):
    image_path = os.path.join(IMAGE_FOLDER, file_name)
    image = cv2.imread(image_path)
    
    try:
        if i < len(image_files) // 2:
            # Save clean image
            output_path = os.path.join(CLEAN_FOLDER, file_name)
            cv2.imwrite(output_path, image)
            labels.append(f"{file_name},clean")
            print(f"✅ Clean image saved: {output_path}")
        else:
            # Apply a random steganographic technique
            payload = np.random.randint(0, 255, 512, dtype=np.uint8)
            method = random.choice(["lsb", "noise"])

            if method == "lsb":
                stego_image = embed_lsb(image, payload)
            elif method == "noise":
                stego_image = embed_noise(image)

            output_path = os.path.join(STEGO_FOLDER, file_name)
            cv2.imwrite(output_path, stego_image)
            labels.append(f"{file_name},stego")
            print(f"🔹 {method.upper()} payload embedded in: {output_path}")

    except Exception as e:
        print(f"❌ Error processing {file_name}: {e}")

# Save Labels
df = pd.DataFrame([l.split(",") for l in labels], columns=["filename", "label"])
df.to_csv(LABELS_FILE, index=False)

print("✅ Image Steganography Embedding Complete!")
