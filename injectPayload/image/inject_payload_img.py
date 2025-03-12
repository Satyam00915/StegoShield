import os
import random
import numpy as np
import cv2
from stegano import lsb
import pandas as pd

# Paths
CLEAN_IMAGES_DIR = "dataset/images/resized"
STEGO_IMAGES_DIR = "dataset/images/stego"
LABELS_FILE = "dataset/image_labels.csv"

# Ensure output directory exists
os.makedirs(STEGO_IMAGES_DIR, exist_ok=True)

# Get all clean images
image_files = os.listdir(CLEAN_IMAGES_DIR)
random.shuffle(image_files)  # Shuffle for randomness

# Select half of the dataset for stego embedding
num_stego = len(image_files) // 2
stego_images = image_files[:num_stego]
clean_images = image_files[num_stego:]

# Function to generate random binary noise
def generate_random_binary_noise(size=1024):
    return os.urandom(size).hex()  # Convert to hexadecimal string

# Function to embed binary noise into images
def embed_binary_noise(image_path, output_path):
    try:
        binary_payload = generate_random_binary_noise()
        stego_img = lsb.hide(image_path, binary_payload)  # Hide noise in LSB
        stego_img.save(output_path)
    except Exception as e:
        print(f"Error processing {image_path}: {e}")

# Embed noise into selected images
for img in stego_images:
    input_path = os.path.join(CLEAN_IMAGES_DIR, img)
    output_path = os.path.join(STEGO_IMAGES_DIR, img)
    embed_binary_noise(input_path, output_path)

# Create labels file (0 for clean, 1 for stego)
data = []
for img in clean_images:
    data.append((img, 0))  # Clean image
for img in stego_images:
    data.append((img, 1))  # Stego image

df = pd.DataFrame(data, columns=["filename", "label"])
df.to_csv(LABELS_FILE, index=False)

print(f"✅ Stego dataset created with {num_stego} images containing random binary noise.")
print(f"✅ Labels saved in {LABELS_FILE}")
