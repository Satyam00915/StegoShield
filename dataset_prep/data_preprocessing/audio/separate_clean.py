import os
import shutil
import csv

# Define paths
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
PREPROCESSED_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/preprocessed")
CLEAN_AUDIO_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/clean")
STEGO_AUDIO_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/stego")
LABELS_FILE = os.path.join(PROJECT_DIR, "dataset/audio/labels.csv")

# Create the clean audio folder if it doesn't exist
os.makedirs(CLEAN_AUDIO_FOLDER, exist_ok=True)

# Read labels from CSV
with open(LABELS_FILE, "r") as f:
    reader = csv.reader(f)
    next(reader)  # Skip header
    labels = {row[0]: row[1] for row in reader}  # {filename: label}

# Move clean files to the clean folder
for file_name, label in labels.items():
    source_path = os.path.join(PREPROCESSED_FOLDER, file_name)
    target_path = os.path.join(CLEAN_AUDIO_FOLDER, file_name)
    
    if label == "clean":
        if os.path.exists(source_path):
            shutil.move(source_path, target_path)
            print(f"✅ Moved clean file: {file_name} → clean/")
        else:
            print(f"⚠️ File not found: {source_path}")

print("✅ Clean audio separation complete!")
