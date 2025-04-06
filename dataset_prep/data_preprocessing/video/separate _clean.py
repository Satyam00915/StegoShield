import os
import shutil
import pandas as pd

# Define paths
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
STEGO_FOLDER = os.path.join(PROJECT_DIR, "dataset/videos/stego")
CLEAN_FOLDER = os.path.join(PROJECT_DIR, "dataset/videos/clean")
LABELS_FILE = os.path.join(PROJECT_DIR, "dataset/videos/labels.csv")

# Create clean folder if it doesn't exist
os.makedirs(CLEAN_FOLDER, exist_ok=True)

# Read labels.csv to find clean files
if os.path.exists(LABELS_FILE):
    df = pd.read_csv(LABELS_FILE)
    
    # Filter clean files
    clean_files = df[df["label"] == "clean"]["filename"].tolist()

    for file_name in clean_files:
        stego_path = os.path.join(STEGO_FOLDER, file_name)
        clean_path = os.path.join(CLEAN_FOLDER, file_name)
        
        if os.path.exists(stego_path):
            shutil.move(stego_path, clean_path)  # Move clean file
            print(f"✅ Moved: {file_name} → {CLEAN_FOLDER}")

    print("🎯 Clean files successfully separated and removed from the stego folder.")
else:
    print(f"❌ Labels file not found at {LABELS_FILE}")
