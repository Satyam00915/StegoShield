import os
import random
import wave
import numpy as np
import torch
import stat
import csv

# Define paths
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
CLEAN_AUDIO_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/preprocessed")
STEGO_AUDIO_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/stego")
LABELS_FILE = os.path.join(PROJECT_DIR, "dataset/audio/labels.csv")

# Ensure output directory exists
os.makedirs(STEGO_AUDIO_FOLDER, exist_ok=True)

# Fix read-only permissions
for file_name in os.listdir(CLEAN_AUDIO_FOLDER):
    file_path = os.path.join(CLEAN_AUDIO_FOLDER, file_name)
    os.chmod(file_path, stat.S_IWRITE)  # Make file writable

# Sample payloads
TEXT_PAYLOAD = "HiddenStegoPayload"
BINARY_PAYLOAD = bytes([random.randint(0, 255) for _ in range(128)])  # Random binary payload

# Function to embed text payload using LSB
def embed_text_lsb(audio_path, output_path, payload):
    try:
        with wave.open(audio_path, "rb") as audio:
            params = audio.getparams()
            frames = audio.readframes(audio.getnframes())

        # Ensure writable array
        audio_data = np.frombuffer(frames, dtype=np.int16).copy()
        
        payload_bin = ''.join(format(ord(char), '08b') for char in payload)

        if len(payload_bin) > len(audio_data):  # Prevent overflow
            raise ValueError(f"Payload too large for {audio_path}")

        for i in range(len(payload_bin)):
            audio_data[i] = (audio_data[i] & ~1) | int(payload_bin[i])

        # Convert back to bytes properly
        audio_bytes = audio_data.astype(np.int16).tobytes()

        with wave.open(output_path, "wb") as stego_audio:
            stego_audio.setparams(params)
            stego_audio.writeframes(audio_bytes)  # Ensure proper alignment
        
        print(f"🔹 Text payload embedded in: {output_path}")
    except Exception as e:
        print(f"❌ Error embedding text payload in {audio_path}: {e}")

# Function to embed binary payload
def embed_binary(audio_path, output_path, payload):
    try:
        with wave.open(audio_path, "rb") as audio:
            params = audio.getparams()
            frames = audio.readframes(audio.getnframes())
        
        audio_data = bytearray(frames)
        payload_size = min(len(payload), len(audio_data))
        for i in range(payload_size):
            audio_data[i] ^= payload[i]
        
        with wave.open(output_path, "wb") as stego_audio:
            stego_audio.setparams(params)
            stego_audio.writeframes(audio_data)
        
        print(f"🔹 Binary payload embedded in: {output_path}")
    except Exception as e:
        print(f"❌ Error embedding binary payload in {audio_path}: {e}")

# Function to add high-frequency noise
def embed_noise(audio_path, output_path):
    try:
        with wave.open(audio_path, "rb") as audio:
            params = audio.getparams()
            frames = audio.readframes(audio.getnframes())

        # Ensure writable array
        audio_data = np.frombuffer(frames, dtype=np.int16).copy()
        
        noise = np.random.normal(0, 10, len(audio_data)).astype(np.int16)

        # Ensure audio does not overflow
        audio_data = np.clip(audio_data + noise, -32768, 32767)

        # Convert back properly
        audio_bytes = audio_data.astype(np.int16).tobytes()

        with wave.open(output_path, "wb") as stego_audio:
            stego_audio.setparams(params)
            stego_audio.writeframes(audio_bytes)  # Fix buffer alignment

        print(f"🔹 Noise payload embedded in: {output_path}")
    except Exception as e:
        print(f"❌ Error embedding noise payload in {audio_path}: {e}")

# Function to add adversarial perturbation
def embed_adversarial(audio_path, output_path):
    try:
        with wave.open(audio_path, "rb") as audio:
            params = audio.getparams()
            frames = audio.readframes(audio.getnframes())

        # Ensure correct dtype
        audio_data = np.frombuffer(frames, dtype=np.int16).copy().astype(np.float32)

        # Generate adversarial perturbation (small noise)
        perturbation = torch.randn(audio_data.shape) * 0.001

        # Apply perturbation and clip to prevent overflow
        audio_data = np.clip(audio_data + perturbation.numpy(), -32768, 32767).astype(np.int16)

        # Convert back to bytes properly
        audio_bytes = audio_data.tobytes()

        with wave.open(output_path, "wb") as stego_audio:
            stego_audio.setparams(params)
            stego_audio.writeframes(audio_bytes)

        print(f"🔹 Adversarial noise embedded in: {output_path}")
    except Exception as e:
        print(f"❌ Error embedding adversarial payload in {audio_path}: {e}")


# Get all WAV files and shuffle
audio_files = [f for f in os.listdir(CLEAN_AUDIO_FOLDER) if f.endswith(".wav")]
random.shuffle(audio_files)

# Process half of the dataset
half_dataset = len(audio_files) // 2
selected_files = audio_files[:half_dataset]

# List to store labels
labels = []

# Mark original files as clean
for file_name in audio_files:
    labels.append((file_name, "clean"))

# Embed payloads and mark processed files as stego
for file_name in selected_files:
    input_path = os.path.join(CLEAN_AUDIO_FOLDER, file_name)
    output_path = os.path.join(STEGO_AUDIO_FOLDER, file_name)
    
    # Choose a random payload type
    payload_type = random.choice(["text", "binary", "noise", "adversarial"])
    
    if payload_type == "text":
        embed_text_lsb(input_path, output_path, TEXT_PAYLOAD)
    elif payload_type == "binary":
        embed_binary(input_path, output_path, BINARY_PAYLOAD)
    elif payload_type == "noise":
        embed_noise(input_path, output_path)
    elif payload_type == "adversarial":
        embed_adversarial(input_path, output_path)
    
    labels.append((file_name, "stego"))

# Save labels to CSV
with open(LABELS_FILE, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "label"])
    writer.writerows(labels)

print("✅ Steganography embedding complete!")
print("✅ Labels file created:", LABELS_FILE)
