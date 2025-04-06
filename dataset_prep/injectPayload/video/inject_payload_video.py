import os
import cv2
import numpy as np
import torch
import random
from pydub import AudioSegment

# Define Paths
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
CLEAN_VIDEO_FOLDER = os.path.join(PROJECT_DIR, "dataset/videos/preprocessed")
STEGO_VIDEO_FOLDER = os.path.join(PROJECT_DIR, "dataset/videos/stego")
AUDIO_FOLDER = os.path.join(PROJECT_DIR, "dataset/videos/preprocessed_audio_from_videos")
LABELS_FILE = os.path.join(PROJECT_DIR, "dataset/videos/labels.csv")

# Ensure output directory exists
os.makedirs(STEGO_VIDEO_FOLDER, exist_ok=True)

# Sample Payloads
BINARY_PAYLOAD = bytes([random.randint(0, 255) for _ in range(512)])

# Function to extract frames from video
def extract_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    
    cap.release()
    return frames, fps

# Function to save frames back to video and add original audio
def save_video_with_audio(frames, output_video_path, audio_path, fps):
    height, width, layers = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # MP4-compatible codec
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame in frames:
        out.write(frame)
    
    out.release()  # Close video writer

    if os.path.exists(audio_path):
        try:
            # Load the original audio
            audio = AudioSegment.from_file(audio_path)
            output_with_audio = output_video_path.replace(".mp4", "_final.mp4")
            audio.export(output_with_audio, format="mp4")

            print(f"✅ Audio added to: {output_with_audio}")
        except Exception as e:
            print(f"❌ Error adding audio to {output_video_path}: {e}")
    else:
        print(f"⚠️ No audio file found for {output_video_path}. Video saved without audio.")

# Payload 1: Adversarial Perturbation
def embed_adversarial(frames):
    return [np.clip(frame + torch.randn(frame.shape).numpy() * 5, 0, 255).astype(np.uint8) for frame in frames]

# Payload 2: High-Frequency Noise
def embed_noise(frames):
    return [np.clip(frame + np.random.normal(0, 15, frame.shape).astype(np.uint8), 0, 255).astype(np.uint8) for frame in frames]

# Payload 3: Binary Data Injection (LSB)
def embed_binary(frames, payload):
    bin_payload = ''.join(format(byte, '08b') for byte in payload)
    index = 0
    binary_frames = []

    for frame in frames:
        frame_copy = frame.copy()
        for i in range(frame.shape[0]):
            for j in range(frame.shape[1]):
                for k in range(3):  # RGB channels
                    if index < len(bin_payload):
                        frame_copy[i, j, k] = (frame_copy[i, j, k] & 254) | int(bin_payload[index])
                        index += 1
        binary_frames.append(frame_copy)

    return binary_frames

# Payload 4: Frame Manipulation (Subtle Brightness Shift)
def embed_frame_manipulation(frames):
    return [cv2.convertScaleAbs(frame, alpha=1.1, beta=5) for frame in frames]

# Get video files
video_files = [f for f in os.listdir(CLEAN_VIDEO_FOLDER) if f.endswith(".mp4")]
random.shuffle(video_files)

labels = []
num_clean = len(video_files) // 2  # Ensure 50% of dataset is clean
num_stego = len(video_files) - num_clean  # Remaining 50% will have payloads

# Process videos
for i, file_name in enumerate(video_files):
    input_path = os.path.join(CLEAN_VIDEO_FOLDER, file_name)
    output_path = os.path.join(STEGO_VIDEO_FOLDER, file_name)
    audio_path = os.path.join(AUDIO_FOLDER, file_name.replace(".mp4", ".wav"))  # Adjust extension if needed

    frames, fps = extract_frames(input_path)

    try:
        if i < num_clean:
            # Keep clean video
            save_video_with_audio(frames, output_path, audio_path, fps)
            labels.append(f"{file_name},clean")
            print(f"🔹 Clean video saved: {output_path}")
        else:
            # Embed random payload
            payload_type = random.choice(["adversarial", "noise", "binary", "frame_manipulation"])

            if payload_type == "adversarial":
                modified_frames = embed_adversarial(frames)
                print(f"🔹 Adversarial payload embedded in: {output_path}")
            elif payload_type == "noise":
                modified_frames = embed_noise(frames)
                print(f"🔹 Noise payload embedded in: {output_path}")
            elif payload_type == "binary":
                modified_frames = embed_binary(frames, BINARY_PAYLOAD)
                print(f"🔹 Binary payload embedded in: {output_path}")
            elif payload_type == "frame_manipulation":
                modified_frames = embed_frame_manipulation(frames)
                print(f"🔹 Frame manipulation payload embedded in: {output_path}")

            save_video_with_audio(modified_frames, output_path, audio_path, fps)
            labels.append(f"{file_name},stego")

    except Exception as e:
        print(f"❌ Error processing {input_path}: {e}")

# Save Labels
with open(LABELS_FILE, "w") as f:
    f.write("filename,label\n")  # Column header changed to 'label'
    for label in labels:
        f.write(label + "\n")

print("✅ Video Steganography Embedding Complete!")
