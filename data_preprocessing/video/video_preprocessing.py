import os
import cv2
from moviepy import VideoFileClip

# Define paths based on your dataset structure
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
input_folder = os.path.join(PROJECT_DIR, "dataset/videos/clean")
output_folder = os.path.join(PROJECT_DIR, "dataset/videos/processed")
audio_output_folder = os.path.join(PROJECT_DIR, "dataset/videos/processed_audio_from_videos")

os.makedirs(output_folder, exist_ok=True)
os.makedirs(audio_output_folder, exist_ok=True)

# Parameters
target_resolution = (256, 256)

def preprocess_video(file_path, output_path, extract_audio_path):
    try:
        # Extract audio using MoviePy
        video_clip = VideoFileClip(file_path)
        audio_clip = video_clip.audio
        if audio_clip:
            audio_clip.write_audiofile(extract_audio_path, codec="pcm_s16le", fps=16000)
        else:
            print(f"⚠ Warning: No audio found in {file_path}")

        # Process video frames using OpenCV
        cap = cv2.VideoCapture(file_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, cap.get(cv2.CAP_PROP_FPS), target_resolution)

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, target_resolution)  # Resize to 256x256
            out.write(frame)

        cap.release()
        out.release()
        video_clip.close()

        print(f"✅ Processed {file_path}")
        
    except Exception as e:
        print(f"❌ Error processing {file_path}: {e}")

# Process all videos
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_video_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".mp4")
    output_audio_path = os.path.join(audio_output_folder, os.path.splitext(file_name)[0] + ".wav")

    preprocess_video(input_path, output_video_path, output_audio_path)

print(f"✅ Video preprocessing complete. Processed files saved in '{output_folder}'.")
