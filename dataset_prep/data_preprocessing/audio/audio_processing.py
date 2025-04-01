import os
import subprocess
import imageio_ffmpeg

# Auto-detect FFmpeg binary
FFMPEG_PATH = imageio_ffmpeg.get_ffmpeg_exe()

# Define paths based on your dataset structure
PROJECT_DIR = "C:/old/college/sem 6/Special Project/Project/StegoShield"
INPUT_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/clean")
FIXED_MP3_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/fixed_mp3")
CONVERTED_WAV_FOLDER = os.path.join(PROJECT_DIR, "dataset/audio/preprocessed")

# Ensure output directories exist
os.makedirs(FIXED_MP3_FOLDER, exist_ok=True)
os.makedirs(CONVERTED_WAV_FOLDER, exist_ok=True)

# Function to fix MP3 metadata
def fix_mp3_metadata(input_file, output_file):
    try:
        subprocess.run([
            FFMPEG_PATH, "-i", input_file,
            "-map_metadata", "-1", "-c:v", "copy", "-c:a", "copy",
            output_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✅ Fixed metadata: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error fixing {input_file}: {e}")

# Function to convert MP3 to WAV (16kHz, mono)
def convert_mp3_to_wav(input_file, output_file):
    try:
        subprocess.run([
            FFMPEG_PATH, "-i", input_file,
            "-ar", "16000", "-ac", "1", output_file
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"🎵 Converted: {input_file} -> {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error converting {input_file}: {e}")

# Process all MP3 files in the clean audio folder
for file_name in os.listdir(INPUT_FOLDER):
    if file_name.lower().endswith(".mp3"):
        input_path = os.path.join(INPUT_FOLDER, file_name)
        fixed_mp3_path = os.path.join(FIXED_MP3_FOLDER, file_name)
        converted_wav_path = os.path.join(CONVERTED_WAV_FOLDER, os.path.splitext(file_name)[0] + ".wav")

        # Fix metadata
        fix_mp3_metadata(input_path, fixed_mp3_path)

        # Convert to WAV
        convert_mp3_to_wav(fixed_mp3_path, converted_wav_path)

print("✅ Audio preprocessing complete!")
