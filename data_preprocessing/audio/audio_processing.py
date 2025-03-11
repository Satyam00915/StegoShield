import os
import librosa
import soundfile as sf
import numpy as np

# Paths
input_folder = "dataset/audio/clean"
output_folder = "dataset/audio/processed"
os.makedirs(output_folder, exist_ok=True)

# Parameters
target_sample_rate = 16000  # 16 kHz
target_duration = 5  # seconds

def preprocess_audio(file_path, output_path):
    try:
        # Load audio
        audio, sr = librosa.load(file_path, sr=target_sample_rate)
        
        # Ensure fixed length (trim/pad)
        target_length = target_sample_rate * target_duration
        if len(audio) > target_length:
            audio = audio[:target_length]  # Trim
        else:
            audio = np.pad(audio, (0, target_length - len(audio)))  # Pad

        # Save processed file
        sf.write(output_path, audio, target_sample_rate)
    except Exception as e:
        print(f"Skipping {file_path}: {e}")

# Process all audio files
for file_name in os.listdir(input_folder):
    input_path = os.path.join(input_folder, file_name)
    output_path = os.path.join(output_folder, os.path.splitext(file_name)[0] + ".wav")
    preprocess_audio(input_path, output_path)

print(f"✅ Audio preprocessing complete. Processed files saved in '{output_folder}'.")
