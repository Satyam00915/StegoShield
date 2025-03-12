import os
import stat

preprocessed_folder = "C:/old/college/sem 6/Special Project/Project/StegoShield/dataset/audio/preprocessed"

for file_name in os.listdir(preprocessed_folder):
    file_path = os.path.join(preprocessed_folder, file_name)
    
    # Remove read-only flag
    os.chmod(file_path, stat.S_IWRITE)

print("✅ All files are now writable.")
