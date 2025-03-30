import os
import shutil
import random

# Paths
clean_image_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\images\clean"
stego_image_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\images\stego"
output_dir = r"C:\old\college\sem 6\Special Project\Project\StegoShield\dataset\split_data\split_images"

# Train, Validation, Test split ratio
train_ratio = 0.7
val_ratio = 0.15
test_ratio = 0.15

# Function to split and move files
def split_and_move_files(source_path, category):
    files = os.listdir(source_path)
    random.shuffle(files)

    total_files = len(files)
    train_count = int(total_files * train_ratio)
    val_count = int(total_files * val_ratio)
    
    split_data = {
        "train": files[:train_count],
        "val": files[train_count:train_count + val_count],
        "test": files[train_count + val_count:]
    }

    for split, file_list in split_data.items():
        split_folder = os.path.join(output_dir, split, category)
        os.makedirs(split_folder, exist_ok=True)

        for file in file_list:
            shutil.move(os.path.join(source_path, file), os.path.join(split_folder, file))

# Splitting both clean and stego images
split_and_move_files(clean_image_path, "clean")
split_and_move_files(stego_image_path, "stego")

print("✅ Image files successfully split into train, val, and test folders!")
