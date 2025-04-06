from PIL import Image
import os

# Paths
input_folder = "dataset/images/clean"
output_folder = "dataset/images/resized"
os.makedirs(output_folder, exist_ok=True)

# Resize and preprocess images
for img_name in os.listdir(input_folder):
    img_path = os.path.join(input_folder, img_name)
    
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")  # Ensure RGB mode
            img = img.resize((256, 256))  # Resize to 256x256
            img.save(os.path.join(output_folder, img_name), format="PNG")  # Save as PNG
    except Exception as e:
        print(f"Skipping {img_name}: {e}")

print(f"✅ Images resized and saved in '{output_folder}'.")
