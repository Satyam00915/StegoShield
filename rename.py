import os  

folder_path = r"C:\old\college\sem 6\Special Project\Project\StegoShield\videos"  # Use raw string (r"") for Windows paths

files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]  

for count, filename in enumerate(files, start=0):  # Start count from 2082
    old_path = os.path.join(folder_path, filename)  
    new_extension = filename.split('.')[-1]  # Preserve file extension  
    new_name = f"video_{count}.{new_extension}"  
    new_path = os.path.join(folder_path, new_name)  

    try:
        os.rename(old_path, new_path)  
    except OSError as e:
        print(f"Error renaming {old_path}: {e}")

print("Renaming complete!")  
