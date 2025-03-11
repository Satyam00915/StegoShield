import os
import requests
import json
from tqdm import tqdm

# Pexels API Key (Replace with your own API key)
PEXELS_API_KEY = "72Wi1nHfwqZhaPwnxGMID0Eohinx62ISlsZ94Hr7ZdgureykwNVUv99D"
PEXELS_URL = "https://api.pexels.com/videos/search"

# Parameters
NUM_VIDEOS = 1000
MAX_STORAGE = 10 * 1024 * 1024 * 1024  # 10GB
VIDEO_DURATION_LIMIT = 120  # 2 minutes
DOWNLOAD_FOLDER = "video3"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)

def get_video_links(query="nature", per_page=80):
    headers = {"Authorization": PEXELS_API_KEY}
    params = {"query": query, "per_page": per_page}
    response = requests.get(PEXELS_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        video_links = []
        
        for video in data.get("videos", []):
            if video["duration"] <= VIDEO_DURATION_LIMIT:
                for file in video["video_files"]:
                    if file["quality"] == "hd":
                        video_links.append(file["link"])
                        break
        return video_links
    else:
        print(f"Failed to fetch videos: {response.status_code}")
        return []

def download_video(url, count):
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(DOWNLOAD_FOLDER, f"video_{count}.mp4")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return os.path.getsize(file_path)
    except Exception as e:
        print(f"❌ Failed to download {url}: {e}")
    return 0

def main():
    total_size = 0
    count = 0
    queries = ["Minimalist", "Cyberpunk", "Space", "Rain", "Shadows", "Loop", "Reflection", "Bokeh", "Silhouettes"]
    
    for query in queries:
        print(f"Fetching videos for: {query}")
        video_links = get_video_links(query)
        
        for url in tqdm(video_links, desc=f"Downloading {query}"):
            if count >= NUM_VIDEOS or total_size >= MAX_STORAGE:
                print("Download limit reached.")
                return
            
            video_size = download_video(url, count)
            total_size += video_size
            count += 1
            
            print(f"✅ Downloaded {count}/{NUM_VIDEOS}, Total Size: {total_size / (1024 * 1024 * 1024):.2f} GB")
    
    print("🎉 Download complete!")

if __name__ == "__main__":
    main()
