import os
import requests
import json
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

# Freesound API Key and URL
FREESOUND_API_KEY = os.getenv("FREESOUND_API_KEY")
FREESOUND_URL = os.getenv("FREESOUND_URL")

# Parameters
NUM_SOUNDS = 3000  # Total number of sounds to download
DOWNLOAD_FOLDER = "audio"
os.makedirs(DOWNLOAD_FOLDER, exist_ok=True)
TAGS = [
    "field-recording", "multisample", "drum", "loop", "ambient", "synth", "percussion", "electronic",
    "ambience", "sound", "voice", "music", "bass", "industrial", "water", "dark", "metal", "soundscape",
    "atmosphere", "effect", "samples", "weird", "beat", "soundtrack", "sci-fi", "birds", "fx", "alien",
    "underground", "foley", "horror", "sample", "game", "sfx", "piano", "glitch", "vocal", "guitar",
    "drone", "loopable", "snare", "city", "kick"
]

headers = {"Authorization": f"Token {FREESOUND_API_KEY}"}

def get_audio_links(tag, page_size=50):
    params = {
        "query": tag,
        "filter": "duration:[0 TO 120]",  # Keep audio under 2 minutes
        "sort": "downloads_desc",
        "fields": "id,name,previews",
        "page_size": page_size
    }
    response = requests.get(FREESOUND_URL, headers=headers, params=params)
    
    if response.status_code == 200:
        data = response.json()
        return [(sound["id"], sound["name"], sound["previews"].get("preview-hq-mp3")) for sound in data.get("results", [])]
    else:
        print(f"Failed to fetch audio for {tag}: {response.status_code}")
        return []

def download_audio(audio_id, audio_name, audio_url, count):
    if not audio_url:
        return 0
    try:
        response = requests.get(audio_url, stream=True)
        if response.status_code == 200:
            file_path = os.path.join(DOWNLOAD_FOLDER, f"audio_{count}.mp3")
            with open(file_path, 'wb') as file:
                for chunk in response.iter_content(1024):
                    file.write(chunk)
            return os.path.getsize(file_path)
    except Exception as e:
        print(f"❌ Failed to download {audio_name}: {e}")
    return 0

def main():
    count = 0
    for tag in TAGS:
        print(f"Fetching audio samples for: {tag}")
        audio_links = get_audio_links(tag)
        
        for audio_id, audio_name, audio_url in tqdm(audio_links, desc=f"Downloading {tag}"):
            if count >= NUM_SOUNDS:
                print("Download limit reached.")
                return
            
            download_audio(audio_id, audio_name, audio_url, count)
            count += 1
            print(f"✅ Downloaded {count}/{NUM_SOUNDS}")
    
    print("🎉 Audio Download Complete!")

if __name__ == "__main__":
    main()
