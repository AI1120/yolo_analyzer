# download_model.py
import os
import requests
from tqdm import tqdm

# Create models directory
os.makedirs("models", exist_ok=True)

# Model files to download
files = {
    "open_clip_pytorch_model.bin": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_pytorch_model.bin",
    "open_clip_config.json": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/open_clip_config.json",
    "tokenizer.json": "https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K/resolve/main/tokenizer.json"
}

for filename, url in files.items():
    filepath = os.path.join("models", filename)
    if os.path.exists(filepath):
        print(f"✓ {filename} already exists")
        continue
    
    print(f"Downloading {filename}...")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    
    with open(filepath, 'wb') as f, tqdm(
        desc=filename,
        total=total_size,
        unit='iB',
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            size = f.write(chunk)
            bar.update(size)
    
    print(f"✓ {filename} downloaded")

print("\n✅ All model files downloaded to ./models/")