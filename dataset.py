from huggingface_hub import login
import json
import os

# Download and extract COCO dataset
import os
import requests
import zipfile

# Replace 'YOUR_TOKEN' with your actual Hugging Face token
login(token="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")

coco_dir = "/coco"
annotations_dir = os.path.join(coco_dir, "annotations")
val2017_dir = os.path.join(coco_dir, "val2017")

# Create directories if they don't exist
# os.makedirs(annotations_dir, exist_ok=True)
# os.makedirs(val2017_dir, exist_ok=True)

def download_file(url, dest):
    response = requests.get(url, stream=True)
    with open(dest, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)

# Download COCO 2017 Validation images
val_zip_path = os.path.join(coco_dir, "val2017.zip")
annotations_zip_path = os.path.join(coco_dir, "annotations_trainval2017.zip")

download_file("http://images.cocodataset.org/zips/val2017.zip", val_zip_path)
download_file("http://images.cocodataset.org/annotations/annotations_trainval2017.zip", annotations_zip_path)

# Unzip the downloaded files
with zipfile.ZipFile(val_zip_path, 'r') as zip_ref:
    zip_ref.extractall(coco_dir)
with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
    zip_ref.extractall(coco_dir)

print("COCO dataset downloaded and extracted.")