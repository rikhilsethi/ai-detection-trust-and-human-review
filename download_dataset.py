import requests
import zipfile
import os

print("Downloading COCO validation images...")

# Download COCO 2017 validation images (1GB - 5000 images)
url = "http://images.cocodataset.org/zips/val2017.zip"
response = requests.get(url, stream=True)

total = int(response.headers.get('content-length', 0))
downloaded = 0

with open("val2017.zip", "wb") as f:
    for chunk in response.iter_content(chunk_size=8192):
        f.write(chunk)
        downloaded += len(chunk)
        percent = (downloaded / total) * 100
        print(f"Progress: {percent:.1f}%", end="\r")

print("\nDownload complete. Extracting...")

with zipfile.ZipFile("val2017.zip", "r") as zip_ref:
    zip_ref.extractall(".")

print("Done! Images are in the val2017 folder.")