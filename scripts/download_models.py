# scripts/download_models.py
import os
import requests

# Create models folder if it does not exist
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# URLs for model files
model_files = {
    "deploy.prototxt": "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
    "res10_300x300_ssd_iter_140000.caffemodel": "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
    "haarcascade_frontalface_default.xml": "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml",
}

# Download each file
for name, url in model_files.items():
    save_path = os.path.join(MODELS_DIR, name)
    print(f"Downloading {name}...")

    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(save_path, "wb") as file:
            file.write(response.content)

        print(f"Saved to {save_path}")

    except Exception as error:
        print(f"Failed to download {name}: {error}")

print("All model files downloaded successfully.")
