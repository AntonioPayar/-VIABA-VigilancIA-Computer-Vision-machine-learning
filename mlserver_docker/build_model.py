import joblib
from yolo_detector import YOLOObjectDetector
import os
import urllib.request

# Download required files if they don't exist


def download_files():
    files = {
        "yolov3-tiny.weights": "https://pjreddie.com/media/files/yolov3-tiny.weights",
        "yolov3-tiny.cfg": "https://raw.githubusercontent.com/pjreddie/darknet/master/cfg/yolov3-tiny.cfg",
        "coco.names": "https://raw.githubusercontent.com/pjreddie/darknet/master/data/coco.names"
    }

    for filename, url in files.items():
        if not os.path.exists(filename):
            print(f"Downloading {filename}...")
            urllib.request.urlretrieve(url, filename)
            print(f"Downloaded {filename}")


# Download model files
download_files()

# Create and save the model
model = YOLOObjectDetector()
joblib.dump(model, "model.joblib")
print("Object detection model saved as model.joblib")
