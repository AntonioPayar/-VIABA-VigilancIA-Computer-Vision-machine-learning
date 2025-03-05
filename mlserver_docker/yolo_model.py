import torch
import joblib
import numpy as np
import cv2
from PIL import Image
from io import BytesIO
import base64


class YOLOObjectDetector:
    def __init__(self):
        # Load a pre-trained YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
        # Set to evaluation mode
        self.model.eval()

    def predict(self, X):
        """
        Predict objects in images.
        X: List of base64 encoded images or numpy arrays
        Returns: List of detections for each image
        """
        images = []
        for img_data in X:
            # Handle base64 encoded images
            if isinstance(img_data, str) and img_data.startswith('data:image'):
                # Extract the base64 part
                base64_data = img_data.split(',')[1]
                img = Image.open(BytesIO(base64.b64decode(base64_data)))
                images.append(img)
            # Handle numpy arrays
            elif isinstance(img_data, np.ndarray):
                img = Image.fromarray(img_data.astype(np.uint8))
                images.append(img)
            # Handle file paths
            elif isinstance(img_data, str):
                img = Image.open(img_data)
                images.append(img)

        # Run inference
        results = self.model(images)

        # Process and return results
        output = []
        for result in results.xyxy:  # xyxy format is [x1, y1, x2, y2, confidence, class]
            detections = []
            for det in result:
                x1, y1, x2, y2, conf, cls = det.tolist()
                class_name = self.model.names[int(cls)]
                detections.append({
                    'bbox': [float(x1), float(y1), float(x2), float(y2)],
                    'confidence': float(conf),
                    'class': class_name
                })
            output.append(detections)

        return output


# Create and save the model
model = YOLOObjectDetector()
joblib.dump(model, "model.joblib")
print("YOLOv5 object detection model saved as model.joblib")
