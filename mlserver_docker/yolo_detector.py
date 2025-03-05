import numpy as np
from PIL import Image
from io import BytesIO
import base64
import os
import cv2


class YOLOObjectDetector:
    def __init__(self):
        # Initialize empty variables - we'll load the model in load_model()
        self.model = None
        self.classes = []
        self.output_layers = []

    def load_model(self):
        # Get paths to model files in the MLServer models directory
        models_dir = "/app/models/object-detection"
        weights_path = os.path.join(models_dir, "yolov3-tiny.weights")
        config_path = os.path.join(models_dir, "yolov3-tiny.cfg")
        names_path = os.path.join(models_dir, "coco.names")

        # Load the model
        self.model = cv2.dnn.readNet(weights_path, config_path)

        # Load class names
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]

        # Get output layer names
        self.output_layers = self.model.getUnconnectedOutLayersNames()

    def predict(self, X):
        """
        Predict objects in images.
        X: List of base64 encoded images or numpy arrays
        Returns: List of detections for each image
        """
        # Make sure model is loaded
        if self.model is None:
            self.load_model()

        results = []

        for img_data in X:
            # Convert input to OpenCV format
            if isinstance(img_data, str) and img_data.startswith('data:image'):
                # Extract base64 part
                base64_data = img_data.split(',')[1]
                img_bytes = base64.b64decode(base64_data)
                nparr = np.frombuffer(img_bytes, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            elif isinstance(img_data, np.ndarray):
                img = img_data

            # Detect objects
            height, width, _ = img.shape
            blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.model.setInput(blob)
            outs = self.model.forward(self.output_layers)

            # Process outputs
            class_ids = []
            confidences = []
            boxes = []

            for out in outs:
                for detection in out:
                    scores = detection[5:]
                    class_id = np.argmax(scores)
                    confidence = scores[class_id]

                    if confidence > 0.5:
                        # Object detected
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)

                        # Rectangle coordinates
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)

                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)

            # Non-maximum suppression
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

            # Prepare detections
            detections = []
            if len(indices) > 0:
                for i in indices.flatten():
                    x, y, w, h = boxes[i]
                    class_name = self.classes[class_ids[i]]
                    confidence = confidences[i]

                    detections.append({
                        'bbox': [float(x), float(y), float(x + w), float(y + h)],
                        'confidence': float(confidence),
                        'class': class_name
                    })

            results.append(detections)

        return results
