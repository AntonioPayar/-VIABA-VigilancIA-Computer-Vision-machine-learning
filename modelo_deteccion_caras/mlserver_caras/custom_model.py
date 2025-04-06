import os
import numpy as np
import base64
from PIL import Image
from io import BytesIO
from typing import Dict, List, Any
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.errors import InferenceError
import json

import cv2
from sklearn.metrics.pairwise import cosine_similarity


class FacesDetectionModel(MLModel):
    
    async def load(self) -> bool:
        # Create the detector instance directly
        self.model = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

        # No need to load it now - will load on first prediction
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        try:
            # Extract images from payload
            inputs = payload.inputs[0]
            image_input = inputs.data[0]

            # Convertir la imagen de entrada a un objeto PIL Image
            if isinstance(image_input, str) and image_input.startswith("data:image"):
                base64_data = image_input.split(",")[1]
                img_bytes = base64.b64decode(base64_data)
                image = Image.open(BytesIO(img_bytes))
            elif isinstance(image_input, np.ndarray):
                image = Image.fromarray(image_input.astype('uint8'))
            else:
                raise ValueError("Formato de imagen no soportado")

            # Get predictions
            image_data = np.array(image)
            gray = cv2.cvtColor(image_data, cv2.COLOR_BGR2GRAY)
            predictions = self.model.detectMultiScale(gray)
            face_i = predictions[0]
            face_vector = face_i.flatten() 
            
            # Format response
            return InferenceResponse(
                model_name=self.name,
                model_version=self._settings.parameters.version,
                outputs=[
                    ResponseOutput(
                        name="detections_caras",
                        shape=[1],
                        datatype="BYTES",
                        data=[json.dumps(face_vector.tolist())],
                    )
                ],
            )
        except Exception as e:
            raise InferenceError(f"Error during inference: {str(e)}")

