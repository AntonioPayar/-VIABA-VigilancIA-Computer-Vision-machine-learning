import os
import numpy as np
from typing import Dict, List, Any
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.errors import InferenceError
import json

import cv2
import tensorflow as tf
import tensorflow_hub as hub
from sklearn.metrics.pairwise import cosine_similarity


class ObjectDetectionModel(MLModel):
    
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
            image_data = inputs.data

            # Get predictions
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
                        shape=[len(face_vector)],
                        datatype="BYTES",
                        data=[json.dumps(face_vector) for pred in predictions],
                    )
                ],
            )
        except Exception as e:
            raise InferenceError(f"Error during inference: {str(e)}")