import os
import numpy as np
from typing import Dict, List, Any
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.errors import InferenceError
import json

# Import the detector class
from yolo_detector import YOLOObjectDetector


class ObjectDetectionModel(MLModel):
    async def load(self) -> bool:
        # Create the detector instance directly
        self.model = YOLOObjectDetector()

        # No need to load it now - will load on first prediction
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        try:
            # Extract images from payload
            inputs = payload.inputs[0]
            image_data = inputs.data

            # Get predictions
            predictions = self.model.predict(image_data)

            # Format response
            return InferenceResponse(
                model_name=self.name,
                model_version=self._settings.parameters.version,
                outputs=[
                    ResponseOutput(
                        name="detections",
                        shape=[len(predictions)],
                        datatype="BYTES",
                        data=[json.dumps(pred) for pred in predictions],
                    )
                ],
            )
        except Exception as e:
            raise InferenceError(f"Error during inference: {str(e)}")
