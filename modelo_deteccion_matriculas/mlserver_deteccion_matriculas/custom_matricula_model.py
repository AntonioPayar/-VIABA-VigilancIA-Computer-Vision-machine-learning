import json
from mlserver import MLModel
from mlserver.types import InferenceRequest, InferenceResponse, ResponseOutput
from mlserver.errors import InferenceError
from detectar_matricula import detectar_matricula

class MatriculaDetectionModel(MLModel):
    async def load(self) -> bool:
        # No es necesaria ninguna carga adicional aquí, ya que la función se encarga de todo
        self.ready = True
        return self.ready

    async def predict(self, payload: InferenceRequest) -> InferenceResponse:
        try:
            # Se asume que el input es una imagen, se toma el primer input y su primer dato
            inputs = payload.inputs[0]
            image_data = inputs.data[0]
            # Llamar a la función para detectar la matrícula
            matricula = detectar_matricula(image_data)
            return InferenceResponse(
                model_name=self.name,
                model_version=self._settings.parameters.version,
                outputs=[
                    ResponseOutput(
                        name="matricula",
                        shape=[1],
                        datatype="BYTES",
                        data=[json.dumps(matricula)]
                    )
                ]
            )
        except Exception as e:
            raise InferenceError(f"Error during inference: {str(e)}")
