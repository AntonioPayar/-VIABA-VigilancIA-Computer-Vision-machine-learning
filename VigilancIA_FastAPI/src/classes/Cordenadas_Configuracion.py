from pydantic import BaseModel

class CoordenadaXY(BaseModel):
    x: str
    y: str

class Coordenadas(BaseModel):
    esquinaSuperiorIzquierda: CoordenadaXY
    esquinaSuperiorDerecha: CoordenadaXY
    esquinaInferiorIzquierda: CoordenadaXY
    esquinaInferiorDerecha: CoordenadaXY

class DatosCamaras(BaseModel):
    camara1: Coordenadas
    camara2: Coordenadas

class ImagenBase64(BaseModel):
    image_base64: str