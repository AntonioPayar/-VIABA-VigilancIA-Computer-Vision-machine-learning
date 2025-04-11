# Imports de librerias complejas de python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import numpy as np

# Imports de clases propias
from classes.Cordenadas_Configuracion import *
from utils import *

# Imports de librerias sencillas de python
import os
import base64
import io

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite cualquier origen (frontend)
    allow_methods=["*"],  # Permite todos los métodos (GET, POST, etc.)
    allow_headers=["*"],  # Permite todas las cabeceras
)

# Obtiene la ruta absoluta de la carpeta estática
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")


@app.get("/health")
def read_root():
    return {"greeting": "Hello world"}


# Endpoint para la página de configuración
# http://127.0.0.1:8000/static/configuration_page.html
# http://127.0.0.1:8000/static/monitoring_page.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/setConfiguration")
async def recibir_datos(datos: DatosCamaras):
    # Guardar los datos en un archivo
    guardar_datos(datos, "datos_camaras.json")
    return calcularPixelMapaHomografia(datos.camara1, datos.camara2, 0, 0)


@app.post("/getPixelPlano")
async def recibir_datos_pixel_camara01(puntos: CoordenadaXY):
    print("Puntos recibidos:")
    print(puntos.x)
    print(puntos.y)
    # Cargar los datos desde el archivo
    datos = cargar_datos("datos_camaras.json")
    return calcularPixelMapaHomografia(
        datos.camara1, datos.camara2, float(puntos.x), float(puntos.y)
    )


@app.post("/getCamaraBounding")
async def recibir_frame_camara01(data: ImagenBase64):
    # Decodificar el base64 y convertir a imagen PIL
    np_array = np.array(Image.open(io.BytesIO(base64.b64decode(data.image_base64))).convert("RGB"))
    print(np_array)
