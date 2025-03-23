#Imports de librerias complejas de python
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

#Imports de clases propias
from classes.Cordenadas_Configuracion import *

#Imports de librerias sencillas de python
import os

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
 return {"greeting":"Hello world"}

# Endpoint para la página de configuración
#http://127.0.0.1:8000/static/configuration_page.html
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.post("/getConfiguration")
async def recibir_datos(datos: DatosCamaras):
    print(datos.model_dump())  # Imprimir el JSON recibido en la consola
    return {"mensaje": "Datos recibidos correctamente"}
