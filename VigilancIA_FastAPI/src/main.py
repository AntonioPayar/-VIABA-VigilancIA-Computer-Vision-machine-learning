from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import os

app = FastAPI()

# Obtiene la ruta absoluta de la carpeta estática
static_dir = os.path.join(os.path.dirname(__file__), "static")

# Verifica si la carpeta existe
if not os.path.isdir(static_dir):
    raise RuntimeError(f"Directory '{static_dir}' does not exist")

# Montar archivos estáticos correctamente
#http://127.0.0.1:8000/static/configuration_page.html
app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/health")
def read_root():
 return {"greeting":"Hello world"}
