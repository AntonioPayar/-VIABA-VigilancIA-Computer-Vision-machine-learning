import numpy as np
#Imports de clases propias
from classes.Cordenadas_Configuracion import *

def calcularPixelMapa(datos: DatosCamaras):
    cam1_points = np.array([
    [238, 462, 1],
    [775, 462, 1],
    [4, 615, 1],
    [1000, 611, 1]
])

    return {"mensaje": "Datos recibidos correctamente"}