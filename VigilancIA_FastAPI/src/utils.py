import numpy as np
#Imports de clases propias
from classes.Cordenadas_Configuracion import *

import cv2
import numpy as np
import json


# Función para guardar los datos
def guardar_datos(datos: DatosCamaras, archivo: str):
    with open(archivo, 'w') as f:
        json.dump(datos.model_dump(), f, indent=4)

# Función para cargar los datos
def cargar_datos(archivo: str) -> DatosCamaras:
    with open(archivo, 'r') as f:
        data = json.load(f)
    
    print("Datos cargados...")
    return DatosCamaras(**data)

def calcularPixelMapaHomografia_2(camara1, camara2):
    # Datos de las esquinas en cam1 y cam2 (x1, y1) -> (x2, y2)
    cam1_points = np.array([
        [float(camara1.esquinaSuperiorIzquierda.x), float(camara1.esquinaSuperiorIzquierda.y)],
        [float(camara1.esquinaSuperiorDerecha.x), float(camara1.esquinaSuperiorDerecha.y)],
        [float(camara1.esquinaInferiorIzquierda.x), float(camara1.esquinaInferiorIzquierda.y)],
        [float(camara1.esquinaInferiorDerecha.x), float(camara1.esquinaInferiorDerecha.y)]
    ], dtype=np.float32)

    cam2_points = np.array([
        [float(camara2.esquinaSuperiorIzquierda.x), float(camara2.esquinaSuperiorIzquierda.y)],
        [float(camara2.esquinaSuperiorDerecha.x), float(camara2.esquinaSuperiorDerecha.y)],
        [float(camara2.esquinaInferiorIzquierda.x), float(camara2.esquinaInferiorIzquierda.y)],
        [float(camara2.esquinaInferiorDerecha.x), float(camara2.esquinaInferiorDerecha.y)]
    ], dtype=np.float32)

    # Calcular la matriz de transformación de homografía
    H, _ = cv2.findHomography(cam1_points, cam2_points)

    # Transformar el punto deseado
    punto_origen = np.array([[140, 515]], dtype=np.float32)
    punto_origen = np.array([punto_origen])  # Convertir a la forma adecuada para la transformación

    punto_transformado = cv2.perspectiveTransform(punto_origen, H)

    x2, y2 = punto_transformado[0][0]
    
    print(f"Coordenada transformada en cam2 (Homografía): ({round(x2)}, {round(y2)})")
    return {"mensaje": f"Transformada en cam2 (Homografía): ({round(x2)}, {round(y2)})"}


def calcularPixelMapa(camara1: DatosCamaras, camara2: DatosCamaras,x1,y1):
    #Calculo de la distancia entre las esquinas superiores de las camaras
    print(str(round(float(camara1.esquinaSuperiorIzquierda.x)))+","+str(round(float(camara1.esquinaSuperiorIzquierda.y))))
    print(str(round(float(camara2.esquinaSuperiorIzquierda.x)))+","+str(round(float(camara2.esquinaSuperiorIzquierda.y))))

    # Datos de las esquinas en cam1 y cam2 (x1, y1) -> (x2, y2)
    cam1_points = np.array([
        [round(float(camara1.esquinaSuperiorIzquierda.x)),round(float(camara1.esquinaSuperiorIzquierda.y)), 1],
        [round(float(camara1.esquinaSuperiorDerecha.x)),round(float(camara1.esquinaSuperiorDerecha.y)), 1],
        [round(float(camara1.esquinaInferiorIzquierda.x)),round(float(camara1.esquinaInferiorIzquierda.y)), 1],
        [round(float(camara1.esquinaInferiorDerecha.x)),round(float(camara1.esquinaInferiorDerecha.y)), 1]
    ])

    cam2_points_x = np.array([round(float(camara2.esquinaSuperiorIzquierda.x)),
                              round(float(camara2.esquinaSuperiorDerecha.x)),
                              round(float(camara2.esquinaInferiorIzquierda.x)),
                              round(float(camara2.esquinaInferiorDerecha.x))])  # x2 valores
    
    cam2_points_y = np.array([round(float(camara2.esquinaSuperiorIzquierda.y)),
                              round(float(camara2.esquinaSuperiorDerecha.y)),
                              round(float(camara2.esquinaInferiorIzquierda.y)),
                              round(float(camara2.esquinaInferiorDerecha.y))])  # y2 valores
    
    sol_x = np.linalg.lstsq(cam1_points, cam2_points_x, rcond=None)[0]
    sol_y = np.linalg.lstsq(cam1_points, cam2_points_y, rcond=None)[0]

    # Calcular la nueva coordenada (x2, y2) para (x1, y1) = (529, 546)
    x2 = sol_x[0] * x1 + sol_x[1] * y1 + sol_x[2]
    y2 = sol_y[0] * x1 + sol_y[1] * y1 + sol_y[2]

    return {"x": round(x2) - 70,"y": round(y2)}