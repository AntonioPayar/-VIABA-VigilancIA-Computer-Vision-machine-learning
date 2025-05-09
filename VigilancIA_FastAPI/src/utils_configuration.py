import numpy as np

# Imports de clases propias
from classes.Cordenadas_Configuracion import *

import cv2
import numpy as np
import json


# Función para guardar los datos
def guardar_datos(datos: DatosCamaras, archivo: str):
    with open(archivo, "w") as f:
        json.dump(datos.model_dump(), f, indent=4)


# Función para cargar los datos
def cargar_datos(archivo: str) -> DatosCamaras:
    with open(archivo, "r") as f:
        data = json.load(f)

    print("Datos cargados...")
    return DatosCamaras(**data)


def escalar_coordenadas(pts_originales, altura_original, altura_nueva):
    """
    Escala las coordenadas Y de un conjunto de puntos para cambiar la altura de la imagen.

    Args:
        pts_originales (np.array): Matriz NumPy de coordenadas (x, y) originales.
        altura_original (int): Altura original de la imagen.
        altura_nueva (int): Nueva altura de la imagen.

    Returns:
        np.array: Matriz NumPy de coordenadas (x, y) escaladas.
    """

    proporcion_escalado = altura_nueva / altura_original
    pts_escalados = (
        pts_originales.copy()
    )  # Crear una copia para no modificar el original
    pts_escalados[:, 1] = np.round(
        pts_originales[:, 1] * proporcion_escalado
    )  # Escalar la coordenada y.
    return pts_escalados


def calcularPixelMapaHomografia(camara1: DatosCamaras, camara2: DatosCamaras, x1, y1):
    pts_img1 = np.array(
        [
            [
                round(float(camara1.esquinaSuperiorIzquierda.x)),
                round(float(camara1.esquinaSuperiorIzquierda.y)),
            ],  # esquina superior izquierda
            [
                round(float(camara1.esquinaSuperiorDerecha.x)),
                round(float(camara1.esquinaSuperiorDerecha.y)),
            ],  # esquina superior derecha
            [
                round(float(camara1.esquinaInferiorDerecha.x)),
                round(float(camara1.esquinaInferiorDerecha.y)),
            ],  # esquina inferior derecha
            [
                round(float(camara1.esquinaInferiorIzquierda.x)),
                round(float(camara1.esquinaInferiorIzquierda.y)),
            ],  # esquina inferior izquierda
        ],
        dtype=np.float32,
    )

    pts_img2 = np.array(
        [
            [
                round(float(camara2.esquinaSuperiorIzquierda.x)),
                round(float(camara2.esquinaSuperiorIzquierda.y)),
            ],  # esquina superior izquierda
            [
                round(float(camara2.esquinaSuperiorDerecha.x)),
                round(float(camara2.esquinaSuperiorDerecha.y)),
            ],  # esquina superior derecha
            [
                round(float(camara2.esquinaInferiorDerecha.x)),
                round(float(camara2.esquinaInferiorDerecha.y)),
            ],  # esquina inferior derecha
            [
                round(float(camara2.esquinaInferiorIzquierda.x)),
                round(float(camara2.esquinaInferiorIzquierda.y)),
            ],  # esquina inferior izquierda
        ],
        dtype=np.float32,
    )

    # píxel (Objetivo) en imagen01:
    pixel = np.array([[[x1, y1]]], dtype=np.float32)
    # Calcular la homografía entre las dos imágenes
    H, _ = cv2.findHomography(pts_img1, pts_img2)

    # Transformar el pixel usando la homografía
    transformed_pixel = cv2.perspectiveTransform(pixel, H)
    new_x, new_y = transformed_pixel[0][0]
    return {"x": round(new_x), "y": round(new_y)}
