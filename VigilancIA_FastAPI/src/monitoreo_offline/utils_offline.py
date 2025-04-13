import torch
import numpy as np
import pandas as pd
from PIL import Image

# Imports de clases propias
from classes.Cordenadas_Configuracion import *
from classes.Clases_Detecciones import Persona
from monitoreo_offline.sort import Sort

import cv2

YOLO = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
SORT_TRACKER = Sort(max_age=20, min_hits=3, iou_threshold=0.2)
COLOR = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "yellow",
    "lime",
    "pink",
    "teal",
    "brown",
    "gray",
    "olive",
    "maroon",
    "navy",
]
LISTAS_PERSONAS = {}


def add_padding(np_array):
    original = np_array
    h, w, c = original.shape

    # Calculamos cuánto padding necesitamos
    padding_total = 640 - h
    padding_top = padding_total // 2
    padding_bottom = padding_total - padding_top

    # Agregamos padding con np.pad
    return np.pad(
        original,
        ((padding_top, padding_bottom), (0, 0), (0, 0)),
        mode="constant",
        constant_values=0,
    )


def preprocess_image_numpy(img_np):
    # Aseguramos que el tipo sea uint8, valores entre 0-255
    if img_np.dtype != np.uint8:
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)

    return img_np  # (H, W, 3)


def preprocess_image(img_np):
    img_preprocessed = img_np / 255.0  # Normalizamos
    img_preprocessed = np.transpose(img_preprocessed, (2, 0, 1))  # (HWC) -> (CHW)
    img_tensor = torch.tensor(img_preprocessed).float()
    return img_tensor.unsqueeze(
        0
    )  # Añadimos una dimensión extra para el batch (1, C, H, W)


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


def trackerar_detecciones(people_detections):
    dets_for_sort = []

    for _, row in people_detections.iterrows():
        x1, y1, x2, y2, conf = (
            float(row["xmin"]),
            float(row["ymin"]),
            float(row["xmax"]),
            float(row["ymax"]),
            float(row["confidence"]),
        )
        dets_for_sort.append([x1, y1, x2, y2, conf])

    dets_for_sort = np.array(dets_for_sort)

    if len(dets_for_sort) == 0:
        dets_for_sort = np.empty((0, 5))
    else:
        dets_for_sort = np.array(dets_for_sort)
    # Ejecutar el tracker
    tracked_objects = SORT_TRACKER.update(dets_for_sort)
    return tracked_objects


def detect_objects(camara1: DatosCamaras, camara2: DatosCamaras, image_tensor):
    # Realizamos la inferencia
    results = YOLO(image_tensor)
    # Obtener las bounding boxes, clases y scores
    detections = results.pandas().xyxy[0]
    # Filtrar solo las detecciones de la clase "persona" (class == 0)
    people_detections = detections[detections["class"] == 0]

    if people_detections.empty:
        return []
    else:
        # Aplicar el tracker
        tracked_objects = trackerar_detecciones(people_detections)

        bounding_boxes = []
        for track in tracked_objects:
            x1, y1, x2, y2, track_id = track[
                :5
            ]  # ID está en track[4] o track[5] dependiendo de versión

            lower_left = calcularPixelMapaHomografia(camara1, camara2, x1, y2)
            lower_right = calcularPixelMapaHomografia(camara1, camara2, x2, y2)

            color = COLOR[int(track_id) % len(COLOR)]

            persona_detectada = Persona(
                int(track_id), 1.0, color, lower_right, lower_left
            )
            LISTAS_PERSONAS[int(track_id)] = (
                persona_detectada  # Guardamos la persona en el diccionario
            )
            print(f"ID: {track_id}, Color: {color}")

            bounding_boxes.append(persona_detectada.__dict__)
            print(len(bounding_boxes))
        return bounding_boxes
