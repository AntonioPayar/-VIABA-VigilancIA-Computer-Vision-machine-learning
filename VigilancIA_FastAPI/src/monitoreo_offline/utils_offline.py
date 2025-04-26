import torch
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import base64
from io import BytesIO

# Imports de clases propias
from classes.Cordenadas_Configuracion import *
from classes.Clases_Detecciones import Persona, Vehiculo
from monitoreo_offline.logic_people_detection import *
from monitoreo_offline.logic_car_detection import *

import cv2
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

print("Cargando el modelo YOLOv5...")
YOLO = torch.hub.load("ultralytics/yolov5", "yolov5s", pretrained=True)
print("Modelo YOLOv5 cargado")


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
LISTAS_VEHICULOS = {}


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


def imagen_bounding_boxes(
    image, draw, fuente, tipo_deteccion, trackeos, detections=None
):
    """
    Dibuja bounding boxes en la imagen y la convierte a Base64.
    Si detections es None o está vacío, convierte la imagen original a Base64.
    """
    for index, row in detections.iterrows():
        x1, y1, x2, y2 = (
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )

        if tipo_deteccion == "personas":
            color, name = comprobar_detecciones_personas(trackeos, x1, y1, x2, y2)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        elif tipo_deteccion == "coches":
            color, name = comprobar_detecciones_coches(trackeos, x1, y1, x2, y2)
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)

        # Calcular la posición del texto
        texto_x = x1
        texto_y = y1 - 15 - 5  # Colocar el texto encima del bounding box

        # Dibujar el texto
        draw.text((texto_x, texto_y), name, fill=color, font=fuente)

    return image


def procesamiento_deteccion_personas(
    people_detections, image_tensor, camara1, camara2, img_bgr
):
    max_frames_reconocimiento = 1000
    contador = 0
    detecciones_trackeadas = []
    # Aplicar el tracker
    tracked_objects = trackerar_detecciones(people_detections, image_tensor)

    for t in tracked_objects:
        tlwh = t.tlwh  # [x, y, w, h]
        x1, y1 = tlwh[0], tlwh[1]
        x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
        track_id = t.track_id

        row = people_detections.iloc[contador]
        # Obtener las coordenadas
        x1_cara, y1_cara, x2_cara, y2_cara = ajustar_imagen_cara(
            image_tensor,
            int(row["xmin"]),
            int(row["ymin"]),
            int(row["xmax"]),
            int(row["ymax"]),
        )

        lower_left = calcularPixelMapaHomografia(camara1, camara2, x1, y2)
        lower_right = calcularPixelMapaHomografia(camara1, camara2, x2, y2)

        color = COLOR[int(track_id) % len(COLOR)]

        # Creamos la persona detectada
        persona_detectada = Persona(
            int(track_id), 1.0, color, lower_right, lower_left, x1, y1, x2, y2
        )

        # Comprobamos si la persona ya ha sido detectada
        persona_encontrada = LISTAS_PERSONAS.get(int(track_id))
        if persona_encontrada is None:
            name = reconocimiento_caras(
                imagen_bgr=img_bgr,
                x1=x1_cara,
                y1=y1_cara,
                x2=x2_cara,
                y2=y2_cara,
            )
            persona_detectada.setNombre(name)
        else:
            if (
                persona_encontrada.nombre == "Desconocido"
                and persona_encontrada.contador % max_frames_reconocimiento == 0
            ):
                print("HOLASSDASODINAOFUNDUJIFNISDUGBNISUDGFB")
                # Si la persona ya fue detectada pero no se le ha asignado un nombre
                name = reconocimiento_caras(
                    imagen_bgr=img_bgr,
                    x1=x1_cara,
                    y1=y1_cara,
                    x2=x2_cara,
                    y2=y2_cara,
                )
                persona_detectada.setNombre(name)

            persona_detectada.setContador(LISTAS_PERSONAS[int(track_id)].contador)
            persona_detectada.setNombre(LISTAS_PERSONAS[int(track_id)].nombre)

        LISTAS_PERSONAS[int(track_id)] = (
            persona_detectada  # Guardamos la persona en el diccionario
        )
        print(
            f"ID: {track_id}, Color: {color}, Nombre: {persona_detectada.nombre}, Contador: {persona_detectada.contador}"
        )
        contador += 1
        detecciones_trackeadas.append(persona_detectada.__dict__)

    return detecciones_trackeadas


def procesamiento_deteccion_coches(
    coches_detections, image_tensor, camara1, camara2, img_bgr
):
    detecciones_trackeadas = []
    # Aplicar el tracker
    tracked_objects = trackerar_detecciones(coches_detections, image_tensor)

    for t in tracked_objects:
        tlwh = t.tlwh  # [x, y, w, h]
        x1, y1 = tlwh[0], tlwh[1]
        x2, y2 = x1 + tlwh[2], y1 + tlwh[3]
        track_id = t.track_id

        lower_left = calcularPixelMapaHomografia(camara1, camara2, x1, y2)
        lower_right = calcularPixelMapaHomografia(camara1, camara2, x2, y2)

        color = COLOR[int(track_id) % len(COLOR)]

        # Crear el objeto vehículo detectado
        vehiculo_detectado = Vehiculo(
            int(track_id), 1.0, color, lower_right, lower_left, x1, y1, x2, y2
        )

        # Comprobamos si la persona ya ha sido detectada
        if LISTAS_VEHICULOS.get(int(track_id)) is None:
            # Reconocimiento de matriculas
            vehiculo_detectado.setMatricula("111111")
        else:
            vehiculo_detectado.setMatricula(LISTAS_VEHICULOS[int(track_id)].matricula)

        # Guardar el vehículo en el diccionario global si fuera necesario
        LISTAS_VEHICULOS[int(track_id)] = vehiculo_detectado

        print(f"Vehículo ID: {track_id}, Color: {color}")

        detecciones_trackeadas.append(vehiculo_detectado.__dict__)

    return detecciones_trackeadas


def detect_objects(camara1: DatosCamaras, camara2: DatosCamaras, image_tensor, img_bgr):
    # Realizamos la inferencia
    results = YOLO(image_tensor)
    # Obtener las bounding boxes, clases y scores
    detections = results.pandas().xyxy[0]
    # Filtrar solo las detecciones de la clase "persona" (class == 0)
    people_detections = detections[detections["class"] == 0]
    coches_detections = detections[detections["class"] == 2]
    # Cremos una imagen para agregar las detecciones
    image = Image.fromarray(image_tensor)
    draw = ImageDraw.Draw(image)
    fuente = ImageFont.load_default()

    personas_trackeadas = []
    if not people_detections.empty:
        personas_trackeadas = procesamiento_deteccion_personas(
            people_detections, image_tensor, camara1, camara2, img_bgr
        )
        # Obtenemos la imagen con bounding boxes
        image = imagen_bounding_boxes(
            image, draw, fuente, "personas", personas_trackeadas, people_detections
        )

    vehiculos_trackeados = []
    if not coches_detections.empty:
        vehiculos_trackeados = procesamiento_deteccion_coches(
            coches_detections, image_tensor, camara1, camara2, img_bgr
        )
        # Obtenemos la imagen con bounding boxes
        image = imagen_bounding_boxes(
            image, draw, fuente, "coches", vehiculos_trackeados, coches_detections
        )

    buffered = BytesIO()
    image.save(buffered, format="JPEG")
    image_bounding = base64.b64encode(buffered.getvalue()).decode()

    return personas_trackeadas, vehiculos_trackeados, image_bounding
