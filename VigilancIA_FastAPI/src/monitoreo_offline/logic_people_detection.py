import insightface
import faiss
import sqlite3
from sklearn.preprocessing import normalize
import argparse

import cv2
import os
import numpy as np

from ByteTrack.yolox.tracker.byte_tracker import BYTETracker

args = argparse.Namespace(
    track_thresh=0.50,  # Este parámetro define el umbral de confianza para detectar objetos que se van a seguir.
    track_buffer=160,  # Este parámetro define el número de fotogramas que se almacenan en el búfer de seguimiento.
    match_thresh=0.95,  # Este parámetro define el umbral de similitud para asociar detecciones con pistas existentes.
    mot20=False,  # Este parámetro indica si se está utilizando el conjunto de datos MOT20 (Multiple Object Tracking 20).
)

SORT_TRACKER = BYTETracker(args, frame_rate=40)

CARAS = insightface.app.FaceAnalysis(
    name="buffalo_l", providers=["CPUExecutionProvider"]
)
CARAS.prepare(ctx_id=0)  # (ctx_id=0 normalmente es tu GPU principal)
DBV = faiss.read_index("files/index_insight.faiss")  # Cargar índice FAISS y nombres

# Leer base de datos con nombres
conn = sqlite3.connect("files/caras_insight.db")
cursor = conn.cursor()
cursor.execute("SELECT nombre, vector_index FROM personas")
id_to_name = {row[1]: row[0] for row in cursor.fetchall()}
conn.close()
print("Modelo buffalo_l cargado")


def comprobar_detecciones_personas(detecciones_trackeadas, x1, y1, x2, y2):
    tolerancia = 5
    for dict_persona in detecciones_trackeadas:
        if (
            abs(dict_persona["x1_imagen_original"] - x1) <= tolerancia
            and abs(dict_persona["y1_imagen_original"] - y1) <= tolerancia
            and abs(dict_persona["x2_imagen_original"] - x2) <= tolerancia
            and abs(dict_persona["y2_imagen_original"] - y2) <= tolerancia
        ):
            return dict_persona["color"], dict_persona["nombre"]
    return "red", "None"  # Color por defecto si no se encuentra coincidencia


def trackerar_detecciones(people_detections, image_tensor):
    global SORT_TRACKER
    # Convertir detecciones a formato esperado por ByteTrack: [x1, y1, x2, y2, score]
    dets_for_tracker = people_detections[
        ["xmin", "ymin", "xmax", "ymax", "confidence"]
    ].to_numpy()
    # ByteTrack requiere info de tamaño de imagen: (alto, ancho)
    image_height, image_width = image_tensor.shape[1:3]  # tensor shape: [C, H, W]
    # Aplicar tracker de ByteTrack
    online_targets = SORT_TRACKER.update(
        dets_for_tracker, [image_height, image_width], (image_height, image_width)
    )
    return online_targets


def guardar_cara(imagen_cara):
    # Crear carpeta si no existe
    carpeta = "caras_guardadas"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Crear un número aleatorio
    numero_random = np.random.randint(10000, 99999)
    nombre_archivo = os.path.join(carpeta, f"cara_{numero_random}.jpg")

    # Guardar la imagen
    cv2.imwrite(nombre_archivo, imagen_cara)
    print(f"Cara guardada en: {nombre_archivo}")


def ajustar_imagen_cara(img_bgr, x1, y1, x2, y2):
    margen = 0.3  # Margen de expansión (20%)
    altura, anchura = img_bgr.shape[:2]

    # Ajustar el y1 hacia arriba
    desplazamiento = int((y2 - y1) * margen)
    y1_des = max(0, y1 - desplazamiento)  # Evitar que se salga de la imagen

    return x1, y1_des, x2, y1


def reconocimiento_caras(imagen_bgr, x1, y1, x2, y2):
    print("Reconociendo...")
    threshold = 0.4  # Umbral de distancia de similitud

    # Recortar imagen de la persona detectada
    cara = imagen_bgr[y1:y2, x1:x2]

    guardar_cara(cara)

    faces = CARAS.get(cara)

    for face in faces:
        embedding = face.embedding.astype("float32").reshape(1, -1)
        embedding = normalize(embedding, axis=1)  # ✅ NORMALIZACIÓN L2

        D, I = DBV.search(embedding, 1)

        print(D, I)

        if D[0][0] < threshold:
            nombre = id_to_name.get(I[0][0], "Desconocido")
            print(f"Nombre encontrado: {nombre}")
            return nombre
        else:
            return "Desconocido"
    return "Desconocido"
