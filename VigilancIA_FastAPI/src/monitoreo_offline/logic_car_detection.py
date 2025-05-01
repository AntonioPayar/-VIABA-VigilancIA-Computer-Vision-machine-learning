import numpy as np
import base64
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO
import cv2
import os
import base64

# Cargar el modelo YOLO a partir del archivo .pt (se asume que ya existe en la ruta)
YOLO_MATRICULAS = YOLO("files/license_plate_detector.pt")
EASYOCR = easyocr.Reader(["en"])
print("Modelo MATRICULAS -EOCR cargado")


def comprobar_detecciones_coches(detecciones_trackeadas, x1, y1, x2, y2):
    tolerancia = 5
    for dict_coche in detecciones_trackeadas:
        if (
            abs(dict_coche["x1_imagen_original"] - x1) <= tolerancia
            and abs(dict_coche["y1_imagen_original"] - y1) <= tolerancia
            and abs(dict_coche["x2_imagen_original"] - x2) <= tolerancia
            and abs(dict_coche["y2_imagen_original"] - y2) <= tolerancia
        ):
            return dict_coche["color"], dict_coche["matricula"]
    return "red", "None"  # Color por defecto si no se encuentra coincidencia


def guardar_matricula(imagen_matricula):
    # Crear carpeta si no existe
    carpeta = "files/matricula_guardadas"
    if not os.path.exists(carpeta):
        os.makedirs(carpeta)

    # Crear un número aleatorio
    numero_random = np.random.randint(10000, 99999)
    nombre_archivo = os.path.join(carpeta, f"matricula_{numero_random}.jpg")

    # Guardar la imagen
    cv2.imwrite(nombre_archivo, imagen_matricula)
    print(f"Matricula guardada en: {nombre_archivo}")


def detectar_matricula(image_input):
    """
    Realiza la inferencia con el modelo YOLO para detectar la matrícula y aplica EasyOCR
    para extraer el texto de la región detectada.

    Se acepta la imagen en formato base64 o como array numpy.
    """
    # Convertir la imagen de entrada a un objeto PIL Image
    if isinstance(image_input, str) and image_input.startswith("data:image"):
        base64_data = image_input.split(",")[1]
        img_bytes = base64.b64decode(base64_data)
        image = Image.open(BytesIO(img_bytes))
    elif isinstance(image_input, np.ndarray):
        image = Image.fromarray(image_input.astype("uint8"))
    else:
        raise ValueError("Formato de imagen no soportado")

    # Ejecutar la inferencia con el modelo YOLO
    results = YOLO_MATRICULAS(image)
    print("Detectando matrícula1...")

    # Verificar si se han detectado cajas
    if (
        results
        and len(results) > 0
        and results[0].boxes is not None
        and len(results[0].boxes) > 0
    ):
        # Tomamos la primera detección (asumiendo que es la matrícula)
        # Obtenemos las coordenadas en formato [x1, y1, x2, y2]
        # La API de ultralytics devuelve tensores; los convertimos a numpy
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Convertir la imagen original a numpy para recortar la región
        img_np = np.array(image)
        # Asegurarse de que las coordenadas sean válidas
        if x1 < 0 or y1 < 0 or x2 > img_np.shape[1] or y2 > img_np.shape[0]:
            return "Out of bounds", base64.b64encode(buffer).decode("utf-8")

        # Recortar la región de interés (ROI) que contiene la matrícula
        roi = img_np[y1:y2, x1:x2]

        # Codificar la imagen en memoria como JPEG
        _, buffer = cv2.imencode(".jpg", roi)
        guardar_matricula(roi)
        # Aplicar EasyOCR sobre la región recortada
        ocr_result = EASYOCR.readtext(roi)
        print("Detectando matrícula2...")

        if ocr_result:
            # Por ejemplo, tomar el texto con mayor confianza
            matricula = max(ocr_result, key=lambda x: x[2])[1]
            return matricula, base64.b64encode(buffer).decode("utf-8")
        else:
            return "Desconocido", base64.b64encode(buffer).decode("utf-8")
    else:
        _, buffer = cv2.imencode(".jpg", np.array(image))
        return "Desconocido", base64.b64encode(buffer).decode("utf-8")
