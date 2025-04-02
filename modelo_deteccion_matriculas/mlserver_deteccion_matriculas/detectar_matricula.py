import cv2
import numpy as np
import base64
from PIL import Image
from io import BytesIO
import easyocr
from ultralytics import YOLO

# Cargar el modelo YOLO a partir del archivo .pt (se asume que ya existe en la ruta)
model = YOLO("models/matricula-detection/license_plate_detector.pt")
# Inicializar el lector de EasyOCR (en este ejemplo para inglés, puedes agregar otros idiomas si lo requieres)
reader = easyocr.Reader(['en'])

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
        image = Image.fromarray(image_input.astype('uint8'))
    else:
        raise ValueError("Formato de imagen no soportado")

    # Ejecutar la inferencia con el modelo YOLO
    results = model(image)

    # Verificar si se han detectado cajas
    if results and len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
        # Tomamos la primera detección (asumiendo que es la matrícula)
        # Obtenemos las coordenadas en formato [x1, y1, x2, y2]
        # La API de ultralytics devuelve tensores; los convertimos a numpy
        box = results[0].boxes.xyxy[0].cpu().numpy()
        x1, y1, x2, y2 = map(int, box)

        # Convertir la imagen original a numpy para recortar la región
        img_np = np.array(image)
        # Asegurarse de que las coordenadas sean válidas
        if x1 < 0 or y1 < 0 or x2 > img_np.shape[1] or y2 > img_np.shape[0]:
            return "Coordenadas de la detección fuera de rango"
        
        # Recortar la región de interés (ROI) que contiene la matrícula
        roi = img_np[y1:y2, x1:x2]

        # Aplicar EasyOCR sobre la región recortada
        ocr_result = reader.readtext(roi)

        if ocr_result:
            # Por ejemplo, tomar el texto con mayor confianza
            matricula = max(ocr_result, key=lambda x: x[2])[1]
            return matricula
        else:
            return "No se pudo extraer la matrícula"
    else:
        return "No se detectó matrícula"
