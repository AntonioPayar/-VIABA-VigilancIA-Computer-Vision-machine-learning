import cv2
import easyocr
import argparse
import os
import numpy as np
from ultralytics import YOLO

def detect_license_plate(image_path):
    """
    Usa YOLOv8 para detectar la región de la matrícula en la imagen.
    """
    model = YOLO("license_plate_detector.pt")  # Modelo preentrenado de YOLOv8
    results = model(image_path)
    image = cv2.imread(image_path)
    
    for result in results:
        for box in result.boxes.xyxy:  # Extrae coordenadas (x1, y1, x2, y2)
            x1, y1, x2, y2 = map(int, box[:4])
            plate_img = image[y1:y2, x1:x2]
            cv2.imwrite("debug_plate.jpg", plate_img)
            print("Se ha guardado la imagen de la matrícula detectada en 'debug_plate.jpg'.")
            return plate_img, (x1, y1, x2 - x1, y2 - y1), image
    
    print("No se detectó ninguna matrícula en la imagen.")
    return None

def refine_plate_crop(plate_img):
    """
    Refina el recorte de la matrícula eliminando bordes innecesarios
    usando detección de contornos.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        plate_img = plate_img[y:y+h, x:x+w]
    
    return plate_img

def preprocess_plate_for_ocr(plate_img):
    """
    Aplica filtros para mejorar la detección OCR en la matrícula.
    """
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 50, 200)
    
    return edged

def extract_text_from_plate(plate_img):
    """
    Usa EasyOCR para extraer el texto de la matrícula.
    """
    #refined_plate = refine_plate_crop(plate_img)
    #preprocessed_plate = preprocess_plate_for_ocr(refined_plate)
    
    reader = easyocr.Reader(['en'])
    results = reader.readtext(plate_img, detail=0)
    
    plate_text = " ".join(results).strip()
    return plate_text

def main():
    parser = argparse.ArgumentParser(description="Extracción de matrícula de coche usando YOLOv8.")
    parser.add_argument("image", help="Ruta de la imagen del coche.")
    args = parser.parse_args()
    
    detection = detect_license_plate(args.image)
    if detection is None:
        print("No se pudo detectar la matrícula en la imagen proporcionada.")
        return

    plate_img, (x, y, w, h), original_image = detection
    text = extract_text_from_plate(plate_img)
    print("Texto detectado en la matrícula:", text)
    
    cv2.rectangle(original_image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    cv2.imwrite("output.jpg", original_image)
    print("Imagen con la matrícula resaltada guardada en 'output.jpg'.")

if __name__ == '__main__':
    main()
