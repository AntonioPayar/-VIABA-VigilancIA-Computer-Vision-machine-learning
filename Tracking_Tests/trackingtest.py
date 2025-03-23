import cv2
import numpy as np
import time
import random
from ultralytics import YOLO
from sort import Sort
import os

RANDOM_NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace", 
                "Henry", "Isabel", "Jack", "Kate", "Leo", "Mia", "Noah", 
                "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Taylor"]

#  YOLO
print("Loading YOLO model...")
model = YOLO('yolov8n.pt') 

# Inicializar SORT
mot_tracker = Sort(max_age=30, min_hits=3, iou_threshold=0.3)

# Video
video_path = 'videotest2.webm'
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {video_path}")

# Obtener propiedades del video para igualar la salida
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

print(f"Video loaded: {frame_width}x{frame_height} at {fps} FPS")

output_path = 'tracked_output.mp4'
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

# Diccionario para almacenar el seguimiento
track_info = {}

def get_random_color():
    return (random.randint(128, 255), random.randint(128, 255), random.randint(128, 255))

# Contador de proceso para actualizaciones de progreso
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
processed_frames = 0

print(f"Starting video processing. Total frames: {frame_count}")
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("End of video reached")
        break
    
    processed_frames += 1
    if processed_frames % 30 == 0: 
        elapsed = time.time() - start_time
        progress = processed_frames / frame_count * 100
        fps_processing = processed_frames / elapsed if elapsed > 0 else 0
        print(f"Progress: {progress:.1f}% ({processed_frames}/{frame_count}) - Processing FPS: {fps_processing:.1f}")
        
    results = model(frame)
    
    # Extraer detecciones para personas (clase 0 es persona en yolo)
    detections = []
    
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Coordenadas de la caja
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            conf = float(box.conf[0].cpu().numpy())
            cls = int(box.cls[0].cpu().numpy())
            
            # Solo rastrear personas 
            if cls == 0 and conf > 0.5:
                # Formato para SORT: [x1, y1, x2, y2, confianza]
                detections.append([x1, y1, x2, y2, conf])
    
    # Actualizar el rastreador SORT
    track_bbs_ids = mot_tracker.update(np.array(detections) if len(detections) > 0 else np.empty((0, 5)))
    
    # Dibujar objetos rastreados
    for track in track_bbs_ids:
        x1, y1, x2, y2, track_id = track
        x1, y1, x2, y2, track_id = int(x1), int(y1), int(x2), int(y2), int(track_id)
        

        if track_id not in track_info:
            random_name = random.choice(RANDOM_NAMES)
            random_color = get_random_color()
            track_info[track_id] = {
                'name': f"{random_name}",
                'color': random_color
            }
        
        name = track_info[track_id]['name']
        color = track_info[track_id]['color']
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Dibujar etiqueta con fondo negro 
        label = f"{name} (ID: {track_id})"
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
        cv2.rectangle(frame, (x1, y1 - 25), (x1 + text_size[0], y1), color, -1)
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Escribir el cuadro en el video de salida
    out.write(frame)

total_time = time.time() - start_time
print(f"Processing complete! Total time: {total_time:.2f} seconds")
print(f"Output saved to: {os.path.abspath(output_path)}")

cap.release()
out.release()
