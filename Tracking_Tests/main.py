import cv2
import numpy as np
import time
import random
from ultralytics import YOLO
from sort import Sort
import asyncio
from fastapi import FastAPI, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import StreamingResponse, HTMLResponse
from pathlib import Path
import itertools


#---Prueba de fastapi que itera videos de un folder y asocia una id/nombre a cada bounding box del YOLO---



VIDEO_FOLDER = Path("./videos")  
MODEL_PATH = 'yolov8n.pt'
MAX_AGE = 30  
MIN_HITS = 3  
IOU_THRESHOLD = 0.3 
CONFIDENCE_THRESHOLD = 0.5 
PERSON_CLASS_ID = 0 

RANDOM_NAMES = ["Alice", "Bob", "Charlie", "David", "Emma", "Frank", "Grace",
                "Henry", "Isabel", "Jack", "Kate", "Leo", "Mia", "Noah",
                "Olivia", "Paul", "Quinn", "Rachel", "Sam", "Taylor"]

# --- YOLO ---
print("Loading YOLO model...")
try:
    model = YOLO(MODEL_PATH)
    print("YOLO model loaded successfully.")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    exit()

# --- FastAPI Setup ---
app = FastAPI()
templates = Jinja2Templates(directory="templates")

# --- Helper Func ---
def get_random_color():
    return (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255))

def find_video_files(folder_path: Path):
    video_extensions = {".mp4", ".webm", ".avi", ".mov", ".mkv"}
    video_files = [f for f in folder_path.iterdir() if f.is_file() and f.suffix.lower() in video_extensions]
    print(f"Found video files: {[f.name for f in video_files]}")
    return video_files

# --- Logica de procesado ---
async def process_videos():
    print("DEBUG: process_videos generator started.") 
    video_files = find_video_files(VIDEO_FOLDER)
    print(f"DEBUG: Found videos in {VIDEO_FOLDER.resolve()}: {[f.name for f in video_files]}")

    if not video_files:
        print(f"DEBUG: No video files found in {VIDEO_FOLDER}. Stopping stream.") 

        return 

    video_cycle = itertools.cycle(video_files)

    while True:
        video_path = next(video_cycle)
        print(f"\nDEBUG: --- Attempting to process Video: {video_path.name} ---")

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"DEBUG: Error: Cannot open video file: {video_path.name}") 
            await asyncio.sleep(2) 
            continue 

        print(f"DEBUG: Successfully opened video: {video_path.name}") 

        # Reset del sort para cada video
        mot_tracker = Sort(max_age=MAX_AGE, min_hits=MIN_HITS, iou_threshold=IOU_THRESHOLD)
        track_info = {}
        assigned_names = set()

        frame_idx = 0
        start_time = time.time()

        while True:
            ret, frame = cap.read()
            if not ret:
                cap.release()
                break 

            frame_idx += 1

            try: 
                results = model(frame, verbose=False)

                # Extrae personas
                detections = []
                for r in results:
                    boxes = r.boxes
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy())
                        cls = int(box.cls[0].cpu().numpy())

                        if cls == PERSON_CLASS_ID and conf > CONFIDENCE_THRESHOLD:
                            detections.append([x1, y1, x2, y2, conf])

                # Update SORT 
                np_detections = np.array(detections) if detections else np.empty((0, 5))
                track_bbs_ids = mot_tracker.update(np_detections)

                #Misma lógica de trackingtest
                for track in track_bbs_ids:
                     if len(track) == 5:
                         x1, y1, x2, y2, track_id = map(int, track)
                         if track_id not in track_info:
                             available_names = [name for name in RANDOM_NAMES if name not in assigned_names]
                             if not available_names: available_names = RANDOM_NAMES
                             random_name = random.choice(available_names)
                             assigned_names.add(random_name)
                             track_info[track_id] = {'name': f"{random_name}",'color': get_random_color()}

                         name = track_info[track_id]['name']
                         color = track_info[track_id]['color']
                         cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                         label = f"{name} (ID: {track_id})"
                         (label_width, label_height), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                         cv2.rectangle(frame, (x1, y1 - label_height - baseline - 5), (x1 + label_width, y1), color, -1)
                         cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2) # Black text
                     else:
                          print(f"DEBUG: Unexpected track format: {track}")


                ret_encode, buffer = cv2.imencode('.jpg', frame)
                if not ret_encode:
                    print("DEBUG: Error encoding frame")
                    continue

                frame_bytes = buffer.tobytes()

                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

                await asyncio.sleep(0.001) 

            except Exception as e:
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 print(f"ERROR processing frame {frame_idx} in {video_path.name}: {e}")
                 import traceback
                 traceback.print_exc() 
                 print(f"!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
                 continue


        elapsed_time = time.time() - start_time
        print(f"DEBUG: Video '{video_path.name}' processing finished in {elapsed_time:.2f} seconds.") # <-- ADDED
        cap.release() 
        await asyncio.sleep(1) 



# --- FastAPI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(process_videos(), media_type='multipart/x-mixed-replace; boundary=frame')

# --- Main Execution ---
if __name__ == "__main__":
    import uvicorn
    print("Starting FastAPI server...")
    print(f"Video source folder: {VIDEO_FOLDER.resolve()}")
    print(f"Access the stream at http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000) 