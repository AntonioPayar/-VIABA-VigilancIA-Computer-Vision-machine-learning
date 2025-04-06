import cv2
import asyncio
import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import json
import numpy as np
from sort import Sort

app = FastAPI()

# Endpoints de inferencia
YOLO_URL = "http://localhost:8080/v2/models/object-detection/infer"           # Modelo YOLO (detección general)
FACE_URL = ""         # Modelo de reconocimiento facial
MATRICULA_URL = ""   # Modelo de detección de matrículas

# Inicializa el tracker SORT
tracker = Sort(max_age=5, min_hits=3, iou_threshold=0.3)
# Diccionario para guardar el valor de "identificado" por track id
track_identificados = {}


def compute_iou(boxA, boxB):
    # Calcula el Intersection over Union (IoU) entre dos cajas
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA + 1)
    interH = max(0, yB - yA + 1)
    interArea = interW * interH
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


async def call_mlserver(url: str, image_data: str):
    if not url:
        # Devuelve una respuesta simulada sin inferencia.
        return {"outputs": [{"data": []}]}
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1],
                "datatype": "BYTES",
                "data": [image_data]
            }
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            return await response.json()


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture("video.mp4")  # Reemplaza con la ruta a tu video
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Codifica el frame a JPEG y luego a base64 para enviarlo al mlserver de YOLO
            ret_enc, buffer = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            frame_data = f"data:image/jpeg;base64,{b64_frame}"

            # Llama a YOLO para obtener las detecciones
            yolo_response = await call_mlserver(YOLO_URL, frame_data)

            # Procesa las detecciones y crea una lista con diccionarios
            detections_list = []
            if yolo_response.get("outputs"):
                detections_data = json.loads(yolo_response["outputs"][0]["data"][0])
                for detection in detections_data:
                    try:
                        bbox = detection.get("bbox", [])
                        if len(bbox) == 4:
                            # Suponemos que la respuesta incluye "class" y "confidence"
                            x1, y1, x2, y2 = map(int, bbox)
                            detections_list.append({
                                "bbox": [x1, y1, x2, y2],
                                "confidence": detection.get("confidence", 0),
                                "class": detection.get("class", "").lower(),
                                "identificado": None
                            })
                    except Exception as e:
                        print(f"Error parsing json {detection_json}:", e)

            # Prepara el array para SORT: cada fila [x1, y1, x2, y2, confidence]
            if detections_list:
                dets_array = np.array([[d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3], d["confidence"]]
                                       for d in detections_list])
            else:
                dets_array = np.empty((0, 5))

            # Actualiza el tracker
            tracks = tracker.update(dets_array)  # tracks: [x1, y1, x2, y2, track_id]

            # Para cada track, asocia la detección de YOLO que más se solape (IoU)
            for track in tracks:
                tx1, ty1, tx2, ty2, track_id = map(int, track)
                track_box = [tx1, ty1, tx2, ty2]
                best_iou = 0
                best_det = None
                for det in detections_list:
                    iou = compute_iou(track_box, det["bbox"])
                    if iou > best_iou:
                        best_iou = iou
                        best_det = det
                # Si se encontró una detección asociada
                if best_det is not None:
                    # Si no se ha identificado o el valor actual es "N/A", se reintenta la inferencia
                    if (track_id not in track_identificados) or (track_identificados.get(track_id) == "N/A"):
                        # Recorta la región del track
                        crop = frame[ty1:ty2, tx1:tx2]
                        ret_crop, crop_buffer = cv2.imencode(".jpg", crop)
                        b64_crop = base64.b64encode(crop_buffer).decode("utf-8")
                        crop_data = f"data:image/jpeg;base64,{b64_crop}"
                        if best_det["class"] == "coche":
                            extra_response = await call_mlserver(MATRICULA_URL, crop_data)
                        elif best_det["class"] == "persona":
                            extra_response = await call_mlserver(FACE_URL, crop_data)
                        else:
                            extra_response = None

                        # Se espera que la respuesta extra tenga la información identificada
                        if extra_response and extra_response.get("outputs"):
                            try:
                                valor = json.loads(extra_response["outputs"][0]["data"][0])
                            except Exception as e:
                                print("Error procesando respuesta extra:", e)
                                valor = "N/A"
                        else:
                            valor = "N/A"
                        best_det["identificado"] = valor
                        track_identificados[track_id] = valor
                    else:
                        # Si ya se identificó (valor distinto de "N/A"), se reutiliza el valor
                        best_det["identificado"] = track_identificados[track_id]

            # Dibuja en el frame los bounding boxes, el track id y el campo "identificado"
            for track in tracks:
                tx1, ty1, tx2, ty2, track_id = map(int, track)
                identificado_val = track_identificados.get(track_id, "N/A")
                cv2.rectangle(frame, (tx1, ty1), (tx2, ty2), (0, 255, 0), 2)
                cv2.putText(frame, f"ID {track_id} - {identificado_val}", (tx1, ty1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            # Re-codifica el frame con las anotaciones y envíalo por WebSocket
            ret_out, buffer_out = cv2.imencode(".jpg", frame)
            b64_frame_out = base64.b64encode(buffer_out).decode("utf-8")
            await websocket.send_text(b64_frame_out)
            await asyncio.sleep(0.033)  # Aproximadamente 30 FPS

    except Exception as e:
        print("Error en WebSocket:", e)
    finally:
        cap.release()
        await websocket.close()


@app.get("/")
async def get():
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
      <title>Streaming con YOLO, SORT y Extra Inferencia</title>
    </head>
    <body>
      <h1>Streaming de Video</h1>
      <img id="video" width="640" height="480"/>
      <script>
        const ws = new WebSocket("ws://localhost:8001/ws");
        ws.onmessage = function(event) {
          document.getElementById("video").src = "data:image/jpeg;base64," + event.data;
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
