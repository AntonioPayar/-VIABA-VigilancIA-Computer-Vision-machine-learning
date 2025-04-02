import cv2
import asyncio
import aiohttp
from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
import base64
import json
import numpy as np

app = FastAPI()

# URL del modelo YOLO expuesto en mlserver
YOLO_URL = ""

async def call_mlserver(url: str, frame_data: str):
    payload = {
        "inputs": [
            {
                "name": "input-0",
                "shape": [1],
                "datatype": "BYTES",
                "data": [frame_data]
            }
        ]
    }
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as response:
            return await response.json()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    cap = cv2.VideoCapture("video.mp4")  # Reemplaza "video.mp4" con la ruta a tu video
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Reinicia el video en bucle
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            # Codifica el frame a JPEG y luego a base64
            ret2, buffer = cv2.imencode(".jpg", frame)
            b64_frame = base64.b64encode(buffer).decode("utf-8")
            frame_data = f"data:image/jpeg;base64,{b64_frame}"

            # Llama al mlserver para obtener inferencia (detecciones)
            yolo_response = await call_mlserver(YOLO_URL, frame_data)

            # Procesa la respuesta y dibuja los bounding boxes.
            # Se asume que la respuesta es similar a:
            # { "outputs": [ { "name": "detections", "data": [json_string_deteccion1, ...] } ] }
            if yolo_response.get("outputs"):
                detections_data = yolo_response["outputs"][0]["data"]
                for detection_json in detections_data:
                    try:
                        detection = json.loads(detection_json)
                        bbox = detection.get("bbox", [])
                        if len(bbox) == 4:
                            x1, y1, x2, y2 = map(int, bbox)
                            label = detection.get("class", "object")
                            conf = detection.get("confidence", 0)
                            # Dibuja la caja y la etiqueta en el frame
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    except Exception as e:
                        print("Error al procesar la detección:", e)

            # Re-codifica el frame con las cajas dibujadas
            ret3, buffer2 = cv2.imencode(".jpg", frame)
            b64_frame_out = base64.b64encode(buffer2).decode("utf-8")
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
      <title>Streaming de Video con YOLO</title>
    </head>
    <body>
      <h1>Video Streaming</h1>
      <img id="video" width="640" height="480"/>
      <script>
        const ws = new WebSocket("ws://localhost:8000/ws");
        ws.onmessage = function(event) {
          document.getElementById("video").src = "data:image/jpeg;base64," + event.data;
        };
      </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)
