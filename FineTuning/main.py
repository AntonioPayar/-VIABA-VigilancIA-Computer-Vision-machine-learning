from fastapi import FastAPI, Response, Request
from fastapi.responses import HTMLResponse, StreamingResponse
import cv2
import numpy as np
import os
import glob
import random
import time
import collections
import asyncio
import torch

# Import components from the logic file
from inference_logic import (
    load_trained_model,
    get_inference_transform,
    predict_chunk,
    class_names_loaded,
    CLIP_LEN,
    FPS_TO_PROCESS,
    PREDICTION_THRESHOLD,
    SMOOTHING_WINDOW,
    VAL_DIR,
    MAX_INFERENCE_VIDEOS,
    MODEL_SAVE_PATH,
    INPUT_SIZE,
    FRAME_RATE
)

app = FastAPI()

# --- Load model and transform on startup ---
@app.on_event("startup")
async def startup_event():
    print("Initializing model...")
    app.state.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {app.state.device}")
    app.state.model = load_trained_model(app.state.device)
    app.state.transform = get_inference_transform()

    if not app.state.model or not app.state.transform:
        print("Error: Failed to load model or transform.")
        app.state.initialization_failed = True
    else:
        app.state.initialization_failed = False
        print("Model and transform loaded successfully.")

# --- Video stream generator ---
async def video_stream_generator():
    if getattr(app.state, 'initialization_failed', True):
        print("Error: Initialization failed.")
        error_img = np.zeros((300, 400, 3), dtype=np.uint8)
        cv2.putText(error_img, "Init Error", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        _, frame_bytes = cv2.imencode('.jpg', error_img)
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes.tobytes() + b'\r\n')
        return

    model = app.state.model
    transform = app.state.transform
    device_local = app.state.device

    print("Starting video stream...")
    if not os.path.isdir(VAL_DIR):
        print(f"Error: Directory '{VAL_DIR}' not found.")
        return

    video_files = []
    for ext in ['*.avi', '*.mp4', '*.mkv']:
        video_files.extend(glob.glob(os.path.join(VAL_DIR, '*', ext)))

    if not video_files:
        print(f"No videos found in '{VAL_DIR}' with specified extensions.")
        return

    random.shuffle(video_files)
    video_files_to_process = video_files[:MAX_INFERENCE_VIDEOS]
    print(f"Processing {len(video_files_to_process)} videos.")

    for video_path in video_files_to_process:
        print(f"Streaming video: {video_path}")
        actual_category = os.path.basename(os.path.dirname(video_path)) or "Unknown"
        if actual_category not in class_names_loaded:
            actual_category = "Unknown"

        if not os.path.isfile(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            continue

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        frame_skip = max(1, int(round(original_fps / FPS_TO_PROCESS))) if original_fps > 0 else 1
        target_frame_time = 1.0 / FPS_TO_PROCESS

        frame_buffer = collections.deque(maxlen=CLIP_LEN)
        recent_predictions = collections.deque(maxlen=SMOOTHING_WINDOW)
        fight_detected_smoothed = False
        last_class_name = "Processing..."
        last_fight_prob = 0.0

        while True:
            frame_start_time = time.time()
            ret, frame = cap.read()
            if not ret:
                break

            if len(frame_buffer) == CLIP_LEN:
                frames_to_predict = list(frame_buffer)
                class_name, fight_prob = predict_chunk(frames_to_predict, model, device_local, transform, class_names_loaded)
                if class_name:
                    last_class_name = class_name
                    last_fight_prob = fight_prob
                    is_fight_chunk = (class_name == 'Fight' and fight_prob >= PREDICTION_THRESHOLD)
                    recent_predictions.append(is_fight_chunk)
                    fight_detected_smoothed = len(recent_predictions) == SMOOTHING_WINDOW and all(recent_predictions)
                else:
                    last_class_name = "Pred Error"
                    last_fight_prob = 0.0

            display_frame = frame.copy()
            pred_color = (0, 0, 255) if last_class_name == 'Fight' else (0, 255, 0)
            cv2.putText(display_frame, f"Prediction: {last_class_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            cv2.putText(display_frame, f"Confidence: {last_fight_prob:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, pred_color, 2)
            actual_color = (0, 0, 255) if actual_category == 'Fight' else (0, 255, 0)
            cv2.putText(display_frame, f"Actual: {actual_category}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, actual_color, 2)
            state_text = "FIGHT DETECTED" if fight_detected_smoothed else "Normal"
            state_color = (0, 0, 255) if fight_detected_smoothed else (0, 255, 0)
            cv2.putText(display_frame, f"Smoothed State: {state_text}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, state_color, 2)
            if fight_detected_smoothed:
                cv2.rectangle(display_frame, (0, 0), (display_frame.shape[1], display_frame.shape[0]), (0, 0, 255), 5)

            ret, buffer = cv2.imencode('.jpg', display_frame)
            if not ret:
                continue
            yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

            elapsed_time = time.time() - frame_start_time
            sleep_time = target_frame_time - elapsed_time
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

        cap.release()
        print(f"Finished video: {video_path}")
        await asyncio.sleep(2)

    print("All videos streamed.")
    finish_img = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(finish_img, "Stream Finished", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    _, frame_bytes = cv2.imencode('.jpg', finish_img)
    yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + frame_bytes.tobytes() + b'\r\n')

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def read_root():
    html_content = f"""
    <html>
        <head>
            <title>Fight Detection Inference</title>
            <style>
                body {{ font-family: sans-serif; background-color: #f0f0f0; margin: 20px; }}
                h1 {{ text-align: center; color: #333; }}
                .video-container {{ display: flex; justify-content: center; margin-top: 20px; border: 1px solid #ccc; background-color: #fff; padding: 10px; box-shadow: 2px 2px 5px rgba(0,0,0,0.1); }}
                img {{ max-width: 90%; height: auto; }}
            </style>
        </head>
        <body>
            <h1>Fight Detection - Live Inference Simulation</h1>
            <div class="video-container">
                <img src="/video_feed" alt="Video Stream">
            </div>
            <p style="text-align: center; margin-top: 15px; color: #555;">
                Streaming up to {MAX_INFERENCE_VIDEOS} random videos from the validation set.<br>
                Model: {os.path.basename(MODEL_SAVE_PATH)} | Input: {INPUT_SIZE}x{INPUT_SIZE}@{FRAME_RATE}fps | Clip: {CLIP_LEN} frames
            </p>
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video_feed")
async def video_feed():
    return StreamingResponse(video_stream_generator(), media_type="multipart/x-mixed-replace; boundary=frame")