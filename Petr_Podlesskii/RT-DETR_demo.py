import os
import cv2
import contextlib
import torch
import time
import numpy as np
from collections import deque
from ultralytics import YOLO, RTDETR

# Initialize models silently
with open(os.devnull, 'w') as f:
    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
        rtdetr_model = RTDETR('rtdetr-l.pt')
        yolo_model = YOLO('yolov8n.pt')

# Initialize webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise IOError("Cannot open webcam")

# Initialize performance tracking
fps_deque = deque(maxlen=30)
rtdetr_times = deque(maxlen=30)
yolo_times = deque(maxlen=30)


def draw_rtdetr_stats(frame, fps, rtdetr_time):
    cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    rtdetr_avg = np.mean(list(rtdetr_times)) if rtdetr_times else 0
    cv2.putText(frame, f'RT-DETR: {rtdetr_avg*1000:.1f}ms', (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)


def draw_yolo_stats(frame, fps, yolo_time):
    cv2.rectangle(frame, (10, 10), (250, 80), (0, 0, 0), -1)
    cv2.putText(frame, f'FPS: {fps:.1f}', (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    yolo_avg = np.mean(list(yolo_times)) if yolo_times else 0
    cv2.putText(frame, f'YOLO: {yolo_avg*1000:.1f}ms', (20, 65),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)


def process_frame(frame, model, is_rtdetr=True):
    start_time = time.time()
    results = model(frame, verbose=False)  # Disable verbose output
    inference_time = time.time() - start_time

    if is_rtdetr:
        rtdetr_times.append(inference_time)
    else:
        yolo_times.append(inference_time)

    frame_with_boxes = frame.copy()
    color = (0, 255, 0) if is_rtdetr else (255, 0, 0)

    for r in results:
        boxes = r.boxes
        for box in boxes:
            b = box.xyxy[0].cpu().numpy().astype(int)
            conf = float(box.conf)
            cls = int(box.cls)
            class_name = model.names[cls]

            cv2.rectangle(frame_with_boxes, (b[0], b[1]), (b[2], b[3]), color, 2)
            label = f'{class_name} {conf:.2f}'
            cv2.putText(frame_with_boxes, label, (b[0], b[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    return frame_with_boxes, inference_time


running = True
while running:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        break

    frame_rtdetr, rtdetr_time = process_frame(frame, rtdetr_model, True)
    frame_yolo, yolo_time = process_frame(frame, yolo_model, False)

    fps = 1.0 / (time.time() - start_time)
    fps_deque.append(fps)
    avg_fps = np.mean(list(fps_deque))

    # Draw stats separately for each side
    draw_rtdetr_stats(frame_rtdetr, avg_fps, rtdetr_time)
    draw_yolo_stats(frame_yolo, avg_fps, yolo_time)

    combined_frame = np.hstack((frame_rtdetr, frame_yolo))
    cv2.imshow('RT-DETR vs YOLOv8 Comparison', combined_frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q') or cv2.getWindowProperty('RT-DETR vs YOLOv8 Comparison', cv2.WND_PROP_VISIBLE) < 1:
        running = False

cap.release()
cv2.destroyAllWindows()
