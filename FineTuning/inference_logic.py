# inference_logic.py (Revised)

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import video
from torch.cuda.amp import autocast # Use torch.amp.autocast if using PyTorch 1.6+
import cv2
import os

# --- Configuration ---
MODEL_SAVE_PATH = "fight_detector_model_long_epoch.pth"
CLIP_LEN = 16
INPUT_SIZE = 112
FRAME_RATE = 10
FPS_TO_PROCESS = FRAME_RATE
PREDICTION_THRESHOLD = 0.7
SMOOTHING_WINDOW = 2
VAL_DIR = 'RWF-2000/val'
MAX_INFERENCE_VIDEOS = 10

class_names_loaded = ['NonFight', 'Fight']

# --- Model Loading Function ---
def load_trained_model(device):
    print("Loading model architecture...")
    num_classes = len(class_names_loaded)
    model_arch = video.r2plus1d_18(weights=None)
    num_ftrs = model_arch.fc.in_features
    model_arch.fc = nn.Linear(num_ftrs, num_classes)

    if not os.path.exists(MODEL_SAVE_PATH):
        print(f"FATAL ERROR: Trained model not found at '{MODEL_SAVE_PATH}'.")
        return None
    try:
        state_dict = torch.load(MODEL_SAVE_PATH, map_location=device)
        model_arch.load_state_dict(state_dict)
        model = model_arch.to(device)
        model.eval()
        print(f"Successfully loaded trained model from {MODEL_SAVE_PATH}")
        return model
    except Exception as e:
        print(f"FATAL ERROR: Failed to load model weights from {MODEL_SAVE_PATH}: {e}")
        return None

# --- Inference Transform Definition ---
def get_inference_transform():
    kinetics_mean = [0.43216, 0.394666, 0.37645]
    kinetics_std = [0.22803, 0.22145, 0.216989]
    print("Creating inference transform.")
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((INPUT_SIZE, INPUT_SIZE)),
        transforms.Normalize(mean=kinetics_mean, std=kinetics_std),
    ])

# --- Inference Chunk Prediction ---
def predict_chunk(chunk_frames, model, device, transform, class_names):
    if len(chunk_frames) != CLIP_LEN:
        while len(chunk_frames) < CLIP_LEN:
             if not chunk_frames: return None, None
             chunk_frames.append(chunk_frames[-1])
        chunk_frames = list(chunk_frames)[:CLIP_LEN]

    try:
        processed_frames = [transform(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in chunk_frames]
        video_tensor = torch.stack(processed_frames, dim=1).unsqueeze(0)
    except Exception as e:
        print(f"Error during inference transformation: {e}")
        return None, None

    video_tensor = video_tensor.to(device)
    with torch.no_grad():
        with autocast(): # Or torch.amp.autocast('cuda'):
            outputs = model(video_tensor)
        probabilities = torch.softmax(outputs.float(), dim=1)
        fight_prob = probabilities[0, class_names.index('Fight')].item()
        predicted_class_idx = torch.argmax(probabilities, dim=1).item()
        predicted_class_name = class_names[predicted_class_idx]
    return predicted_class_name, fight_prob
