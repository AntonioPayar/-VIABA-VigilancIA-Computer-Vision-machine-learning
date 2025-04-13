# train_model.py - Longer Epochs Configuration

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from torchvision.models import video
from torch.cuda.amp import GradScaler, autocast # For Mixed Precision

import cv2
import os
import glob
import random
from tqdm import tqdm

# --- Configuration (Longer Epochs) ---
TRAIN_DIR = 'RWF-2000/train'
VAL_DIR = 'RWF-2000/val'
MODEL_SAVE_PATH = "fight_detector_model_long_epoch.pth"

# Model & Training Parameters (Keep fast input, train longer)
CLIP_LEN = 16           # Keep fast clip length
INPUT_SIZE = 112        # Keep fast resolution
FRAME_RATE = 15         # Moderate frame rate
PRETRAINED_MODEL = True
FREEZE_BACKBONE = False # Fine-tune the ENTIRE model
MODEL_ARCH = 'r2plus1d_18'

# Training Hyperparameters
NUM_EPOCHS = 55         # Target ~4-5 hours 
BATCH_SIZE = 8          # Keep batch size from working config
LEARNING_RATE = 5e-5    # Keep LR suitable for full fine-tuning
LR_STEP_SIZE = 18       # Adjust LR schedule step for more epochs 
LR_GAMMA = 0.1          # Standard decay factor
WEIGHT_DECAY = 1e-4     # Keep weight decay

# --- Dataset Class ---
class RWFDataset(Dataset):
    def __init__(self, root_dir, clip_len=16, frame_rate=15, size=224, mode='train'):
        self.root_dir = root_dir
        self.clip_len = clip_len
        self.frame_rate = frame_rate
        self.size = size
        self.mode = mode
        self.video_paths = []
        self.labels = []
        self.classes = {'NonFight': 0, 'Fight': 1}

        if not os.path.isdir(root_dir): raise ValueError(f"Dataset directory not found: {root_dir}")
        for class_name, label in self.classes.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.isdir(class_dir): continue
            extensions = ['*.avi', '*.mp4', '*.mkv']
            for ext in extensions:
                found_videos = glob.glob(os.path.join(class_dir, ext))
                self.video_paths.extend(found_videos)
                self.labels.extend([label] * len(found_videos))
        if not self.video_paths: raise ValueError(f"No videos found in {root_dir} with extensions {extensions}")

        kinetics_mean = [0.43216, 0.394666, 0.37645]
        kinetics_std = [0.22803, 0.22145, 0.216989]

        self.frame_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((size, size)), # Uses INPUT_SIZE (112)
            transforms.Normalize(mean=kinetics_mean, std=kinetics_std),
        ])

        if self.mode == 'train':
            self.video_transform = transforms.Compose([
                transforms.RandomHorizontalFlip(p=0.5),
            ])
        else:
            self.video_transform = transforms.Compose([])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return torch.zeros((3, self.clip_len, self.size, self.size)), torch.tensor(-1)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if original_fps <= 0: original_fps = 30

        frames = []
        indices_to_sample = []
        sampling_step = max(1.0, original_fps / self.frame_rate)

        if total_frames >= self.clip_len * sampling_step:
            max_start_frame = total_frames - int(self.clip_len * sampling_step)
            start_frame_idx = random.randint(0, max_start_frame) if self.mode == 'train' else max_start_frame // 2
        else:
            start_frame_idx = 0
            sampling_step = max(1.0, total_frames / self.clip_len)

        for i in range(self.clip_len):
            frame_index = int(start_frame_idx + i * sampling_step)
            indices_to_sample.append(min(frame_index, total_frames - 1))

        last_valid_frame = None
        current_frame_pos = -1
        indices_to_sample.sort()

        for target_idx in indices_to_sample:
            try:
                if target_idx != current_frame_pos + 1: cap.set(cv2.CAP_PROP_POS_FRAMES, target_idx)
                ret, frame = cap.read()
                current_frame_pos = target_idx
                if not ret or frame is None:
                    if last_valid_frame is None:
                        cap.release(); return torch.zeros((3, self.clip_len, self.size, self.size)), torch.tensor(label)
                    frame = last_valid_frame
                else:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    last_valid_frame = frame
                frame_tensor = self.frame_transform(frame)
                frames.append(frame_tensor)
            except Exception as e:
                 if last_valid_frame is not None: frames.append(self.frame_transform(last_valid_frame))
                 else:
                     cap.release(); return torch.zeros((3, self.clip_len, self.size, self.size)), torch.tensor(label)
        cap.release()

        while len(frames) < self.clip_len:
            if not frames: return torch.zeros((3, self.clip_len, self.size, self.size)), torch.tensor(label)
            frames.append(frames[-1])
        frames = frames[:self.clip_len]
        video_tensor = torch.stack(frames, dim=1)
        video_tensor = self.video_transform(video_tensor)
        return video_tensor, torch.tensor(label)


# --- Model Loading (Keep as is) ---
def load_model_for_training(model_arch='r2plus1d_18', num_classes=2, pretrained=True):
    print(f"Loading {model_arch} model... Pretrained={pretrained}")
    weights = video.R2Plus1D_18_Weights.DEFAULT if pretrained else None
    model = video.r2plus1d_18(weights=weights)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"Replaced final layer for {num_classes} classes.")
    for param in model.parameters():
        param.requires_grad = True
    print("All model parameters set to require gradients (full fine-tuning).")
    return model

# --- Training Function  ---
def train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs):
    scaler = GradScaler("cuda")
    best_val_acc = 0.0
    print(f"\n--- Starting Training for {num_epochs} Epochs ---")

    for epoch in range(num_epochs):
        print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")
        # --- Training Phase ---
        model.train()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        progress_bar = tqdm(train_loader, desc="Training", leave=False)
        for inputs, labels in progress_bar:
            inputs, labels = inputs.to(device), labels.to(device)
            if (labels == -1).any(): continue
            optimizer.zero_grad()
            with autocast("cuda"):
                outputs = model(inputs)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total_preds += labels.size(0)
            correct_preds += (predicted == labels).sum().item()
            progress_bar.set_postfix(loss=loss.item())
        epoch_loss = running_loss / total_preds if total_preds > 0 else 0
        epoch_acc = correct_preds / total_preds if total_preds > 0 else 0
        print(f"Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        # --- Validation Phase ---
        model.eval()
        running_loss, correct_preds, total_preds = 0.0, 0, 0
        progress_bar_val = tqdm(val_loader, desc="Validation", leave=False)
        with torch.no_grad():
             for inputs, labels in progress_bar_val:
                inputs, labels = inputs.to(device), labels.to(device)
                if (labels == -1).any(): continue
                with autocast("cuda"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total_preds += labels.size(0)
                correct_preds += (predicted == labels).sum().item()
                progress_bar_val.set_postfix(loss=loss.item())
        epoch_loss = running_loss / total_preds if total_preds > 0 else 0
        epoch_acc = correct_preds / total_preds if total_preds > 0 else 0
        print(f"Val Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if epoch_acc > best_val_acc:
            best_val_acc = epoch_acc
            print(f"*** New best validation accuracy: {best_val_acc:.4f}. Saving model to {MODEL_SAVE_PATH} ***")
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
        else:
            print(f"(Validation accuracy did not improve from {best_val_acc:.4f})")
        scheduler.step()
        print(f"Current LR: {scheduler.get_last_lr()}")

    print(f"\n--- Training finished ---")
    print(f"Best Validation Accuracy achieved: {best_val_acc:.4f}")
    print(f"Final model saved to {MODEL_SAVE_PATH}")


# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting Training Script (Longer Epoch Config) ---")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    class_names = ['NonFight', 'Fight']
    num_classes = len(class_names)

    model = load_model_for_training(
        model_arch=MODEL_ARCH,
        num_classes=num_classes,
        pretrained=PRETRAINED_MODEL
    )
    model = model.to(device)

    print("Setting up datasets...")
    try:
        train_dataset = RWFDataset(TRAIN_DIR, clip_len=CLIP_LEN, frame_rate=FRAME_RATE, size=INPUT_SIZE, mode='train')
        val_dataset = RWFDataset(VAL_DIR, clip_len=CLIP_LEN, frame_rate=FRAME_RATE, size=INPUT_SIZE, mode='val')
    except ValueError as e: print(f"Error initializing dataset: {e}"); exit()
    if len(train_dataset) == 0 or len(val_dataset) == 0: print("Error: Training or validation dataset is empty."); exit()

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    print(f"Using AdamW optimizer with LR={LEARNING_RATE}, Weight Decay={WEIGHT_DECAY}")
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    print(f"Using StepLR scheduler with step_size={LR_STEP_SIZE}, gamma={LR_GAMMA}")

    train(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=NUM_EPOCHS)

    print("--- Training Script Finished ---")