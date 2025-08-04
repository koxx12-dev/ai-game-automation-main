import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms, utils as tv_utils
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import subprocess
import socket
import random
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from config import * # Import all settings

# === EARLY STOPPING ===
class EarlyStopping:
    """Stops training when a monitored metric has stopped improving."""
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, metric):
        if self.best_score is None:
            self.best_score = metric
            return
        if metric < self.best_score + self.min_delta:
            self.counter += 1
        else:
            self.best_score = metric
            self.counter = 0
        if self.counter >= self.patience:
            self.early_stop = True

# === DATASET WITH OVERSAMPLING ===
class WoWSequenceDataset(Dataset):
    """Custom dataset for loading sequences of frames and actions."""
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length

        frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        actions = np.load(actions_file).astype(np.float32)

        # Ensure frames and actions align
        min_len = min(len(frame_paths), len(actions))
        self.frame_paths = frame_paths[:min_len]
        self.actions = actions[:min_len]

        # Oversample sequences where an action (key press or mouse click) occurs
        self.indices = []
        num_keys = len(COMMON_KEYS)
        action_frames = 0

        for i in range(len(self.frame_paths) - self.sequence_length + 1):
            # Check the last frame in the sequence for an action
            last_action = self.actions[i + self.sequence_length - 1]
            key_press = np.sum(last_action[:num_keys]) > 0
            mouse_click = np.sum(last_action[num_keys+2:]) > 0 # Mouse buttons are after pos

            if key_press or mouse_click:
                self.indices.extend([i] * OVERSAMPLE_ACTION_FRAMES_MULTIPLIER)
                action_frames += 1
            else:
                self.indices.append(i)

        print(f"Loaded from {frame_dir}: {len(self.frame_paths)} frames, "
              f"{action_frames} action sequences -> {len(self.indices)} total sequences after oversampling.")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        start_index = self.indices[idx]
        end_index = start_index + self.sequence_length

        # Load sequence of images
        imgs = []
        for i in range(start_index, end_index):
            img = cv2.imread(self.frame_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        seq_actions = self.actions[start_index:end_index]
        return torch.stack(imgs), torch.tensor(seq_actions, dtype=torch.float32)

# === MODEL DEFINITION ===
class BehaviorCloningCNNRNN(nn.Module):
    """CNN-LSTM model for behavior cloning."""
    def __init__(self, output_dim):
        super().__init__()
        # CNN for feature extraction from images
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )

        # Calculate CNN output size dynamically
        with torch.no_grad():
            # FIX: Use the correct image dimensions for the dummy input
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # LSTM for processing sequences of features
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )

        # Output heads for different actions
        self.key_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, len(COMMON_KEYS)) # No sigmoid, use BCEWithLogitsLoss
        )
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid() # Position is normalized (0-1)
        )
        self.mouse_click_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, 2) # No sigmoid, use BCEWithLogitsLoss
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        # Pass each frame in the sequence through the CNN
        x_reshaped = x.view(b * s, c, h, w)
        feat = self.cnn(x_reshaped)
        
        # Reshape for LSTM: (batch, seq_len, features)
        feat_reshaped = feat.view(b, s, -1)
        
        # Pass sequence through LSTM
        lstm_out, _ = self.lstm(feat_reshaped)
        
        # Apply heads to each time step's output
        lstm_out_reshaped = lstm_out.reshape(b * s, -1)
        
        key_out = self.key_head(lstm_out_reshaped)
        pos_out = self.mouse_pos_head(lstm_out_reshaped)
        click_out = self.mouse_click_head(lstm_out_reshaped)
        
        # Combine outputs and reshape back to sequence
        concat = torch.cat([key_out, pos_out, click_out], dim=1)
        return concat.view(b, s, -1)

# === LOSS FUNCTION ===
def weighted_bce_mse_loss(outputs, targets):
    """Calculates a combined loss for keys, clicks, and mouse position."""
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()

    num_keys = len(COMMON_KEYS)
    
    # Key and click predictions (use BCEWithLogitsLoss)
    key_out, click_out = outputs[..., :num_keys], outputs[..., num_keys+2:]
    key_tgt, click_tgt = targets[..., :num_keys], targets[..., num_keys+2:]
    
    # Mouse position predictions (use MSELoss)
    pos_out = outputs[..., num_keys:num_keys+2]
    pos_tgt = targets[..., num_keys:num_keys+2]
    
    loss_keys = bce_loss(key_out, key_tgt)
    loss_clicks = bce_loss(click_out, click_tgt)
    loss_pos = mse_loss(pos_out, pos_tgt)
    
    return loss_keys + loss_clicks + loss_pos

# === VALIDATION & TENSORBOARD UTILS ===
def validate(model, dataloader, device, writer, epoch):
    """Evaluates the model on the validation set."""
    model.eval()
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for seqs, acts in dataloader:
            seqs, acts = seqs.to(device), acts.to(device)
            out = model(seqs)

            # Aggregate predictions and targets over the validation window
            # Use sigmoid to convert logits to probabilities for metric calculation
            preds = torch.sigmoid(out[:, -VALIDATION_WINDOW:, :]).mean(dim=1)
            tgts = acts[:, -VALIDATION_WINDOW:, :].max(dim=1)[0]

            all_preds.append(preds.cpu().numpy())
            all_tgts.append(tgts.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_tgts = np.vstack(all_tgts)

    # Separate keys and clicks for F1 score calculation
    num_keys = len(COMMON_KEYS)
    key_preds = all_preds[:, :num_keys]
    click_preds = all_preds[:, num_keys+2:]
    key_tgts = all_tgts[:, :num_keys]
    click_tgts = all_tgts[:, num_keys+2:]

    # Find best threshold for F1 score
    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in THRESHOLD_SWEEP:
        binary_preds = (np.hstack([key_preds, click_preds]) > thresh).astype(int)
        binary_tgts = np.hstack([key_tgts, click_tgts])
        
        _, _, f1, _ = precision_recall_fscore_support(
            binary_tgts, binary_preds, average="samples", zero_division=0
        )
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nValidation Best F1: {best_f1:.4f} at threshold {best_thresh:.2f}")

    # Log confusion matrix with the best threshold
    cm_preds = (np.hstack([key_preds, click_preds]) > best_thresh).astype(int).flatten()
    cm_tgts = np.hstack([key_tgts, click_tgts]).flatten()
    cm = confusion_matrix(cm_tgts, cm_preds)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=['No Action', 'Action'], yticklabels=['No Action', 'Action'])
    ax.set_title(f"Confusion Matrix (Thresh={best_thresh:.2f})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    writer.add_figure("Validation/ConfusionMatrix", fig, epoch)
    plt.close(fig)

    return best_f1, best_thresh

def find_free_port(start=6006, end=6099):
    """Finds an available port for TensorBoard."""
    # ... (implementation is the same)
    return 6006 # Placeholder

# === MAIN TRAINING LOOP ===
def train():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    early_stopper = EarlyStopping()
    
    # FIX: Define image transformations using the correct, unified dimensions from config
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), # This now uses the correct 224x224
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Load all specified datasets
    datasets = []
    for d in DATA_DIRS:
        fdir = os.path.join(d, "frames")
        afile = os.path.join(d, ACTIONS_FILE)
        if os.path.exists(fdir) and os.path.exists(afile):
            ds = WoWSequenceDataset(fdir, afile, SEQUENCE_LENGTH, transform)
            if len(ds) > 0:
                datasets.append(ds)
    if not datasets:
        print("❌ No valid datasets found! Check DATA_DIRS in config.py.")
        return

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"\nTotal sequences: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    num_workers_to_use = 4 
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers_to_use, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers_to_use, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = len(COMMON_KEYS) + 4 # keys + mouse (x, y, l_click, r_click)
    model = BehaviorCloningCNNRNN(output_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    best_f1 = 0.0
    print(f"\n🚀 Starting training on {device} for {EPOCHS} epochs…\n")

    try:
        for epoch in range(1, EPOCHS + 1):
            model.train()
            running_loss = 0.0
            for i, (seqs, acts) in enumerate(train_loader, 1):
                seqs, acts = seqs.to(device), acts.to(device)
                
                optimizer.zero_grad()
                outputs = model(seqs)
                loss = weighted_bce_mse_loss(outputs, acts)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                
                running_loss += loss.item()
                if i % 100 == 0:
                    step = (epoch - 1) * len(train_loader) + i
                    writer.add_scalar("Loss/train_batch", loss.item(), step)
                    print(f"  Epoch {epoch}/{EPOCHS} | Step {i}/{len(train_loader)} | Loss: {loss.item():.4f}")

            avg_loss = running_loss / len(train_loader)
            writer.add_scalar("Loss/train_epoch", avg_loss, epoch)
            print(f"Epoch {epoch} Summary | Avg Loss: {avg_loss:.4f}")

            # Validation
            # MODIFICATION: Receive both F1 score and the best threshold
            val_f1, best_thresh = validate(model, val_loader, device, writer, epoch) #
            writer.add_scalar("F1/validation", val_f1, epoch) #
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch) #
            scheduler.step(val_f1) #

            # Early stopping check
            early_stopper(val_f1) #
            if early_stopper.early_stop: #
                print(f"🛑 Early stopping triggered at epoch {epoch}.") #
                break

            # Save checkpoint
            ckpt_path = MODEL_SAVE_PATH_TEMPLATE.format(epoch) #
            torch.save(model.state_dict(), ckpt_path) #

            # Save best model
            if val_f1 > best_f1: #
                best_f1 = val_f1 #
                torch.save(model.state_dict(), MODEL_FILE) #
                print(f"⭐ New best model saved to {MODEL_FILE} (F1={best_f1:.4f})") #

                # Save the best threshold to a file alongside the model
                best_threshold_path = os.path.join(os.path.dirname(MODEL_FILE), "best_threshold.txt") if os.path.dirname(MODEL_FILE) else "best_threshold.txt"
                try:
                    with open(best_threshold_path, "w") as f:
                        f.write(str(best_thresh))
                    print(f"   Saved best threshold ({best_thresh:.2f}) to {best_threshold_path}")
                except Exception as e:
                    print(f"   Could not save best threshold: {e}")

    except KeyboardInterrupt: #
        print("\n⏹ Training interrupted by user. Saving final state...") #
    finally:
        writer.close() #
        print("\n✅ Training complete. TensorBoard logs saved.") #

if __name__ == "__main__":
    train()

