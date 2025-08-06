import os
import re
import math
from pathlib import Path
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, random_split
from torchvision import transforms
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
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

# === DATASET ===
class WoWSequenceDataset(Dataset):
    """Custom dataset for loading sequences of frames and actions."""
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.transform = transform
        self.sequence_length = sequence_length

        frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir) if f.endswith(".jpg")])
        actions = np.load(actions_file).astype(np.float32)

        min_len = min(len(frame_paths), len(actions))
        self.frame_paths = frame_paths[:min_len]
        self.actions = actions[:min_len]

        self.indices = []
        num_keys = len(COMMON_KEYS)
        action_frames = 0

        for i in range(len(self.frame_paths) - self.sequence_length + 1):
            last_action = self.actions[i + self.sequence_length - 1]
            key_press = np.sum(last_action[:num_keys]) > 0
            mouse_click = np.sum(last_action[num_keys+2:]) > 0

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

        imgs = []
        for i in range(start_index, end_index):
            img = cv2.imread(self.frame_paths[i])
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(img)
            imgs.append(img)

        seq_actions = self.actions[start_index:end_index]
        return torch.stack(imgs), torch.tensor(seq_actions, dtype=torch.float32)

# === NEW: POSITIONAL ENCODING ===
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        x: Tensor, shape [batch_size, seq_len, embedding_dim]
        """
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

# === UPDATED MODEL: TRANSFORMER ===
class BehaviorCloningTransformer(nn.Module):
    """CNN-Transformer model for behavior cloning."""
    def __init__(self, output_dim, d_model, nhead, nlayers, dropout):
        super().__init__()
        # 1. CNN Feature Extractor
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out_size = self.cnn(dummy_input).shape[1]

        # 2. Projection layer to match transformer's d_model
        self.input_proj = nn.Linear(cnn_out_size, d_model)
        
        # 3. Positional Encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_len=SEQUENCE_LENGTH)
        
        # 4. Transformer Encoder
        encoder_layers = nn.TransformerEncoderLayer(d_model, nhead, dropout=dropout, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, nlayers)
        
        self.d_model = d_model

        # 5. Output heads
        self.key_head = nn.Sequential(nn.Linear(d_model, len(COMMON_KEYS)))
        self.mouse_pos_head = nn.Sequential(nn.Linear(d_model, 2), nn.Sigmoid())
        self.mouse_click_head = nn.Sequential(nn.Linear(d_model, 2))

    def forward(self, x):
        b, s, c, h, w = x.shape
        x_reshaped = x.view(b * s, c, h, w)
        
        # Pass through CNN
        feat = self.cnn(x_reshaped)
        feat_reshaped = feat.view(b, s, -1)
        
        # Project and add positional encoding
        projected_feat = self.input_proj(feat_reshaped) * math.sqrt(self.d_model)
        pos_encoded_feat = self.pos_encoder(projected_feat)
        
        # Pass through Transformer
        transformer_out = self.transformer_encoder(pos_encoded_feat)
        
        # Pass through output heads
        key_out = self.key_head(transformer_out)
        pos_out = self.mouse_pos_head(transformer_out)
        click_out = self.mouse_click_head(transformer_out)
        
        # Concatenate for loss calculation
        return torch.cat([key_out, pos_out, click_out], dim=2)

# === LOSS FUNCTION ===
def weighted_bce_mse_loss(outputs, targets):
    bce_loss = nn.BCEWithLogitsLoss()
    mse_loss = nn.MSELoss()
    num_keys = len(COMMON_KEYS)
    
    key_out, click_out = outputs[..., :num_keys], outputs[..., num_keys+2:]
    key_tgt, click_tgt = targets[..., :num_keys], targets[..., num_keys+2:]
    pos_out = outputs[..., num_keys:num_keys+2]
    pos_tgt = targets[..., num_keys:num_keys+2]
    
    loss_keys = bce_loss(key_out, key_tgt)
    loss_clicks = bce_loss(click_out, click_tgt)
    loss_pos = mse_loss(pos_out, pos_tgt)
    
    return loss_keys + loss_clicks + (loss_pos * 5) # Weight mouse position loss higher

# === VALIDATION ===
def validate(model, dataloader, device, writer, epoch):
    model.eval()
    all_preds, all_tgts = [], []

    with torch.no_grad():
        for seqs, acts in dataloader:
            seqs, acts = seqs.to(device), acts.to(device)
            out = model(seqs)
            
            # Aggregate predictions and targets over a small window at the end of the sequence
            preds = torch.sigmoid(out[:, -VALIDATION_WINDOW:, :]).mean(dim=1)
            tgts = acts[:, -VALIDATION_WINDOW:, :].max(dim=1)[0]

            all_preds.append(preds.cpu().numpy())
            all_tgts.append(tgts.cpu().numpy())

    all_preds = np.vstack(all_preds)
    all_tgts = np.vstack(all_tgts)

    num_keys = len(COMMON_KEYS)
    key_preds, click_preds = all_preds[:, :num_keys], all_preds[:, num_keys+2:]
    key_tgts, click_tgts = all_tgts[:, :num_keys], all_tgts[:, num_keys+2:]

    best_f1 = 0.0
    best_thresh = 0.5
    for thresh in THRESHOLD_SWEEP:
        binary_preds = (np.hstack([key_preds, click_preds]) > thresh).astype(int)
        binary_tgts = np.hstack([key_tgts, click_tgts])
        _, _, f1, _ = precision_recall_fscore_support(binary_tgts, binary_preds, average="samples", zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = thresh

    print(f"\nValidation Best F1: {best_f1:.4f} at threshold {best_thresh:.2f}")

    cm_preds = (np.hstack([key_preds, click_preds]) > best_thresh).astype(int).flatten()
    cm_tgts = np.hstack([key_tgts, click_tgts]).flatten()
    cm = confusion_matrix(cm_tgts, cm_preds)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=['No Action', 'Action'], yticklabels=['No Action', 'Action'])
    ax.set_title(f"Confusion Matrix (Thresh={best_thresh:.2f})")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    writer.add_figure("Validation/ConfusionMatrix", fig, epoch)
    plt.close(fig)

    return best_f1, best_thresh

# === MAIN TRAINING LOOP ===
def train():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    early_stopper = EarlyStopping()
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    datasets = [ds for d in DATA_DIRS if (fdir := os.path.join(d, "frames")) and (afile := os.path.join(d, ACTIONS_FILE)) and os.path.exists(fdir) and os.path.exists(afile) and len(ds := WoWSequenceDataset(fdir, afile, SEQUENCE_LENGTH, transform)) > 0]
    if not datasets:
        print("❌ No valid datasets found! Check DATA_DIRS in config.py.")
        return

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])
    print(f"\nTotal sequences: {len(full_dataset)} | Train: {len(train_ds)} | Val: {len(val_ds)}")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dim = len(COMMON_KEYS) + 4
    
    # Initialize the new Transformer model
    model = BehaviorCloningTransformer(output_dim, D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    start_epoch = 1
    best_f1 = 0.0
    if os.path.exists(MODEL_SAVE_DIR) and (checkpoint_files := list(Path(MODEL_SAVE_DIR).glob("model_epoch_*.pth"))):
        latest_checkpoint_path = max(checkpoint_files, key=os.path.getctime)
        if match := re.search(r"model_epoch_(\d+).pth", latest_checkpoint_path.name):
            try:
                checkpoint = torch.load(latest_checkpoint_path, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                start_epoch = checkpoint['epoch'] + 1
                best_f1 = checkpoint.get('best_f1', 0.0)
                print(f"\n✅ Resuming training from {latest_checkpoint_path} at epoch {start_epoch}. Best F1: {best_f1:.4f}\n")
            except Exception as e:
                print(f"\n⚠️ Could not load checkpoint {latest_checkpoint_path}. Starting fresh. Error: {e}\n")

    print(f"\n🚀 Starting training on {device} for {EPOCHS} epochs…\n")

    try:
        for epoch in range(start_epoch, EPOCHS + 1):
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

            val_f1, best_thresh = validate(model, val_loader, device, writer, epoch)
            writer.add_scalar("F1/validation", val_f1, epoch)
            writer.add_scalar("LearningRate", optimizer.param_groups[0]['lr'], epoch)
            scheduler.step(val_f1)

            early_stopper(val_f1)
            if early_stopper.early_stop:
                print(f"🛑 Early stopping triggered at epoch {epoch}.")
                break

            ckpt_path = MODEL_SAVE_PATH_TEMPLATE.format(epoch)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_f1': best_f1,
            }, ckpt_path)

            if val_f1 > best_f1:
                best_f1 = val_f1
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_f1': best_f1,
                }, MODEL_FILE)
                print(f"⭐ New best model saved to {MODEL_FILE} (F1={best_f1:.4f})")

                best_threshold_path = os.path.join(os.path.dirname(MODEL_FILE) or ".", "best_threshold.txt")
                try:
                    with open(best_threshold_path, "w") as f: f.write(str(best_thresh))
                    print(f"   Saved best threshold ({best_thresh:.2f}) to {best_threshold_path}")
                except Exception as e:
                    print(f"   Could not save best threshold: {e}")

    except KeyboardInterrupt:
        print("\n⏹ Training interrupted by user. Saving final state...")
    finally:
        writer.close()
        print("\n✅ Training complete. TensorBoard logs saved.")

if __name__ == "__main__":
    train()
