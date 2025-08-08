import os
import re
import math
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
from torchvision import transforms
from sklearn.metrics import f1_score, confusion_matrix
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import seaborn as sns
from config import *

# --- EARLY STOPPING ---
class EarlyStopping:
    def __init__(self, patience=EARLY_STOPPING_PATIENCE, min_delta=EARLY_STOPPING_MIN_DELTA):
        self.patience, self.min_delta, self.counter, self.best_score, self.early_stop = patience, min_delta, 0, None, False
    def __call__(self, metric):
        if self.best_score is None or metric > self.best_score + self.min_delta:
            self.best_score, self.counter = metric, 0
        else: self.counter += 1
        if self.counter >= self.patience: self.early_stop = True

# --- DATASET ---
class ActionSequenceDataset(Dataset):
    def __init__(self, frame_dir, actions_file, sequence_length, transform=None):
        self.transform, self.sequence_length = transform, sequence_length
        frame_paths = sorted([os.path.join(frame_dir, f) for f in os.listdir(frame_dir)])
        actions = np.load(actions_file).astype(np.float32)
        min_len = min(len(frame_paths), len(actions))
        self.frame_paths, self.actions = frame_paths[:min_len], actions[:min_len]
    def __len__(self): return len(self.frame_paths) - self.sequence_length + 1
    def __getitem__(self, idx):
        end_idx = idx + self.sequence_length
        imgs = [self.transform(cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)) for p in self.frame_paths[idx:end_idx]]
        return torch.stack(imgs), torch.tensor(self.actions[idx:end_idx], dtype=torch.float32)

# --- MODEL ARCHITECTURE ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__(); self.dropout = nn.Dropout(p=dropout); pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1); div = torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div); self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:,:x.size(1)])

class BehaviorCloningTransformer(nn.Module):
    def __init__(self, d_model, nhead, nlayers, dropout):
        super().__init__()
        self.cnn = nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.BatchNorm2d(32),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.BatchNorm2d(64),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.BatchNorm2d(128),nn.ReLU(),nn.AdaptiveAvgPool2d((6,6)),nn.Flatten())
        cnn_out_size = self.cnn(torch.zeros(1,3,IMG_HEIGHT,IMG_WIDTH)).shape[1]
        self.input_proj = nn.Linear(cnn_out_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, SEQUENCE_LENGTH)
        encoder_layer = nn.TransformerEncoderLayer(d_model,nhead,dropout=dropout,batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, nlayers)
        self.d_model = d_model
        self.key_head = nn.Sequential(nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model//2,len(COMMON_KEYS)))
        # Mouse head now uses Tanh for outputs between -1 and 1, ideal for deltas
        self.mouse_delta_head = nn.Sequential(nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model//2,2),nn.Tanh())
        self.mouse_click_head = nn.Sequential(nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model//2,2))

    def forward(self, x):
        b,s,c,h,w = x.shape; feat = self.cnn(x.view(b*s,c,h,w)).view(b,s,-1)
        proj = self.input_proj(feat)*math.sqrt(self.d_model); enc = self.pos_encoder(proj)
        t_out = self.transformer_encoder(enc)
        key,delta,click = self.key_head(t_out),self.mouse_delta_head(t_out),self.mouse_click_head(t_out)
        return torch.cat([key, delta, click], dim=2)

# --- LOSS & VALIDATION ---
def weighted_loss(outputs, targets):
    num_keys = len(COMMON_KEYS)
    key_out, delta_out, click_out = outputs[...,:num_keys], outputs[...,num_keys:num_keys+2], outputs[...,num_keys+2:num_keys+4]
    key_tgt, delta_tgt, click_tgt = targets[...,:num_keys], targets[...,num_keys:num_keys+2], targets[...,num_keys+2:num_keys+4]
    loss_keys = nn.BCEWithLogitsLoss()(key_out, key_tgt)
    loss_clicks = nn.BCEWithLogitsLoss()(click_out, click_tgt)
    loss_deltas = nn.MSELoss()(delta_out, delta_tgt)
    return loss_keys + 2.0 * loss_clicks + 0.5 * loss_deltas

def validate(model, dataloader, device):
    model.eval()
    all_preds, all_tgts = [], []
    num_keys = len(COMMON_KEYS)
    with torch.no_grad():
        for seqs, acts in dataloader:
            seqs, acts = seqs.to(device), acts.to(device)
            out = model(seqs)[:, -1, :] # Only evaluate the last action in the sequence
            
            key_preds = torch.sigmoid(out[..., :num_keys])
            click_preds = torch.sigmoid(out[..., num_keys+2:num_keys+4])
            
            key_tgts = acts[:, -1, :num_keys]
            click_tgts = acts[:, -1, num_keys+2:num_keys+4]
            
            all_preds.append(torch.cat([key_preds, click_preds], dim=1).cpu().numpy())
            all_tgts.append(torch.cat([key_tgts, click_tgts], dim=1).cpu().numpy())

    all_preds, all_tgts = np.vstack(all_preds), np.vstack(all_tgts)
    best_f1, best_thresh = 0.0, 0.5
    for thresh in THRESHOLD_SWEEP:
        f1 = f1_score(all_tgts, (all_preds > thresh).astype(int), average="samples", zero_division=0)
        if f1 > best_f1: best_f1, best_thresh = f1, thresh
    return best_f1, best_thresh

# --- MAIN TRAINING LOOP ---
def train():
    writer = SummaryWriter(log_dir=TENSORBOARD_LOG_DIR)
    early_stopper = EarlyStopping()
    
    # Added ColorJitter for data augmentation
    transform = transforms.Compose([transforms.ToPILImage(), transforms.ColorJitter(brightness=0.3, contrast=0.2, saturation=0.2), transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
    
    datasets = [ActionSequenceDataset(os.path.join(d, "frames"), os.path.join(d, ACTIONS_FILE), SEQUENCE_LENGTH, transform) for d in DATA_DIRS if os.path.exists(os.path.join(d, "frames"))]
    if not datasets: print("❌ No valid datasets found!"); return

    full_dataset = ConcatDataset(datasets)
    print(f"Found {len(full_dataset)} total sequences.")
    
    # Sequential split for time-series data
    val_size = int(len(full_dataset) * VALIDATION_SPLIT)
    train_size = len(full_dataset) - val_size
    train_ds = Subset(full_dataset, range(train_size))
    val_ds = Subset(full_dataset, range(train_size, len(full_dataset)))
    print(f"Train size: {len(train_ds)}, Validation size: {len(val_ds)}")

    train_loader = DataLoader(train_ds, BATCH_SIZE, shuffle=True, num_workers=os.cpu_count(), pin_memory=True)
    val_loader = DataLoader(val_ds, BATCH_SIZE, num_workers=os.cpu_count(), pin_memory=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BehaviorCloningTransformer(D_MODEL, N_HEAD, N_LAYERS, DROPOUT).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    best_f1 = 0.0
    for epoch in range(1, EPOCHS + 1):
        model.train(); running_loss = 0.0
        for seqs, acts in train_loader:
            seqs, acts = seqs.to(device), acts.to(device)
            optimizer.zero_grad()
            outputs = model(seqs)
            loss = weighted_loss(outputs, acts)
            loss.backward(); optimizer.step(); running_loss += loss.item()
        
        avg_loss = running_loss / len(train_loader)
        val_f1, best_thresh = validate(model, val_loader, device)
        writer.add_scalar("Loss/train", avg_loss, epoch)
        writer.add_scalar("F1/validation", val_f1, epoch)
        print(f"Epoch {epoch}/{EPOCHS} | Loss: {avg_loss:.4f} | Val F1: {val_f1:.4f} @ Thresh: {best_thresh:.2f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save({'model_state_dict': model.state_dict()}, MODEL_FILE)
            print(f"⭐ New best model saved to {MODEL_FILE}")
            with open(os.path.join(os.path.dirname(MODEL_FILE) or ".", "best_threshold.txt"), "w") as f: f.write(str(best_thresh))

        early_stopper(val_f1)
        if early_stopper.early_stop: print(f"🛑 Early stopping at epoch {epoch}."); break
    writer.close()

if __name__ == "__main__":
    train()