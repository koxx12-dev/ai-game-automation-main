import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
import math
import os
from collections import deque
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from torchvision import transforms
from config import *

# --- SCREEN DIMENSIONS ---
with mss.mss() as sct: monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]

# --- MODEL ARCHITECTURE (Must match training script) ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=50):
        super().__init__(); self.dropout=nn.Dropout(p=dropout); pe=torch.zeros(max_len,d_model)
        pos=torch.arange(max_len).unsqueeze(1); div=torch.exp(torch.arange(0,d_model,2)*(-math.log(10000.0)/d_model))
        pe[:,0::2],pe[:,1::2]=torch.sin(pos*div),torch.cos(pos*div); self.register_buffer('pe',pe.unsqueeze(0))
    def forward(self, x): return self.dropout(x + self.pe[:, :x.size(1)])

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
        self.mouse_delta_head = nn.Sequential(nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model//2,2),nn.Tanh())
        self.mouse_click_head = nn.Sequential(nn.Linear(d_model,d_model//2),nn.ReLU(),nn.Dropout(dropout),nn.Linear(d_model//2,2))

    def forward(self, x):
        b,s,c,h,w = x.shape; feat = self.cnn(x.view(b*s,c,h,w)).view(b,s,-1)
        proj = self.input_proj(feat)*math.sqrt(self.d_model); enc = self.pos_encoder(proj)
        t_out = self.transformer_encoder(enc)
        key,delta,click = self.key_head(t_out),self.mouse_delta_head(t_out),self.mouse_click_head(t_out)
        return torch.cat([key, delta, click], dim=2)

# --- SETUP ---
device = torch.device("cpu")
model = BehaviorCloningTransformer(D_MODEL, N_HEAD, N_LAYERS, DROPOUT)
try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device)['model_state_dict'])
    threshold_path = os.path.join(os.path.dirname(MODEL_FILE) or ".", "best_threshold.txt")
    if os.path.exists(threshold_path):
        with open(threshold_path, 'r') as f: action_threshold = float(f.read().strip())
    else: action_threshold = 0.5
    print(f"✅ Model '{MODEL_FILE}' loaded. Threshold: {action_threshold:.2f}")
except Exception as e: print(f"❌ Error loading model: {e}"); exit(1)

model.eval()
k_ctrl, m_ctrl = KeyboardController(), MouseController()
running, ai_enabled = True, False

# --- STATE & UTILS ---
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
pressed_keys, pressed_buttons = set(), set()
transform = transforms.Compose([transforms.ToPILImage(), transforms.Resize((IMG_HEIGHT, IMG_WIDTH)), transforms.ToTensor(), transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])])
def on_press(key):
    global running, ai_enabled
    if key == keyboard.Key.f12: running = False
    elif key == keyboard.Key.f10:
        ai_enabled = not ai_enabled
        if not ai_enabled: release_all_inputs()
        print(f"🤖 AI control: {'ENABLED' if ai_enabled else 'DISABLED'}")
def release_all_inputs():
    for k in list(pressed_keys): k_ctrl.release(KEY_MAPPING.get(k)); pressed_keys.remove(k)
    for b in list(pressed_buttons): m_ctrl.release(b); pressed_buttons.remove(b)
def capture_frame():
    with mss.mss() as sct: return transform(cv2.cvtColor(np.array(sct.grab(sct.monitors[1])), cv2.COLOR_BGRA2RGB))

# --- MAIN CONTROL LOGIC ---
def apply_output(output):
    num_keys = len(COMMON_KEYS)
    key_logits, delta_out, click_logits = output[:num_keys], output[num_keys:num_keys+2], output[num_keys+2:num_keys+4]
    key_probs, click_probs = torch.sigmoid(key_logits).numpy(), torch.sigmoid(click_logits).numpy()
    
    # Handle keys
    for i, key_str in enumerate(COMMON_KEYS):
        if key_probs[i] > action_threshold and key_str not in pressed_keys:
            k_ctrl.press(KEY_MAPPING.get(key_str)); pressed_keys.add(key_str)
        elif key_probs[i] <= action_threshold and key_str in pressed_keys:
            k_ctrl.release(KEY_MAPPING.get(key_str)); pressed_keys.remove(key_str)
            
    # Handle clicks
    if click_probs[0] > action_threshold and Button.left not in pressed_buttons:
        m_ctrl.press(Button.left); pressed_buttons.add(Button.left)
    elif click_probs[0] <= action_threshold and Button.left in pressed_buttons:
        m_ctrl.release(Button.left); pressed_buttons.remove(Button.left)
    if click_probs[1] > action_threshold and Button.right not in pressed_buttons:
        m_ctrl.press(Button.right); pressed_buttons.add(Button.right)
    elif click_probs[1] <= action_threshold and Button.right in pressed_buttons:
        m_ctrl.release(Button.right); pressed_buttons.remove(Button.right)
        
    # Handle mouse movement using deltas
    dx, dy = delta_out[0].item() * MOUSE_SENSITIVITY, delta_out[1].item() * MOUSE_SENSITIVITY
    m_ctrl.move(int(dx), int(dy))

# --- MAIN LOOP ---
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press); listener.start()
    print("\n" + "="*50 + "\n🟢 Starting AI inference...\n   [F10] to ENABLE/DISABLE AI | [F12] to quit.\n" + "="*50 + "\n")
    
    for _ in range(SEQUENCE_LENGTH): frame_sequence.append(capture_frame())
    
    try:
        while running:
            if ai_enabled:
                frame_sequence.append(capture_frame())
                with torch.no_grad():
                    input_tensor = torch.stack(list(frame_sequence)).unsqueeze(0).to(device)
                    output = model(input_tensor)
                    apply_output(output[:, -1, :].squeeze())
            time.sleep(1.0 / INFERENCE_FPS)
    finally:
        release_all_inputs(); listener.stop(); print("\n✅ Inference stopped.")