import torch
import torch.nn as nn
import numpy as np
import cv2
import mss
import time
from collections import deque
from pynput.keyboard import Controller as KeyboardController
from pynput.mouse import Controller as MouseController, Button
from pynput import keyboard
from torchvision import transforms
from config import * # Import all settings from the config file

# --- SCREEN DIMENSIONS ---
with mss.mss() as sct:
    monitor = sct.monitors[1]
SCREEN_WIDTH, SCREEN_HEIGHT = monitor["width"], monitor["height"]

# === MODEL DEFINITION ===
# This MUST be identical to the one in the training script.
class BehaviorCloningCNNRNN(nn.Module):
    def __init__(self, output_dim):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=2, padding=2), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten()
        )
        
        # FIX: Ensure dummy input uses the correct, consistent image dimensions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 3, IMG_HEIGHT, IMG_WIDTH)
            cnn_out_size = self.cnn(dummy_input).shape[1]
        
        self.lstm = nn.LSTM(
            input_size=cnn_out_size,
            hidden_size=256,
            num_layers=2,
            batch_first=True,
            dropout=0.2
        )
        
        self.key_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, len(COMMON_KEYS))
        )
        self.mouse_pos_head = nn.Sequential(
            nn.Linear(256, 64), nn.ReLU(),
            nn.Linear(64, 2), nn.Sigmoid()
        )
        self.mouse_click_head = nn.Sequential(
            nn.Linear(256, 32), nn.ReLU(),
            nn.Linear(32, 2)
        )

    def forward(self, x):
        b, s, c, h, w = x.shape
        x_reshaped = x.view(b * s, c, h, w)
        feat = self.cnn(x_reshaped)
        feat_reshaped = feat.view(b, s, -1)
        lstm_out, _ = self.lstm(feat_reshaped)
        lstm_out_reshaped = lstm_out.reshape(b * s, -1)
        key_out = self.key_head(lstm_out_reshaped)
        pos_out = self.mouse_pos_head(lstm_out_reshaped)
        click_out = self.mouse_click_head(lstm_out_reshaped)
        concat = torch.cat([key_out, pos_out, click_out], dim=1)
        return concat.view(b, s, -1)

# --- SETUP ---
device = torch.device("cpu") #
output_dim = len(COMMON_KEYS) + 4 #
model = BehaviorCloningCNNRNN(output_dim) #
action_threshold = 0.5 # Default fallback value

try:
    model.load_state_dict(torch.load(MODEL_FILE, map_location=device)) #
    print(f"✅ Model '{MODEL_FILE}' loaded successfully.") #

    # Load the dynamically saved best threshold
    threshold_path = os.path.join(os.path.dirname(MODEL_FILE), "best_threshold.txt") if os.path.dirname(MODEL_FILE) else "best_threshold.txt"
    try:
        with open(threshold_path, 'r') as f:
            action_threshold = float(f.read().strip())
        print(f"   Dynamically loaded action threshold: {action_threshold:.2f}")
    except FileNotFoundError:
        print(f"   Threshold file not found. Using default from config: KEY={KEY_THRESHOLD}, CLICK={CLICK_THRESHOLD}")
        # If file not found, use separate thresholds from config
        action_threshold = None # Will signal to use config values
    except Exception as e:
        print(f"   Error loading threshold file: {e}. Using defaults.")
        action_threshold = None

except Exception as e: #
    print(f"❌ Error loading model: {e}") #
    print("   Ensure you have a trained model file at the correct path.") #
    exit(1) #

model.eval()
keyboard_controller = KeyboardController()
mouse_controller = MouseController()
running = True
ai_enabled = False # Start with AI disabled

# --- STATE MANAGEMENT ---
frame_sequence = deque(maxlen=SEQUENCE_LENGTH)
current_pressed_keys = set()
current_mouse_buttons = set()
# Use a target position for smoothing, not direct setting
target_mouse_pos = (SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2)

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- CONTROL FUNCTIONS ---
def on_press(key):
    """Toggle AI on/off with F10, quit with F12."""
    global running, ai_enabled
    if key == keyboard.Key.f12:
        running = False
        print("🛑 Quit key (F12) pressed.")
    elif key == keyboard.Key.f10:
        ai_enabled = not ai_enabled
        if not ai_enabled:
            release_all_inputs()
        status = "ENABLED" if ai_enabled else "DISABLED"
        print(f"🤖 AI control is now {status}")

def apply_output(output):
    """Interprets model output and converts it to keyboard/mouse actions."""
    global target_mouse_pos
    
    # Use sigmoid to convert logits to probabilities
    probs = torch.sigmoid(output).detach().cpu().numpy() #
    
    key_probs = probs[:len(COMMON_KEYS)] #
    mouse_x, mouse_y = probs[len(COMMON_KEYS)], probs[len(COMMON_KEYS) + 1] #
    left_click_prob, right_click_prob = probs[len(COMMON_KEYS) + 2], probs[len(COMMON_KEYS) + 3] #
    
    # Set target mouse position
    target_mouse_pos = (mouse_x * SCREEN_WIDTH, mouse_y * SCREEN_HEIGHT) #
    
    # MODIFICATION: Determine which threshold to use
    key_thresh_to_use = action_threshold if action_threshold is not None else KEY_THRESHOLD
    click_thresh_to_use = action_threshold if action_threshold is not None else CLICK_THRESHOLD
    
    # Press/release keys based on threshold
    for i, key_str in enumerate(COMMON_KEYS): #
        pynput_key = KEY_MAPPING.get(key_str) #
        if not pynput_key: continue #
        
        # MODIFICATION: Use the dynamically loaded threshold
        is_pressed = key_probs[i] > key_thresh_to_use #
        if is_pressed and key_str not in current_pressed_keys: #
            keyboard_controller.press(pynput_key) #
            current_pressed_keys.add(key_str) #
        elif not is_pressed and key_str in current_pressed_keys: #
            keyboard_controller.release(pynput_key) #
            current_pressed_keys.remove(key_str) #
            
    # Handle mouse clicks
    # MODIFICATION: Use the dynamically loaded threshold
    left_click = left_click_prob > click_thresh_to_use #
    right_click = right_click_prob > click_thresh_to_use #
    
    if left_click and Button.left not in current_mouse_buttons: #
        mouse_controller.press(Button.left) #
        current_mouse_buttons.add(Button.left) #
    elif not left_click and Button.left in current_mouse_buttons: #
        mouse_controller.release(Button.left) #
        current_mouse_buttons.remove(Button.left) #
        
    if right_click and Button.right not in current_mouse_buttons: #
        mouse_controller.press(Button.right) #
        current_mouse_buttons.add(Button.right) #
    elif not right_click and Button.right in current_mouse_buttons: #
        mouse_controller.release(Button.right) #
        current_mouse_buttons.remove(Button.right) #

def smooth_mouse_movement():
    """Interpolates mouse position for smooth movement."""
    current_pos = mouse_controller.position
    diff_x = target_mouse_pos[0] - current_pos[0]
    diff_y = target_mouse_pos[1] - current_pos[1]
    
    if abs(diff_x) > MOUSE_DEADZONE or abs(diff_y) > MOUSE_DEADZONE:
        new_x = int(current_pos[0] + diff_x * SMOOTH_FACTOR)
        new_y = int(current_pos[1] + diff_y * SMOOTH_FACTOR)
        mouse_controller.position = (
            max(0, min(SCREEN_WIDTH - 1, new_x)),
            max(0, min(SCREEN_HEIGHT - 1, new_y))
        )

def capture_frame():
    """Captures and transforms a single screen frame."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = cv2.cvtColor(np.array(sct.grab(monitor)), cv2.COLOR_BGRA2RGB)
        return transform(img)

def release_all_inputs():
    """Releases all currently pressed keys and mouse buttons."""
    for key_str in list(current_pressed_keys):
        if (pynput_key := KEY_MAPPING.get(key_str)):
            keyboard_controller.release(pynput_key)
    current_pressed_keys.clear()
    
    for button in list(current_mouse_buttons):
        mouse_controller.release(button)
    current_mouse_buttons.clear()
    print("All inputs released.")

# --- MAIN LOOP ---
if __name__ == "__main__":
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    print("\n" + "="*50)
    print("🟢 Starting AI inference...")
    print("   Press [F10] to ENABLE/DISABLE AI control.")
    print("   Press [F12] to quit.")
    print("="*50 + "\n")

    # Pre-fill the frame buffer
    for _ in range(SEQUENCE_LENGTH):
        frame_sequence.append(capture_frame())
        time.sleep(0.05)
    
    try:
        frame_interval = 1.0 / INFERENCE_FPS
        last_inference_time = time.time()
        
        while running:
            if ai_enabled:
                current_time = time.time()
                if current_time - last_inference_time >= frame_interval:
                    last_inference_time = current_time
                    
                    frame_sequence.append(capture_frame())
                    
                    with torch.no_grad():
                        input_tensor = torch.stack(list(frame_sequence)).unsqueeze(0).to(device)
                        output = model(input_tensor)
                        # Use the prediction from the very last time step
                        apply_output(output[:, -1, :].squeeze())
                
                smooth_mouse_movement()
            
            # Sleep to prevent high CPU usage, even when AI is disabled
            time.sleep(0.001)
            
    finally:
        release_all_inputs()
        listener.stop()
        print("\n✅ Inference stopped cleanly.")

