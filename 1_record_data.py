import os
import re
import cv2
import mss
import sys
import time
import signal
import numpy as np
import threading
from pynput import keyboard, mouse
from config import * # Import all settings from the config file

# ---------------------- Setup ----------------------
# Create directories if they don't exist
os.makedirs(FRAME_DIR, exist_ok=True)

# Get screen dimensions for mouse coordinate normalization
with mss.mss() as sct:
    monitor = sct.monitors[1]
    SCREEN_WIDTH = monitor["width"]
    SCREEN_HEIGHT = monitor["height"]

print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"Recording at {IMG_WIDTH}x{IMG_HEIGHT} @ {RECORDING_FPS} FPS")
print(f"Keys being recorded: {len(COMMON_KEYS)}")

# ---------------------- Global State ----------------------
pressed_keys = set()
mouse_buttons = {"left": 0, "right": 0}
mouse_position = (0.5, 0.5)           # Normalized position
smoothed_mouse_position = (0.5, 0.5)  # Smoothed for recording
running = True
data_lock = threading.Lock()  # Thread-safe access

# ---------------------- Input Handling ----------------------
def get_key_str(key):
    """Convert pynput key object to standardized string."""
    if hasattr(key, "char") and key.char:
        return key.char.lower()
    elif hasattr(key, "name"):
        return key.name.replace("_l", "").replace("_r", "")
    return None

def on_key_press(key):
    """Handle key press events."""
    global running
    if key == keyboard.Key.f12:
        print("🛑 Quit key (F12) pressed. Stopping recording...")
        running = False
        return
    elif key == keyboard.Key.esc:
        print("🛑 Escape key pressed. Stopping recording...")
        running = False
        return

    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.add(key_str)

def on_key_release(key):
    """Handle key release events."""
    key_str = get_key_str(key)
    if key_str in COMMON_KEYS:
        with data_lock:
            pressed_keys.discard(key_str)

def on_click(x, y, button, pressed):
    """Handle mouse click events."""
    with data_lock:
        if button == mouse.Button.left:
            mouse_buttons["left"] = int(pressed)
        elif button == mouse.Button.right:
            mouse_buttons["right"] = int(pressed)

def on_move(x, y):
    """Handle mouse movement events and apply smoothing."""
    global mouse_position, smoothed_mouse_position
    with data_lock:
        raw_position = (x / SCREEN_WIDTH, y / SCREEN_HEIGHT)
        mouse_position = raw_position

        alpha = 0.3
        smoothed_mouse_position = (
            alpha * raw_position[0] + (1 - alpha) * smoothed_mouse_position[0],
            alpha * raw_position[1] + (1 - alpha) * smoothed_mouse_position[1]
        )

# ---------------------- Recording ----------------------
def capture_frame():
    """Capture and resize a single screen frame."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
        img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
        return img

def get_current_action():
    """Construct the action vector for the current state."""
    with data_lock:
        key_vector = [int(k in pressed_keys) for k in COMMON_KEYS]
        action = (
            key_vector +
            list(smoothed_mouse_position) +
            [mouse_buttons["left"], mouse_buttons["right"]]
        )
        return action.copy(), pressed_keys.copy()

# ---------------------- Signal Handling ----------------------
def signal_handler(signum, frame):
    """Handle system signals for graceful shutdown."""
    global running
    print(f"\n🛑 Received signal {signum}. Stopping recording gracefully...")
    running = False

# ---------------------- Main ----------------------
if __name__ == "__main__":
    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    actions_path = os.path.join(DATA_DIR, ACTIONS_FILE)
    actions = []
    start_index = 0

    # Resume from existing data if found
    if os.path.exists(FRAME_DIR) and os.listdir(FRAME_DIR):
        existing_frames = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
        if existing_frames:
            last_frame_num = -1
            for frame_file in existing_frames:
                match = re.search(r"frame_(\d+).jpg", frame_file)
                if match:
                    last_frame_num = max(last_frame_num, int(match.group(1)))

            if last_frame_num != -1:
                start_index = last_frame_num + 1
                print(f"✅ Resuming from frame {start_index}")
                if os.path.exists(actions_path):
                    try:
                        actions = np.load(actions_path).tolist()
                        if len(actions) > start_index:
                            print(f"⚠️ Truncating action file from {len(actions)} to {start_index} entries.")
                            actions = actions[:start_index]
                        print(f"   Loaded {len(actions)} actions.")
                    except Exception as e:
                        print(f"❌ Error loading actions: {e}. Starting fresh.")
                        actions, start_index = [], 0

    # Startup message
    print("\n" + "=" * 50)
    print("🟢 Starting HUMAN data recording in 5 seconds...")
    print("   Play the game normally.")
    print("   Press [F12] or [ESC] to quit gracefully.")
    print("   Or use [Ctrl+C] in terminal for emergency stop.")
    print("=" * 50 + "\n")
    time.sleep(5)

    # Start input listeners
    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    key_listener.start()
    mouse_listener.start()

    frame_interval = 1.0 / RECORDING_FPS
    i = start_index

    try:
        last_capture_time = time.time()
        while running:
            current_time = time.time()
            if current_time - last_capture_time >= frame_interval:
                last_capture_time = current_time

                try:
                    frame = capture_frame()
                    action, current_keys = get_current_action()

                    expected_length = len(COMMON_KEYS) + 2 + 2
                    if len(action) != expected_length:
                        print(f"⚠️ Invalid action length: {len(action)} (expected {expected_length}). Skipping.")
                        continue

                    frame_path = os.path.join(FRAME_DIR, f"frame_{i:06d}.jpg")
                    cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                    actions.append(action)

                    if current_keys or any(action[-2:]): # Check for active clicks
                        print(f"Frame {i}: keys={list(current_keys)}, mouse=({action[-4]:.2f}, {action[-3]:.2f}), "
                              f"click_L={action[-2]}, click_R={action[-1]}")

                    i += 1

                except KeyboardInterrupt:
                    print("\n🛑 Recording interrupted by user.")
                    break
                except Exception as e:
                    print(f"⚠️ Error capturing frame {i}: {e}. Continuing...")
                    continue

            time.sleep(0.001)

    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user.")
    finally:
        print("🛑 Stopping input listeners...")
        key_listener.stop()
        mouse_listener.stop()

        with data_lock:
            mouse_buttons["left"] = mouse_buttons["right"] = 0

        if actions:
            try:
                expected_length = len(COMMON_KEYS) + 2 + 2
                valid_actions = [a for a in actions if len(a) == expected_length]
                if valid_actions:
                    np.save(actions_path, np.array(valid_actions, dtype=np.float32))
                    print(f"\n✅ Saved {len(valid_actions)} actions to {DATA_DIR}.")
                    if len(valid_actions) != len(actions):
                        print(f"⚠️ Skipped {len(actions) - len(valid_actions)} invalid actions.")
                else:
                    print("\n❌ No valid actions to save!")
            except Exception as e:
                print(f"\n❌ Error saving actions: {e}")
        else:
            print("\nNo actions were recorded!")