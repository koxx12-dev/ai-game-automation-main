import os
import re
import cv2
import mss
import time
import signal
import numpy as np
import threading
import queue
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

# ---------------------- Global State & Threading Primitives ----------------------
pressed_keys = set()
mouse_buttons = {"left": 0, "right": 0}
current_mouse_position = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
running = True
data_lock = threading.Lock()
stop_event = threading.Event()

# Queues for inter-thread communication
# frame_queue will store (frame_data, timestamp)
frame_queue = queue.Queue(maxsize=RECORDING_FPS * 2) 
# save_queue will store (frame_data, frame_path)
save_queue = queue.Queue()

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
    if key in (keyboard.Key.f12, keyboard.Key.f2):
        print("🛑 Quit key pressed. Stopping recording...")
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
    """Handle mouse movement events by updating the current position."""
    global current_mouse_position
    with data_lock:
        current_mouse_position = (x, y)

# ---------------------- Asynchronous Worker Functions ----------------------
def screen_capture_worker():
    """Worker thread to continuously capture the screen."""
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while not stop_event.is_set():
            try:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                
                # Try to put the frame in the queue, but don't block if full
                frame_queue.put_nowait((img, time.time()))
            except queue.Full:
                # If queue is full, it means the main loop is lagging.
                # We can skip this frame to prioritize newer ones.
                continue 
            except Exception as e:
                print(f"⚠️ Error in capture worker: {e}")
                time.sleep(0.5)

def file_save_worker():
    """Worker thread to save frames to disk."""
    while not stop_event.is_set() or not save_queue.empty():
        try:
            # Wait for up to 1 second for an item to appear
            frame, frame_path = save_queue.get(timeout=1)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            save_queue.task_done()
        except queue.Empty:
            # Queue is empty, continue loop until stop_event is set and queue is confirmed empty
            continue
        except Exception as e:
            print(f"⚠️ Error in save worker: {e}")

def get_current_action(mouse_delta):
    """Construct the action vector for the current state."""
    with data_lock:
        key_vector = [int(k in pressed_keys) for k in COMMON_KEYS]
        action = (
            key_vector +
            list(mouse_delta) +
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
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    actions_path = os.path.join(DATA_DIR, ACTIONS_FILE)
    actions = []
    start_index = 0

    if os.path.exists(FRAME_DIR) and os.listdir(FRAME_DIR):
        existing_frames = [f for f in os.listdir(FRAME_DIR) if f.endswith(".jpg")]
        if existing_frames:
            last_frame_num = max([int(re.search(r"frame_(\d+).jpg", f).group(1)) for f in existing_frames if re.search(r"frame_(\d+).jpg", f)])
            start_index = last_frame_num + 1
            print(f"✅ Resuming from frame {start_index}")
            if os.path.exists(actions_path):
                try:
                    loaded_actions = np.load(actions_path).tolist()
                    if len(loaded_actions) > start_index:
                        print(f"⚠️ Truncating action file from {len(loaded_actions)} to {start_index} entries.")
                        actions = loaded_actions[:start_index]
                    else:
                        actions = loaded_actions
                    print(f"   Loaded {len(actions)} actions.")
                except Exception as e:
                    print(f"❌ Error loading actions file: {e}. Starting fresh.")
                    actions, start_index = [], 0

    print("\n" + "=" * 50)
    print("🟢 Starting ASYNCHRONOUS data recording in 5 seconds...")
    print("   Play the game normally.")
    print("   Press [F2] or [F12] to quit gracefully.")
    print("=" * 50 + "\n")
    time.sleep(5)

    # Start input listeners
    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    key_listener.start()
    mouse_listener.start()

    # Start worker threads
    capture_thread = threading.Thread(target=screen_capture_worker, daemon=True)
    save_thread = threading.Thread(target=file_save_worker, daemon=True)
    capture_thread.start()
    save_thread.start()

    frame_interval = 1.0 / RECORDING_FPS
    frame_index = start_index
    last_mouse_position = current_mouse_position

    try:
        while running:
            loop_start_time = time.time()
            
            # --- Get Latest Frame ---
            # Drain the queue to get the most recent frame, ensuring data is not stale
            latest_frame = None
            while not frame_queue.empty():
                try:
                    latest_frame, _ = frame_queue.get_nowait()
                except queue.Empty:
                    break
            
            if latest_frame is None:
                time.sleep(0.001) # Wait briefly if no frames are available
                continue
            
            # --- Calculate Mouse Delta ---
            with data_lock:
                pos_now = current_mouse_position
            
            delta_x = (pos_now[0] - last_mouse_position[0]) / SCREEN_WIDTH
            delta_y = (pos_now[1] - last_mouse_position[1]) / SCREEN_HEIGHT
            mouse_delta = (delta_x, delta_y)
            last_mouse_position = pos_now

            # --- Get Action & Save ---
            action, current_keys = get_current_action(mouse_delta)
            
            expected_length = len(COMMON_KEYS) + 2 + 2
            if len(action) != expected_length:
                print(f"⚠️ Invalid action length: {len(action)}. Skipping frame {frame_index}.")
                continue

            actions.append(action)
            frame_path = os.path.join(FRAME_DIR, f"frame_{frame_index:06d}.jpg")
            save_queue.put((latest_frame, frame_path))

            # Log if there is any action
            mouse_moved = abs(delta_x) > 1e-6 or abs(delta_y) > 1e-6
            if current_keys or any(action[-2:]) or mouse_moved:
                print(f"Frame {frame_index}: keys={list(current_keys)}, delta=({delta_x:.3f}, {delta_y:.3f}), "
                      f"click_L={action[-2]}, click_R={action[-1]}, SaveQueue: {save_queue.qsize()}")

            frame_index += 1
            
            # Maintain the target FPS
            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_interval - elapsed_time
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user.")
    finally:
        if running:
            running = False # Ensure running flag is set to false
        
        print("\n🛑 Shutting down... please wait.")
        
        # Signal workers to stop
        stop_event.set()
        
        # Stop input listeners
        key_listener.stop()
        mouse_listener.stop()
        
        # Wait for worker threads to finish
        print("   Waiting for capture thread to terminate...")
        capture_thread.join(timeout=2)
        print("   Waiting for file save thread to flush queue...")
        save_thread.join(timeout=10) # Give it time to save remaining frames
        
        if save_thread.is_alive():
            print("   ⚠️ Save thread timed out. Some frames may not be saved.")

        # Save actions file
        if actions:
            try:
                # Ensure actions and frames are aligned
                final_action_count = frame_index - start_index
                valid_actions = actions[:final_action_count]
                
                np.save(actions_path, np.array(valid_actions, dtype=np.float32))
                print(f"\n✅ Saved {len(valid_actions)} actions to {actions_path}.")
            except Exception as e:
                print(f"\n❌ Error saving actions: {e}")
        else:
            print("\nNo actions were recorded!")
        
        print("✅ Shutdown complete.")