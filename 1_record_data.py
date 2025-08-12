import os
import re
import cv2
import mss
import time
import signal
import numpy as np
import threading
import queue
from collections import deque
from pynput import keyboard, mouse
from config import * # Import all settings from the config file

# ---------------------- Setup ----------------------
os.makedirs(FRAME_DIR, exist_ok=True)

with mss.mss() as sct:
    monitor = sct.monitors[1]
    SCREEN_WIDTH = monitor["width"]
    SCREEN_HEIGHT = monitor["height"]

print(f"Detected screen resolution: {SCREEN_WIDTH}x{SCREEN_HEIGHT}")
print(f"Recording at {IMG_WIDTH}x{IMG_HEIGHT} @ {RECORDING_FPS} FPS")
print(f"Keys being recorded: {len(COMMON_KEYS)}")
print(f"Intelligent Filtering: ENABLED (Action-Change Trigger)")
print(f"  - Idle Buffer Size: {IDLE_FRAME_BUFFER_SIZE} frames (~{IDLE_FRAME_BUFFER_SIZE/RECORDING_FPS:.1f}s)")
print(f"  - Post-Action Save: {ACTION_POST_SAVE_FRAMES} frames (~{ACTION_POST_SAVE_FRAMES/RECORDING_FPS:.1f}s)")
print(f"  - Mouse Action Threshold: {MOUSE_MOVE_ACTION_THRESHOLD}")


# ---------------------- Global State & Threading Primitives ----------------------
pressed_keys = set()
mouse_buttons = {"left": 0, "right": 0}
current_mouse_position = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
running = True
data_lock = threading.Lock()
stop_event = threading.Event()

frame_queue = queue.Queue(maxsize=RECORDING_FPS * 2) 
save_queue = queue.Queue()

# ---------------------- Input Handling ----------------------
def get_key_str(key):
    if hasattr(key, "char") and key.char: return key.char.lower()
    if hasattr(key, "name"): return key.name.replace("_l", "").replace("_r", "")
    return None

def on_key_press(key):
    global running
    if key in (keyboard.Key.f12, keyboard.Key.f2):
        print("🛑 Quit key pressed. Stopping recording...")
        running = False
        return
    if (key_str := get_key_str(key)) in COMMON_KEYS:
        with data_lock: pressed_keys.add(key_str)

def on_key_release(key):
    if (key_str := get_key_str(key)) in COMMON_KEYS:
        with data_lock: pressed_keys.discard(key_str)

def on_click(x, y, button, pressed):
    with data_lock:
        if button == mouse.Button.left: mouse_buttons["left"] = int(pressed)
        elif button == mouse.Button.right: mouse_buttons["right"] = int(pressed)

def on_move(x, y):
    global current_mouse_position
    with data_lock: current_mouse_position = (x, y)

# ---------------------- Asynchronous Worker Functions ----------------------
def screen_capture_worker():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        while not stop_event.is_set():
            try:
                img = np.array(sct.grab(monitor))
                img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                frame_queue.put_nowait((img, time.time()))
            except queue.Full:
                continue 
            except Exception as e:
                print(f"⚠️ Error in capture worker: {e}")
                time.sleep(0.5)

def file_save_worker():
    while not stop_event.is_set() or not save_queue.empty():
        try:
            frame, frame_path, action = save_queue.get(timeout=1)
            cv2.imwrite(frame_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            save_queue.task_done()
        except queue.Empty:
            continue
        except Exception as e:
            print(f"⚠️ Error in save worker: {e}")

# ---------------------- Signal Handling & Main Logic ----------------------
def signal_handler(signum, frame):
    global running
    print(f"\n🛑 Received signal {signum}. Stopping recording gracefully...")
    running = False

def is_action_happening(key_vector, click_vector, mouse_delta):
    """Check if the current state constitutes a recordable action."""
    key_press = any(key_vector)
    mouse_click = any(click_vector)
    mouse_move = abs(mouse_delta[0]) > MOUSE_MOVE_ACTION_THRESHOLD or \
                 abs(mouse_delta[1]) > MOUSE_MOVE_ACTION_THRESHOLD
    return key_press or mouse_click or mouse_move

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
                        print(f"⚠️ Truncating action file to {start_index} entries to match frames.")
                        actions = loaded_actions[:start_index]
                    else:
                        actions = loaded_actions
                    print(f"   Loaded {len(actions)} actions.")
                except Exception as e:
                    print(f"❌ Error loading actions file: {e}. Starting fresh.")
                    actions, start_index = [], 0

    print("\n" + "=" * 50)
    print("🟢 Starting INTELLIGENT data recording in 5 seconds...")
    print("   Play the game normally. Only action sequences will be saved.")
    print("   Press [F2] or [F12] to quit gracefully.")
    print("=" * 50 + "\n")
    time.sleep(5)

    key_listener = keyboard.Listener(on_press=on_key_press, on_release=on_key_release)
    mouse_listener = mouse.Listener(on_click=on_click, on_move=on_move)
    key_listener.start()
    mouse_listener.start()

    capture_thread = threading.Thread(target=screen_capture_worker, daemon=True)
    save_thread = threading.Thread(target=file_save_worker, daemon=True)
    capture_thread.start()
    save_thread.start()

    frame_interval = 1.0 / RECORDING_FPS
    frame_index = start_index
    last_mouse_position = current_mouse_position
    
    # Data structures for intelligent filtering
    temp_buffer = deque(maxlen=IDLE_FRAME_BUFFER_SIZE)
    frames_to_save_after_action = 0
    is_saving_action = False
    last_action_state = None # NEW: Track the previous action state

    try:
        while running:
            loop_start_time = time.time()
            
            latest_frame, _ = frame_queue.get() # Block until a new frame is ready
            
            with data_lock:
                pos_now = current_mouse_position
                key_vector = [int(k in pressed_keys) for k in COMMON_KEYS]
                click_vector = [mouse_buttons["left"], mouse_buttons["right"]]
            
            delta_x = (pos_now[0] - last_mouse_position[0]) / SCREEN_WIDTH
            delta_y = (pos_now[1] - last_mouse_position[1]) / SCREEN_HEIGHT
            mouse_delta = (delta_x, delta_y)
            last_mouse_position = pos_now

            action = key_vector + list(mouse_delta) + click_vector
            temp_buffer.append({'frame': latest_frame, 'action': action})

            # --- NEW: Define action state based on discrete inputs (keys and clicks) ---
            current_action_state = (tuple(key_vector), tuple(click_vector))
            
            # --- NEW: Define trigger based on change in state or significant movement ---
            significant_move = abs(delta_x) > MOUSE_MOVE_ACTION_THRESHOLD or abs(delta_y) > MOUSE_MOVE_ACTION_THRESHOLD
            action_has_changed = (current_action_state != last_action_state)
            
            is_currently_active = is_action_happening(key_vector, click_vector, mouse_delta)

            # Trigger recording if the action state changes OR there's significant mouse movement
            if action_has_changed or significant_move:
                if not is_saving_action:
                    # Action just started, flush the buffer
                    print(f"\n--- ACTION CHANGE DETECTED! Saving preceding {len(temp_buffer)} frames. ---")
                    for item in list(temp_buffer):
                        frame_path = os.path.join(FRAME_DIR, f"frame_{frame_index:06d}.jpg")
                        save_queue.put((item['frame'], frame_path, item['action']))
                        actions.append(item['action'])
                        frame_index += 1
                    temp_buffer.clear()
                    is_saving_action = True
                
                # Reset the countdown to save frames after this action stops
                frames_to_save_after_action = ACTION_POST_SAVE_FRAMES

            if is_saving_action:
                # We are in an active recording sequence, so save the current frame
                frame_path = os.path.join(FRAME_DIR, f"frame_{frame_index:06d}.jpg")
                save_queue.put((latest_frame, frame_path, action))
                actions.append(action)

                print(f"Frame {frame_index} [RECORDING]: keys={list(p for p,k in zip(COMMON_KEYS, key_vector) if k)}, "
                      f"delta=({delta_x:.3f}, {delta_y:.3f}), click_L/R={action[-2]}/{action[-1]}, SaveQueue: {save_queue.qsize()}", end='\r')
                
                frame_index += 1
                
                # If no action is currently happening, start the countdown to stop recording
                if not is_currently_active:
                    frames_to_save_after_action -= 1
                    if frames_to_save_after_action <= 0:
                        is_saving_action = False
                        print("\n--- ACTION ENDED. Pausing recording. ---")
            else:
                print(f"Idle... Buffer: {len(temp_buffer)}/{IDLE_FRAME_BUFFER_SIZE} | Watching for action changes... ", end='\r')

            # Update the last action state for the next frame
            last_action_state = current_action_state

            elapsed_time = time.time() - loop_start_time
            sleep_time = frame_interval - elapsed_time
            if sleep_time > 0: time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n🛑 Recording interrupted by user.")
    finally:
        if running: running = False
        print("\n🛑 Shutting down... please wait.")
        stop_event.set()
        
        key_listener.stop()
        mouse_listener.stop()

        # Final flush of the buffer in case recording is stopped mid-action
        if is_saving_action and temp_buffer:
            print("   Flushing final items from temporary buffer...")
            for item in list(temp_buffer):
                frame_path = os.path.join(FRAME_DIR, f"frame_{frame_index:06d}.jpg")
                save_queue.put((item['frame'], frame_path, item['action']))
                actions.append(item['action'])
                frame_index += 1
        
        print("   Waiting for capture thread...")
        capture_thread.join(timeout=2)
        print("   Waiting for file save thread to flush queue...")
        save_queue.join()
        save_thread.join(timeout=10)
        
        if actions:
            try:
                np.save(actions_path, np.array(actions, dtype=np.float32))
                print(f"\n✅ Saved {len(actions)} actions to {actions_path}.")
            except Exception as e:
                print(f"\n❌ Error saving actions: {e}")
        else:
            print("\nNo actions were recorded!")
        
        print("✅ Shutdown complete.")