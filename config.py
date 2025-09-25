import os
from pynput import keyboard

# === PATHS ===
DATA_DIR = "data_human"
DATA_DIRS = [DATA_DIR]  # You can add more data directories here
FRAME_DIR = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE = "actions.npy"

MODEL_FILE = "trained_model.pth"  # Final trained model
MODEL_SAVE_DIR = "game_model_checkpoints"  # Directory for epoch checkpoints
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH_TEMPLATE = os.path.join(MODEL_SAVE_DIR, "model_epoch_{}.pth")

# Options: 360x240 (fast), 480x320 (balanced), 640x480 (detailed)
IMG_WIDTH = 480
IMG_HEIGHT = 320
SEQUENCE_LENGTH = 30  # Number of frames the model sees at once

# === FPS-OPTIMIZED RECORDING & INFERENCE FPS ===
RECORDING_FPS = 20
INFERENCE_FPS = 20

# === INTELLIGENT DATA FILTERING (NEW) ===
# This feature saves only frames with meaningful actions, reducing dataset size.
IDLE_FRAME_BUFFER_SIZE = 30 # Frames to hold before an action (e.g., 60 frames = 2s at 30 FPS)
ACTION_POST_SAVE_FRAMES = 15 # Frames to save *after* the last action (e.g., 45 frames = 1.5s at 30 FPS)
MOUSE_MOVE_ACTION_THRESHOLD = 0.005 # Normalized sensitivity for mouse movement to be considered an action

# === FPS-OPTIMIZED TRAINING PARAMETERS ===
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 2e-4

# === TRANSFORMER MODEL PARAMETERS ===
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.1

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 3
VALIDATION_SPLIT = 0.15
VALIDATION_WINDOW = 3
THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

# === FPS-OPTIMIZED INFERENCE THRESHOLDS ===
KEY_THRESHOLD = 0.5
CLICK_THRESHOLD = 0.5
MOUSE_DEADZONE = 2 # pixel movement dead zone

# === TENSORBOARD ===
TENSORBOARD_LOG_DIR = "runs/behavior_cloning_improved"

# === USED KEY MAPPING ===
COMMON_KEYS = [
    "w", "a", "s", "d", "q", "e", "f", "r",
    "1", "2", "3", "4", "5", "6"#, "7", "8", "9", "0",
    "tab", "space"#, "`", "f1"#, "shift", "ctrl",
]

KEY_MAPPING = {
    # Alphanumeric
    **{char: keyboard.KeyCode.from_char(char) for char in "abcdefghijklmnopqrstuvwxyz1234567890"},
    # Function keys
    **{f'f{i}': getattr(keyboard.Key, f'f{i}') for i in range(1, 13)},
    # Modifier keys
    'shift': keyboard.Key.shift,
    'ctrl': keyboard.Key.ctrl,
    'alt': keyboard.Key.alt,
    # Special keys
    'space': keyboard.Key.space,
    'enter': keyboard.Key.enter,
    'backspace': keyboard.Key.backspace,
    'tab': keyboard.Key.tab,
    'escape': keyboard.Key.esc,
    'insert': keyboard.Key.insert,
    'delete': keyboard.Key.delete,
    'home': keyboard.Key.home,
    'end': keyboard.Key.end,
    'page_up': keyboard.Key.page_up,
    'page_down': keyboard.Key.page_down,
    # Arrow keys
    'up': keyboard.Key.up,
    'down': keyboard.Key.down,
    'left': keyboard.Key.left,
    'right': keyboard.Key.right,
    # Symbol keys
    '`': keyboard.KeyCode.from_char('`'),
    '-': keyboard.KeyCode.from_char('-'),
    '=': keyboard.KeyCode.from_char('='),
    '[': keyboard.KeyCode.from_char('['),
    ']': keyboard.KeyCode.from_char(']'),
    '\\': keyboard.KeyCode.from_char('\\'),
    ';': keyboard.KeyCode.from_char(';'),
    "'": keyboard.KeyCode.from_char("'"),
    ',': keyboard.KeyCode.from_char(','),
    '.': keyboard.KeyCode.from_char('.'),
    '/': keyboard.KeyCode.from_char('/'),
}