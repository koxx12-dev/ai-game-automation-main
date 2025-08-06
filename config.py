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

# === IMAGE & SEQUENCE SETTINGS ===
# Unified image dimensions for recording, training, and inference.
IMG_WIDTH = 360
IMG_HEIGHT = 240
SEQUENCE_LENGTH = 15  # Number of frames the model sees at once

# === RECORDING & INFERENCE FPS ===
RECORDING_FPS = 30
INFERENCE_FPS = 30

# === TRAINING PARAMETERS ===
BATCH_SIZE = 32
EPOCHS = 15
LEARNING_RATE = 1e-4

# === TRANSFORMER MODEL PARAMETERS ===
# These are used for the Transformer-based model architecture.
D_MODEL = 256  # The dimension of the transformer model (embedding size)
N_HEAD = 8     # Number of attention heads in the multi-head attention models
N_LAYERS = 3   # Number of sub-encoder-layers in the transformer encoder
DROPOUT = 0.1  # Dropout value

# === DATASET BALANCING & VALIDATION ===
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 15
VALIDATION_SPLIT = 0.15
VALIDATION_WINDOW = 3  # Timesteps to aggregate for validation metrics
THRESHOLD_SWEEP = [0.3, 0.4, 0.5, 0.6, 0.7] # For finding best validation threshold

# === INFERENCE ACTION THRESHOLDS ===
KEY_THRESHOLD = 0.5  # Stricter threshold after validation
CLICK_THRESHOLD = 0.5
# Mouse smoothing parameters
MOUSE_SMOOTHING_ALPHA = 0.3  # Prediction smoothing (higher=more responsive)
SMOOTH_FACTOR = 0.6  # Movement smoothing (higher=faster)
MOUSE_DEADZONE = 2  # Pixels

# === EARLY STOPPING & TENSORBOARD ===
EARLY_STOPPING_PATIENCE = 5
EARLY_STOPPING_MIN_DELTA = 0.005 # Minimum F1 score improvement
TENSORBOARD_LOG_DIR = "runs/behavior_cloning_transformer_experiment"

# === COMPREHENSIVE KEY MAPPING ===
# This list defines the order and size of the keyboard action space for the model.
COMMON_KEYS = [
    "w","a","s","d",                # Movement
    "space","shift","ctrl",         # Actions
    "1","2","3","4","5","6",        # Hotkeys
    "e","r","tab",                  # Other
]

# This dictionary maps the string representation to pynput key objects for inference.
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
