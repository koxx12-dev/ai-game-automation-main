import os
from pynput import keyboard

# === PATHS ===
DATA_DIR = "data_human"
DATA_DIRS = [DATA_DIR]
FRAME_DIR = os.path.join(DATA_DIR, "frames")
ACTIONS_FILE = "actions.npy"

MODEL_FILE = "trained_model_v2.pth" # Renamed to reflect new architecture
MODEL_SAVE_DIR = "game_model_checkpoints_v2"
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)
MODEL_SAVE_PATH_TEMPLATE = os.path.join(MODEL_SAVE_DIR, "model_epoch_{}.pth")

# === IMAGE & SEQUENCE SETTINGS ===
IMG_WIDTH = 360
IMG_HEIGHT = 240
SEQUENCE_LENGTH = 15

# === RECORDING & INFERENCE FPS ===
RECORDING_FPS = 30
INFERENCE_FPS = 30

# === TRAINING PARAMETERS ===
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 2e-4

# === TRANSFORMER MODEL PARAMETERS ===
D_MODEL = 256
N_HEAD = 8
N_LAYERS = 3
DROPOUT = 0.1

# === DATASET & VALIDATION ===
# How many times to repeat frames with actions vs. frames with no actions
OVERSAMPLE_ACTION_FRAMES_MULTIPLIER = 1 # Lowered because we now filter out most inaction frames
VALIDATION_SPLIT = 0.15 # Use the last 15% of data for validation
VALIDATION_WINDOW = 3
THRESHOLD_SWEEP = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]

# === INFERENCE & CONTROL ===
KEY_THRESHOLD = 0.5
CLICK_THRESHOLD = 0.5
# Determines how much the mouse moves based on model output. Tune this to your liking.
MOUSE_SENSITIVITY = 150 # Pixels moved per max model output

# === DATA RECORDING SETTINGS ===
# How much the mouse has to move (in normalized screen space) to trigger a save
MOUSE_DELTA_THRESHOLD = 0.005

# === EARLY STOPPING & TENSORBOARD ===
EARLY_STOPPING_PATIENCE = 10
EARLY_STOPPING_MIN_DELTA = 0.002
TENSORBOARD_LOG_DIR = "runs/behavior_cloning_v2"

# === KEY MAPPING ===
COMMON_KEYS = [
    "w","a","s","d",                        # Movement
    "1","2","3","4","5","6",                # Hotkeys
    "e","r","tab","space","shift","ctrl",   # Interact, reload, menu, actions
]

KEY_MAPPING = {
    **{char: keyboard.KeyCode.from_char(char) for char in "abcdefghijklmnopqrstuvwxyz1234567890"},
    'shift': keyboard.Key.shift, 'ctrl': keyboard.Key.ctrl, 'alt': keyboard.Key.alt,
    'space': keyboard.Key.space, 'enter': keyboard.Key.enter, 'backspace': keyboard.Key.backspace,
    'tab': keyboard.Key.tab, 'escape': keyboard.Key.esc
}