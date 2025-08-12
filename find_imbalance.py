import numpy as np
from config import COMMON_KEYS

# Path to your actions file
actions_file = "data_human/actions.npy"
actions = np.load(actions_file)

# The action vector is [keys, delta_x, delta_y, left_click, right_click]
num_keys = len(COMMON_KEYS)
left_click_col = actions[:, num_keys + 2]
right_click_col = actions[:, num_keys + 3]

total_frames = len(actions)
right_click_frames = np.sum(right_click_col)

print(f"Total frames: {total_frames}")
print(f"Frames with right-click active: {int(right_click_frames)}")
print(f"Percentage of time right-click is held: {(right_click_frames / total_frames) * 100:.2f}%")