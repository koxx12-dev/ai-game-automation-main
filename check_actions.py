import numpy as np
import os

# --- Configuration ---
DATA_DIR = "data_human"
ACTIONS_FILE = "actions.npy"
# -------------------

actions_path = os.path.join(DATA_DIR, ACTIONS_FILE)

if not os.path.exists(actions_path):
    print(f"❌ Error: The file '{actions_path}' does not exist!")
else:
    try:
        actions_data = np.load(actions_path)
        print(f"✅ Successfully loaded '{actions_path}'")
        print("-" * 30)
        print(f"Shape of the actions data: {actions_data.shape}")

        if len(actions_data.shape) > 1:
            num_actions = actions_data.shape[0]
            action_length = actions_data.shape[1]
            print(f"➡️ This means there are {num_actions} recorded action entries.")
            print(f"➡️ Each action has a length of {action_length}.")
        else:
             num_actions = 0
             print("➡️ The actions file appears to be empty or malformed.")

        print("-" * 30)

        if num_actions < 100:
            print("⚠️ WARNING: You have very few recorded actions compared to your 9500 frames.")
            print("   This is the most likely cause of the training script stopping silently.")

    except Exception as e:
        print(f"❌ Error loading or reading the numpy file: {e}")