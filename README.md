# AI Game Automation (Behavior Cloning)

This project records human gameplay, trains a CNN+Transformer behavior cloning model, and runs real-time inference to automate in-game keyboard and mouse actions.

## Repository Layout

- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/1_record_data.py` — records gameplay frames and action vectors from the active window.
- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/2_train_model.py` — trains the behavior cloning model and saves checkpoints/best model.
- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/3_run_inference.py` — loads the trained model and controls input in real time.
- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/config.py` — central configuration for paths, model parameters, controls, thresholds, and training settings.
- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/check_actions.py` — quick stats on recorded action distribution.
- `/home/runner/work/ai-game-automation-main/ai-game-automation-main/find_imbalance.py` — checks whether recorded action data is sparse/imbalanced.

## Requirements

- Python 3.10+ recommended
- A desktop environment (scripts use active-window capture and keyboard/mouse hooks)
- GPU optional (training uses CUDA if available; inference currently runs on CPU)

Install dependencies:

```bash
python -m venv venv
source venv/bin/activate   # Linux/macOS
# venv\\Scripts\\activate    # Windows
pip install -r requirements.txt
```

## Configure

All main settings are in `/home/runner/work/ai-game-automation-main/ai-game-automation-main/config.py`, including:

- Data/model paths (`DATA_DIR`, `FRAME_DIR`, `MODEL_FILE`)
- Frame size and sequence length
- Recording and inference FPS
- Keys to learn (`COMMON_KEYS`)
- Training hyperparameters (batch size, epochs, learning rate)
- Model architecture settings (Transformer dimensions/layers)

## Workflow

### 1) Record gameplay data

```bash
python 1_record_data.py
```

- Focus your target game window during the startup delay.
- Press `F2` or `F12` to stop recording.
- Outputs:
  - Frames in `data_human/frames/`
  - Actions in `data_human/actions.npy`

### 2) Train the model

```bash
python 2_train_model.py
```

Optional (disable TensorBoard auto-start):

```bash
python 2_train_model.py --no-tensorboard
```

Training outputs:

- Epoch checkpoints: `game_model_checkpoints/model_epoch_*.pth`
- Best model: `trained_model.pth`
- Best threshold: `best_threshold.txt`
- TensorBoard logs: `runs/behavior_cloning_improved`

### 3) Run inference

```bash
python 3_run_inference.py
```

- Focus the target game window during the startup delay.
- Press `F2` to toggle AI control on/off.
- Press `F12` to quit.

## Data Utilities

Check dataset/action balance:

```bash
python find_imbalance.py
python check_actions.py
```

Use these before training if learning is unstable or action classes are sparse.

## Notes

- Scripts depend on active-window targeting (`pygetwindow`) and global input control (`pynput`), so run with proper desktop permissions.
- Keep `COMMON_KEYS` in `config.py` aligned with the actions you want the model to learn.
- If you change input dimensions or key mappings, re-record data and retrain.
