import os
from pathlib import Path


# 1. DIRECTORY STRUCTURE PATHS

# Resolve base directory relative to this file's location
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_MAESTRO_DIR = DATA_DIR / "raw_maestro"
PROCESSED_DIR = DATA_DIR / "processed"

# Outputs
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
CHECKPOINTS_DIR = OUTPUTS_DIR / "checkpoints"
GENERATED_MIDIS_DIR = OUTPUTS_DIR / "generated_midis"

# Ensure all necessary directories exist automatically

def ensure_directory(path: Path) -> None:
    if path.exists():
        if path.is_dir():
            return
        path.unlink()
    path.mkdir(parents=True, exist_ok=True)

for d in [RAW_MAESTRO_DIR, PROCESSED_DIR, PLOTS_DIR, CHECKPOINTS_DIR, GENERATED_MIDIS_DIR]:
    ensure_directory(d)


# 2. PIANO-ROLL & MIDI PARAMETERS (Tasks 1 & 2)

FS = 16                    # 16 time steps per second/bar
N_KEYS = 88                # Total number of keys on a standard piano
START_PITCH = 21           # MIDI note number for A0 (lowest piano key)
END_PITCH = 108            # MIDI note number for C8 (highest piano key)
WINDOW_LENGTH = 128        # Length of each segmented window
SPARSITY_THRESHOLD = 0.02  # Discard windows with < 2% active cells to prevent silence bias


# 3. TOKENIZATION PARAMETERS (Task 3)

# Configuration for MidiTok REMI tokenization
TOKENIZER_PARAMS = {
    "pitch_range": (START_PITCH, END_PITCH),
    "beat_res": {(0, 4): 8, (4, 12): 4},
    "num_velocities": 32,
    "special_tokens": ["PAD", "BOS", "EOS", "MASK"],
    "use_chords": False,
    "use_rests": False,
    "use_tempos": True,
    "use_time_signatures": False,
}


# 4. MULTI-GENRE METADATA (Task 2 & 4)

# Proxy genres derived from MAESTRO composer/era metadata
PROXY_GENRES = [
    "Baroque",
    "Classical",
    "Romantic",
    "Impressionist_Modern"
]


# 5. GENERAL TRAINING HYPERPARAMETERS

BATCH_SIZE = 64
LEARNING_RATE = 1e-3
EPOCHS_AUTOENCODER = 50
EPOCHS_TRANSFORMER = 100