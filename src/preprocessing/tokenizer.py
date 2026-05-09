from miditok import REMI, TokenizerConfig
from pathlib import Path
from src.config import START_PITCH, END_PITCH

def get_remi_tokenizer():
    """
    Creates and returns a MidiTok REMI tokenizer configured for our project.
    """
    # Configure tokenizer to match our 88-key constraint and standard REMI events
    tokenizer_params = {
        "pitch_range": (START_PITCH, END_PITCH),
        "beat_res": {(0, 4): 8, (4, 12): 4},  # standard timing resolutions
        "num_velocities": 32,                 # quantize velocity into 32 bins
        "special_tokens":[],
        "use_chords": False,                  # keep it simple for baseline tasks
        "use_rests": False,
        "use_tempos": False,
        "use_time_signatures": False,
        "use_programs": False,
    }
    
    config = TokenizerConfig(**tokenizer_params)
    tokenizer = REMI(config)
    return tokenizer

def tokenize_midi(midi_path, tokenizer):
    """
    Converts a MIDI file into a sequence of integer tokens.
    """
    try:
        tokens = tokenizer(Path(midi_path))
        # MidiTok may return multiple tracks; we assume 1 track for MAESTRO piano
        if isinstance(tokens, list) and len(tokens) > 0:
            return tokens.ids
        return tokens.ids
    except Exception as e:
        print(f"Failed to tokenize {midi_path}: {e}")
        return

# Alias for compatibility with notebook imports
get_tokenizer = get_remi_tokenizer