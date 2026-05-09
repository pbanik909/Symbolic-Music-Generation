import numpy as np
import pretty_midi
from src.config import FS, WINDOW_LEN, PITCH_MIN, PITCH_MAX, SPARSITY_THRESHOLD

def process_midi_to_windows(midi_path):
    """
    Parses a MIDI file into binarized, fixed-length piano roll windows.
    Applies a sparsity filter to discard near-silent windows.
    """
    try:
        # Load MIDI using pretty_midi
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Extract piano roll at specified sampling frequency (FS = 16)
        # Returns shape: (128 pitches, Time_steps)
        pr = midi_data.get_piano_roll(fs=FS)
        
        # Slice to the standard 88 piano keys (MIDI 21 to 108)
        pr = pr
        
        # Transpose to (Time_steps, 88 pitches) to match PyTorch RNN convention
        pr = pr.T
        
        # Binarize: Discard velocity, set active notes to 1
        pr[pr > 0] = 1
        
        # Calculate how many full 128-step windows we can extract
        num_windows = pr.shape // WINDOW_LEN
        if num_windows == 0:
            return
            
        # Truncate the end to fit perfectly into windows, then reshape
        pr = pr
        windows = pr.reshape(num_windows, WINDOW_LEN, -1)
        
        valid_windows = []
        cells_per_window = WINDOW_LEN * (PITCH_MAX - PITCH_MIN + 1)
        
        # Sparsity Filter: Discard windows with < 2% active cells
        for w in windows:
            active_cells = np.count_nonzero(w)
            if (active_cells / cells_per_window) >= SPARSITY_THRESHOLD:
                valid_windows.append(w)
                
        return valid_windows

    except Exception as e:
        print(f"Failed to process {midi_path}: {e}")
        return