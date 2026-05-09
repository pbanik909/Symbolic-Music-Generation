import os
import numpy as np
import pretty_midi
import warnings
from pathlib import Path

# Import central configuration
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import FS, START_PITCH, END_PITCH, WINDOW_LENGTH, SPARSITY_THRESHOLD

# Suppress pretty_midi warnings about tempo changes
warnings.filterwarnings("ignore", category=RuntimeWarning)

def midi_to_pianoroll(midi_path: str | Path) -> np.ndarray:
    """
    Parses a MIDI file and converts it to a binarized piano-roll matrix.
    
    Args:
        midi_path: Path to the .mid file.
        
    Returns:
        np.ndarray: Binarized piano-roll of shape (T, 88), where T is time steps.
    """
    try:
        midi_data = pretty_midi.PrettyMIDI(str(midi_path))
        
        # Extract piano roll using our fixed sampling frequency (fs=16)
        # Returns shape: (128 pitches, T time steps)
        piano_roll = midi_data.get_piano_roll(fs=FS)
        
        # Filter strictly to standard 88 piano keys (MIDI 21 to 108)
        # Shape becomes: (88, T)
        piano_roll = piano_roll[START_PITCH:END_PITCH + 1, :]
        
        # Transpose to (T, 88) as sequence models expect (Time, Features)
        piano_roll = piano_roll.T
        
        # Binarize: Set all non-zero values (velocity) to 1 to simplify the learning problem
        piano_roll[piano_roll > 0] = 1
        
        return piano_roll.astype(np.float32)
        
    except Exception as e:
        print(f"Error processing {midi_path}: {e}")
        return np.array([])

def segment_and_filter(piano_roll: np.ndarray) -> np.ndarray:
    """
    Segments the continuous piano roll into fixed-length windows and filters out sparse ones.
    
    Args:
        piano_roll: Full track piano roll of shape (T, 88)
        
    Returns:
        np.ndarray: Array of valid windows, shape (N_valid_windows, WINDOW_LENGTH, 88)
    """
    if len(piano_roll) < WINDOW_LENGTH:
        return np.array([])
    
    # Calculate how many full windows we can extract
    n_windows = len(piano_roll) // WINDOW_LENGTH
    
    # Truncate any trailing time steps that don't fit into a full window
    truncated_roll = piano_roll[:n_windows * WINDOW_LENGTH, :]
    
    # Reshape into (N_windows, WINDOW_LENGTH, 88)
    windows = truncated_roll.reshape(n_windows, WINDOW_LENGTH, -1)
    
    valid_windows = []
    
    # Apply the Sparsity Filter
    for window in windows:
        # Calculate percentage of active cells (since it's binary, mean = percentage of 1s)
        active_ratio = np.mean(window)
        
        if active_ratio >= SPARSITY_THRESHOLD:
            valid_windows.append(window)
            
    if not valid_windows:
        return np.array([])
        
    return np.stack(valid_windows)

def process_single_file(midi_path: str | Path) -> np.ndarray:
    """
    End-to-end wrapper for a single MIDI file.
    """
    pr = midi_to_pianoroll(midi_path)
    if pr.size == 0:
        return np.array([])
    return segment_and_filter(pr)