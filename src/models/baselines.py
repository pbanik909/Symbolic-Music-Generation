import numpy as np
import pretty_midi
from collections import defaultdict
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from src.config import START_PITCH, END_PITCH, FS

class RandomNoteGenerator:
    """
    Generates music by uniformly sampling pitches and durations.
    Acts as the absolute lower bound for all metrics.
    """
    def __init__(self, start_pitch=START_PITCH, end_pitch=END_PITCH, fs=FS):
        self.start_pitch = start_pitch
        self.end_pitch = end_pitch
        self.fs = fs
        self.durations = [0.25, 0.5, 1.0] # fixed set of duration seconds

    def generate(self, num_notes=50, output_path=None):
        midi = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)

        current_time = 0.0
        for _ in range(num_notes):
            # Uniformly sample pitch and duration
            pitch = np.random.randint(self.start_pitch, self.end_pitch + 1)
            duration = np.random.choice(self.durations)
            
            note = pretty_midi.Note(
                velocity=80, # Fixed velocity
                pitch=pitch,
                start=current_time,
                end=current_time + duration
            )
            piano.notes.append(note)
            current_time += duration

        midi.instruments.append(piano)
        if output_path:
            midi.write(str(output_path))
        return midi

class MarkovChainMusic:
    """
    A First-Order Markov Chain that learns transition probabilities between pitches.
    Captures local melodic dependencies but lacks long-range structure.
    """
    def __init__(self, start_pitch=START_PITCH, end_pitch=END_PITCH):
        self.start_pitch = start_pitch
        self.end_pitch = end_pitch
        self.transition_counts = defaultdict(lambda: defaultdict(int))
        self.transition_probs = defaultdict(dict)
        self.durations = [0.25, 0.5] # standard durations for simplicity
        
    def fit(self, sequences):
        """
        Builds the transition matrix by counting pitch transitions.
        Args: sequences: List of pitch lists (e.g., [[60, 62, 64], [55, 59, ...]])
        """
        for seq in sequences:
            for i in range(len(seq) - 1):
                current_pitch = seq[i]
                next_pitch = seq[i+1]
                self.transition_counts[current_pitch][next_pitch] += 1
                
        # Normalize to obtain probabilities
        for current_pitch, next_counts in self.transition_counts.items():
            total_transitions = sum(next_counts.values())
            for next_pitch, count in next_counts.items():
                self.transition_probs[current_pitch][next_pitch] = count / total_transitions
                
    def generate(self, start_note=60, num_notes=50, output_path=None):
        midi = pretty_midi.PrettyMIDI()
        piano = pretty_midi.Instrument(program=0) # Acoustic Grand Piano
        
        current_time = 0.0
        current_pitch = start_note
        
        for _ in range(num_notes):
            duration = np.random.choice(self.durations)
            note = pretty_midi.Note(velocity=80, pitch=current_pitch, start=current_time, end=current_time + duration)
            piano.notes.append(note)
            current_time += duration
            
            # Sample next pitch
            if current_pitch in self.transition_probs and self.transition_probs[current_pitch]:
                next_pitches = list(self.transition_probs[current_pitch].keys())
                probs = list(self.transition_probs[current_pitch].values())
                current_pitch = np.random.choice(next_pitches, p=probs)
            else:
                # Fallback to random if state has no transitions
                current_pitch = np.random.randint(self.start_pitch, self.end_pitch + 1)

        midi.instruments.append(piano)
        if output_path:
            midi.write(str(output_path))
        return midi